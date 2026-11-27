"""
Stage 4F: Graph Model Performance Optimization Architectures & Verified Adjacency Builders
Strictly isolated from Stages 1-4E; preserves Stage 4D SeparatedGraphNet anchor untouched.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import add_self_loops, degree, to_undirected, scatter

from src.models_nextgen import BinaryFocalLoss, SeparatedGraphNet, build_gcn_sparse_adj


# =============================================================================
# 1. VERIFIED DIRECTIONAL & SYMMETRIC SPARSE ADJACENCY BUILDERS
# =============================================================================

def _apply_drop_edge(edge_index, drop_edge_p):
    if drop_edge_p > 0.0 and edge_index.shape[1] > 0:
        keep_mask = torch.rand(edge_index.shape[1], device=edge_index.device) >= drop_edge_p
        if keep_mask.any():
            return edge_index[:, keep_mask]
    return edge_index


def build_true_directed_gcn_adj(edge_index, num_nodes, direction="in", drop_edge_p=0.0, device=None):
    """
    Constructs properly transposed sparse GCN adjacency D_row^{-1/2} (A + I) D_col^{-1/2}
    for torch.sparse.mm(adj, H):
      - direction='in'         : messages flow src -> dst (mat row=dst, col=src), node i receives parents
      - direction='out'        : messages flow dst -> src (mat row=src, col=dst), node i receives children
      - direction='undirected' : symmetric union of both directions + self-loops
    Isolated nodes retain exact self-loop weight 1.0 (zero NaNs/infs).
    """
    ei = _apply_drop_edge(edge_index, drop_edge_p)
    if direction == "undirected":
        ei_dir = to_undirected(ei, num_nodes=num_nodes)
        mat_row, mat_col = ei_dir[0], ei_dir[1]
    elif direction == "in":
        mat_row, mat_col = ei[1], ei[0]
    elif direction == "out":
        mat_row, mat_col = ei[0], ei[1]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    mat_ei = torch.stack([mat_row, mat_col], dim=0)
    mat_ei, _ = add_self_loops(mat_ei, num_nodes=num_nodes)
    row, col = mat_ei[0], mat_ei[1]

    deg_row = degree(row, num_nodes=num_nodes, dtype=torch.float32)
    deg_col = degree(col, num_nodes=num_nodes, dtype=torch.float32)
    deg_row_inv_sqrt = torch.where(deg_row > 0, deg_row.pow(-0.5), torch.zeros_like(deg_row))
    deg_col_inv_sqrt = torch.where(deg_col > 0, deg_col.pow(-0.5), torch.zeros_like(deg_col))
    edge_weight = deg_row_inv_sqrt[row] * deg_col_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        mat_ei, edge_weight, size=(num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()
    return adj.to(device) if device is not None else adj


def build_directed_neighbor_mean_adj(edge_index, num_nodes, direction="in", drop_edge_p=0.0, device=None):
    """
    Row-normalized neighbor-only sparse matrix (NO self-loops):
      - direction='in' : row=dst, col=src -> (adj @ H)[v] = mean_{u -> v} H[u]
      - direction='out': row=src, col=dst -> (adj @ H)[u] = mean_{u -> v} H[v]
    Zero-degree / isolated nodes produce exact 0.0 vectors.
    """
    ei = _apply_drop_edge(edge_index, drop_edge_p)
    if direction == "in":
        row, col = ei[1], ei[0]
    elif direction == "out":
        row, col = ei[0], ei[1]
    else:
        raise ValueError(f"Unknown direction: {direction}")

    deg_row = degree(row, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv = torch.where(deg_row > 0, 1.0 / deg_row, torch.zeros_like(deg_row))
    edge_weight = deg_inv[row]

    mat_idx = torch.stack([row, col], dim=0)
    adj = torch.sparse_coo_tensor(
        mat_idx, edge_weight, size=(num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()
    return adj.to(device) if device is not None else adj


def compute_node_deg_features(edge_index, num_nodes, device=None):
    """Computes snapshot-local normalized structural degree features [log1p(in_deg), log1p(out_deg)]."""
    src, dst = edge_index[0], edge_index[1]
    in_deg = degree(dst, num_nodes=num_nodes, dtype=torch.float32)
    out_deg = degree(src, num_nodes=num_nodes, dtype=torch.float32)
    feats = torch.stack([torch.log1p(in_deg), torch.log1p(out_deg)], dim=-1)
    return feats.to(device) if device is not None else feats


# =============================================================================
# 2. BUILDING BLOCKS (SWIGLU TABULAR STEM & BIDIRECTIONAL LAYERS)
# =============================================================================

class SwiGLUResBlock(nn.Module):
    """Pre-LN SwiGLU residual block for tabular feature interactions."""
    def __init__(self, dim, hidden_mult=2, dropout_rate=0.25):
        super().__init__()
        inner = dim * hidden_mult
        self.ln = nn.LayerNorm(dim)
        self.w_gate = nn.Linear(dim, inner)
        self.w_val = nn.Linear(dim, inner)
        self.w_out = nn.Linear(inner, dim)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        h = self.ln(x)
        gated = F.silu(self.w_gate(h)) * self.w_val(h)
        out = self.w_out(self.drop(gated))
        return x + self.drop(out)


class BiDirGCNLayer(nn.Module):
    """Decoupled [Self || In-Neighbor-GCN || Out-Neighbor-GCN] layer with branch LayerNorm."""
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.self_proj = nn.Linear(in_channels, out_channels)
        self.in_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.out_proj = nn.Linear(in_channels, out_channels, bias=False)

        self.ln_self = nn.LayerNorm(out_channels)
        self.ln_in = nn.LayerNorm(out_channels)
        self.ln_out = nn.LayerNorm(out_channels)

        self.fuse = nn.Linear(out_channels * 3, out_channels)
        self.ln_fuse = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout_rate)

    def forward(self, x, adj_in, adj_out):
        s = self.ln_self(self.self_proj(x))
        h_in = self.ln_in(torch.sparse.mm(adj_in, self.in_proj(x)))
        h_out = self.ln_out(torch.sparse.mm(adj_out, self.out_proj(x)))
        fused = self.ln_fuse(self.fuse(torch.cat([s, h_in, h_out], dim=-1)))
        out = F.relu(fused + s)
        return self.drop(out)


class BiDirRelationalGCN(nn.Module):
    """
    Model F1/F2/F5/F6/F7: Decoupled Bidirectional Relational GCN.
    Supports optional monotonic asinh(x) tail compression, feature dropout,
    structural log-degree inputs, and a Pre-LN SwiGLU tabular stem.
    """
    def __init__(
        self,
        in_channels=93,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        use_tail_asinh=False,
        feat_dropout=0.0,
        use_deg_feats=False,
        stem_mode="none",
    ):
        super().__init__()
        self.use_tail_asinh = use_tail_asinh
        self.feat_dropout = feat_dropout
        self.use_deg_feats = use_deg_feats
        self.stem_mode = stem_mode

        eff_in = in_channels + (2 if use_deg_feats else 0)
        if stem_mode == "swiglu":
            self.stem = nn.Sequential(
                nn.Linear(eff_in, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                SwiGLUResBlock(hidden_channels, hidden_mult=2, dropout_rate=dropout_rate * 0.8),
            )
            conv1_in = hidden_channels
        else:
            self.stem = None
            conv1_in = eff_in

        self.layer1 = BiDirGCNLayer(conv1_in, hidden_channels, dropout_rate=dropout_rate)
        self.layer2 = BiDirGCNLayer(hidden_channels, out_channels, dropout_rate=dropout_rate * 0.7)
        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_state(self):
        return None

    def encode(self, x, graph_bundle):
        h = torch.asinh(x) if self.use_tail_asinh else x
        if self.feat_dropout > 0.0:
            h = F.dropout(h, p=self.feat_dropout, training=self.training)
        if self.use_deg_feats:
            h = torch.cat([h, graph_bundle["deg_feats"]], dim=-1)
        if self.stem is not None:
            h = self.stem(h)

        h1 = self.layer1(h, graph_bundle["gcn_in"], graph_bundle["gcn_out"])
        h2 = self.layer2(h1, graph_bundle["gcn_in"], graph_bundle["gcn_out"])
        return h2

    def forward_step(self, x, graph_bundle, state=None):
        h2 = self.encode(x, graph_bundle)
        logits = self.head(h2).squeeze(-1)
        return logits, state, {"layer_norm_h2": float(h2.detach().norm(dim=1).mean().item())}


class BiDirGraphSAGEMeanMaxLayer(nn.Module):
    """
    Decoupled [Self || InMean || OutMean || InMax || OutMax] with per-node channel gate.
    Zero-degree buckets are explicitly masked so isolated/boundary nodes produce clean 0.0.
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.3):
        super().__init__()
        self.self_proj = nn.Linear(in_channels, out_channels)
        self.in_mean_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.out_mean_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.in_max_proj = nn.Linear(in_channels, out_channels, bias=False)
        self.out_max_proj = nn.Linear(in_channels, out_channels, bias=False)

        self.ln_self = nn.LayerNorm(out_channels)
        self.ln_agg = nn.LayerNorm(out_channels)
        self.agg_mix = nn.Linear(out_channels * 4, out_channels)
        self.node_gate = nn.Linear(out_channels * 2, out_channels)
        self.ln_out = nn.LayerNorm(out_channels)
        self.drop = nn.Dropout(dropout_rate)

    def _masked_max_scatter(self, msg, index, num_nodes):
        if msg.shape[0] == 0:
            return torch.zeros(num_nodes, msg.shape[1], dtype=msg.dtype, device=msg.device)
        out = scatter(msg, index, dim=0, dim_size=num_nodes, reduce="max")
        has_edge = torch.bincount(index, minlength=num_nodes) > 0
        return out * has_edge.unsqueeze(-1).to(out.dtype)

    def forward(self, x, graph_bundle):
        ei = graph_bundle["edge_index"]
        src, dst = ei[0], ei[1]
        num_nodes = x.shape[0]

        s = self.ln_self(self.self_proj(x))
        in_mean = torch.sparse.mm(graph_bundle["mean_in"], self.in_mean_proj(x))
        out_mean = torch.sparse.mm(graph_bundle["mean_out"], self.out_mean_proj(x))

        in_max_msg = self.in_max_proj(x)[src]
        in_max = self._masked_max_scatter(in_max_msg, dst, num_nodes)
        out_max_msg = self.out_max_proj(x)[dst]
        out_max = self._masked_max_scatter(out_max_msg, src, num_nodes)

        agg = self.ln_agg(self.agg_mix(torch.cat([in_mean, out_mean, in_max, out_max], dim=-1)))
        gate = torch.sigmoid(self.node_gate(torch.cat([s, agg], dim=-1)))
        out = F.relu(self.ln_out(s + gate * agg))
        return self.drop(out)


class BiDirGraphSAGEMeanMax(nn.Module):
    """Model F3: 2-Layer Bidirectional GraphSAGE Mean+Max with Per-Node Channel Gating."""
    def __init__(self, in_channels=93, hidden_channels=128, out_channels=64, dropout_rate=0.3):
        super().__init__()
        self.layer1 = BiDirGraphSAGEMeanMaxLayer(in_channels, hidden_channels, dropout_rate=dropout_rate)
        self.layer2 = BiDirGraphSAGEMeanMaxLayer(hidden_channels, out_channels, dropout_rate=dropout_rate * 0.7)
        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_state(self):
        return None

    def forward_step(self, x, graph_bundle, state=None):
        h1 = self.layer1(x, graph_bundle)
        h2 = self.layer2(h1, graph_bundle)
        logits = self.head(h2).squeeze(-1)
        return logits, state, {"layer_norm_h2": float(h2.detach().norm(dim=1).mean().item())}


class BiDirGATv2Net(nn.Module):
    """
    Model F4: Bidirectional Multi-Head GATv2 with Dual-Branch LayerNorm & Residual Skips.
    Learns dynamic attention weights independently along incoming and outgoing edges.
    """
    def __init__(self, in_channels=93, hidden_channels=128, out_channels=64, dropout_rate=0.3, edge_dropout=0.1):
        super().__init__()
        self.drop_rate = dropout_rate
        h_per_head1 = hidden_channels // 4
        self.in_gat1 = GATv2Conv(in_channels, h_per_head1, heads=4, concat=True, dropout=edge_dropout, add_self_loops=True)
        self.out_gat1 = GATv2Conv(in_channels, h_per_head1, heads=4, concat=True, dropout=edge_dropout, add_self_loops=True)
        self.skip1 = nn.Linear(in_channels, hidden_channels)
        self.ln_s1 = nn.LayerNorm(hidden_channels)
        self.ln_i1 = nn.LayerNorm(hidden_channels)
        self.ln_o1 = nn.LayerNorm(hidden_channels)
        self.mix1 = nn.Linear(hidden_channels * 3, hidden_channels)
        self.ln_m1 = nn.LayerNorm(hidden_channels)

        h_per_head2 = out_channels // 2
        self.in_gat2 = GATv2Conv(hidden_channels, h_per_head2, heads=2, concat=True, dropout=edge_dropout, add_self_loops=True)
        self.out_gat2 = GATv2Conv(hidden_channels, h_per_head2, heads=2, concat=True, dropout=edge_dropout, add_self_loops=True)
        self.skip2 = nn.Linear(hidden_channels, out_channels)
        self.ln_s2 = nn.LayerNorm(out_channels)
        self.ln_i2 = nn.LayerNorm(out_channels)
        self.ln_o2 = nn.LayerNorm(out_channels)
        self.mix2 = nn.Linear(out_channels * 3, out_channels)
        self.ln_m2 = nn.LayerNorm(out_channels)

        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_state(self):
        return None

    def forward_step(self, x, graph_bundle, state=None):
        ei_in = graph_bundle["edge_index"]
        ei_out = graph_bundle["edge_index_rev"]

        s1 = self.ln_s1(self.skip1(x))
        i1 = self.ln_i1(self.in_gat1(x, ei_in))
        o1 = self.ln_o1(self.out_gat1(x, ei_out))
        h1 = F.elu(self.ln_m1(self.mix1(torch.cat([s1, i1, o1], dim=-1))) + s1)
        h1 = F.dropout(h1, p=self.drop_rate, training=self.training)

        s2 = self.ln_s2(self.skip2(h1))
        i2 = self.ln_i2(self.in_gat2(h1, ei_in))
        o2 = self.ln_o2(self.out_gat2(h1, ei_out))
        h2 = F.elu(self.ln_m2(self.mix2(torch.cat([s2, i2, o2], dim=-1))) + s2)
        h2 = F.dropout(h2, p=self.drop_rate * 0.7, training=self.training)

        logits = self.head(h2).squeeze(-1)
        return logits, state, {"layer_norm_h2": float(h2.detach().norm(dim=1).mean().item())}


class CausalCtxBiDirGCN(nn.Module):
    """
    Model F11: BiDirRelationalGCN + Strictly Causal Historical Snapshot Attention & FiLM.
    Attends ONLY to strictly prior summaries (< t) within window W; updates bank AFTER logits.
    """
    def __init__(self, in_channels=93, hidden_channels=128, out_channels=64, dropout_rate=0.3, window_size=6, num_heads=4):
        super().__init__()
        self.backbone = BiDirRelationalGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            dropout_rate=dropout_rate,
        )
        self.window_size = window_size
        self.summary_dim = out_channels * 2
        self.init_summary = nn.Parameter(torch.zeros(1, self.summary_dim))
        self.lag_emb = nn.Embedding(window_size + 2, self.summary_dim)
        self.attn = nn.MultiheadAttention(self.summary_dim, num_heads=num_heads, dropout=dropout_rate * 0.5, batch_first=True)
        self.attn_ln = nn.LayerNorm(self.summary_dim)
        self.film_gamma = nn.Linear(self.summary_dim, out_channels)
        self.film_beta = nn.Linear(self.summary_dim, out_channels)
        self.post_ln = nn.LayerNorm(out_channels)

    def init_state(self):
        return ()

    def forward_step(self, x, graph_bundle, history_bank):
        h2 = self.backbone.encode(x, graph_bundle)
        g_t = torch.cat([h2.mean(dim=0, keepdim=True), h2.max(dim=0, keepdim=True)[0]], dim=-1)

        if len(history_bank) == 0:
            hist_seq = self.init_summary
            lags = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            recent = history_bank[-self.window_size :]
            hist_seq = torch.cat(recent, dim=0)
            L = hist_seq.shape[0]
            lags = torch.arange(L, 0, -1, dtype=torch.long, device=x.device)

        k_seq = (hist_seq + self.lag_emb(lags)).unsqueeze(0)
        v_seq = hist_seq.unsqueeze(0)
        q_t = g_t.unsqueeze(0)
        ctx, _ = self.attn(q_t, k_seq, v_seq)
        ctx = self.attn_ln(ctx.squeeze(0))

        gamma = 0.5 * torch.tanh(self.film_gamma(ctx))
        beta = 0.5 * torch.tanh(self.film_beta(ctx))
        h2_mod = (1.0 + gamma) * self.post_ln(h2) + beta

        logits = self.backbone.head(h2_mod).squeeze(-1)
        next_bank = tuple(list(history_bank[-(self.window_size - 1) :]) + [g_t]) if self.window_size > 1 else (g_t,)
        return logits, next_bank, {"layer_norm_h2": float(h2_mod.detach().norm(dim=1).mean().item())}


class AnchorGCNAvgWrapper(nn.Module):
    """Wraps untouched Stage 4D SeparatedGraphNet('gcn_avg') with bundle interface."""
    def __init__(self, in_channels=93, adj_key="legacy_dir"):
        super().__init__()
        self.adj_key = adj_key
        self.core = SeparatedGraphNet(
            in_channels=in_channels,
            hidden_channels=128,
            out_channels=64,
            dropout_rate=0.3,
            graph_mode="gcn_avg",
        )

    def init_state(self):
        return self.core.init_state()

    def forward_step(self, x, graph_bundle, state=None):
        return self.core.forward_step(x, graph_bundle[self.adj_key], state)
