"""
Stage 4D: Next-Generation Temporal & Structural Graph Architectures for Fraud Detection
Isolated experimental models — does not modify Stages 1-4C.
Capstone Project: Temporal Graph Neural Networks with Real-Time Explainable AI
for Online Payment Fraud Ring Detection
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, degree, to_undirected


# =============================================================================
# 1. SPARSE ADJACENCY & REGULARIZATION HELPERS
# =============================================================================

def build_gcn_sparse_adj(edge_index, num_nodes, use_undirected=False, drop_edge_p=0.0, device=None):
    """
    Constructs symmetrically normalized sparse adjacency D^{-1/2} (A + I) D^{-1/2}
    for an isolated snapshot. Optional DropEdge is applied only when drop_edge_p > 0.
    """
    ei = edge_index
    if drop_edge_p > 0.0 and ei.shape[1] > 0:
        keep_mask = torch.rand(ei.shape[1], device=ei.device) >= drop_edge_p
        if keep_mask.any():
            ei = ei[:, keep_mask]

    if use_undirected:
        ei = to_undirected(ei, num_nodes=num_nodes)
    ei, _ = add_self_loops(ei, num_nodes=num_nodes)

    row, col = ei[0], ei[1]
    deg = degree(col, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    adj = torch.sparse_coo_tensor(
        ei, edge_weight, size=(num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()
    return adj.to(device) if device is not None else adj


def build_neighbor_mean_sparse_adj(edge_index, num_nodes, use_undirected=False, drop_edge_p=0.0, device=None):
    """
    Constructs row-normalized neighbor-ONLY sparse aggregation matrix (NO self-loops)
    for GraphSAGE-style separated [self || neighbor] message passing:
    (adj_neigh @ H)[v] = mean_{u -> v} H[u], zero vector if in-degree == 0.
    Note: PyG edge_index[0]=src, edge_index[1]=dst, so row=dst, col=src for sparse.mm.
    """
    ei = edge_index
    if drop_edge_p > 0.0 and ei.shape[1] > 0:
        keep_mask = torch.rand(ei.shape[1], device=ei.device) >= drop_edge_p
        if keep_mask.any():
            ei = ei[:, keep_mask]

    if use_undirected:
        ei = to_undirected(ei, num_nodes=num_nodes)

    src, dst = ei[0], ei[1]
    in_deg = degree(dst, num_nodes=num_nodes, dtype=torch.float32)
    deg_inv = torch.where(in_deg > 0, 1.0 / in_deg, torch.zeros_like(in_deg))
    edge_weight = deg_inv[dst]

    # Message flows src -> dst, so sparse matrix entry is (row=dst, col=src)
    mat_idx = torch.stack([dst, src], dim=0)
    adj = torch.sparse_coo_tensor(
        mat_idx, edge_weight, size=(num_nodes, num_nodes), dtype=torch.float32
    ).coalesce()
    return adj.to(device) if device is not None else adj


# =============================================================================
# 2. CLASS IMBALANCE LOSSES (VERIFIED POSITIVE CLASS = 1 = ILLICIT)
# =============================================================================

class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss with positive class = 1 (Illicit / Fraud).
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t),
    where alpha_t = alpha for y=1 (Illicit) and (1 - alpha) for y=0 (Licit).
    Setting alpha > 0.5 upweights minority Illicit errors relative to Licit errors.
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits, targets):
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_factor = torch.pow(torch.clamp(1.0 - p_t, min=1e-6), self.gamma)
        return (alpha_t * focal_factor * bce).mean()


# =============================================================================
# 3. STABILIZED RESIDUAL & WEIGHT-BOUNDED BUILDING BLOCKS
# =============================================================================

class DualNormGatedResidualBlock(nn.Module):
    """
    Normalizes graph convolution output and residual skip output independently
    before gated combination, preventing high-norm conv activations from
    drowning out local features (Stage 4C audit fix).
    """
    def __init__(self, in_channels, out_channels, dropout_rate=0.3, mode="gated"):
        super().__init__()
        self.mode = mode
        self.ln_conv = nn.LayerNorm(out_channels)
        self.ln_skip = nn.LayerNorm(out_channels)
        self.skip_proj = nn.Linear(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout_rate)

        if mode == "gated":
            self.gate_logit = nn.Parameter(torch.tensor(0.0))
        elif mode == "concat":
            self.mix_proj = nn.Linear(out_channels * 2, out_channels)
        elif mode != "equal":
            raise ValueError(f"Unknown residual mode: {mode}")

    def forward(self, h_conv, x_in):
        c_norm = self.ln_conv(h_conv)
        s_norm = self.ln_skip(self.skip_proj(x_in))

        if self.mode == "gated":
            g = torch.sigmoid(self.gate_logit)
            combined = g * c_norm + (1.0 - g) * s_norm
        elif self.mode == "concat":
            combined = self.mix_proj(torch.cat([s_norm, c_norm], dim=-1))
        else:
            combined = 0.5 * (c_norm + s_norm)

        out = F.relu(combined)
        return self.dropout(out), c_norm.detach(), s_norm.detach()


class TopKSummarizer(nn.Module):
    """Compresses [N_t, d_in] node representations into [k, d_in] summary."""
    def __init__(self, in_channels, k):
        super().__init__()
        self.k = k
        self.proj = nn.Parameter(torch.empty(in_channels, 1))
        nn.init.xavier_uniform_(self.proj)

    def forward(self, h):
        num_nodes = h.shape[0]
        norm = torch.norm(self.proj, p=2) + 1e-8
        scores = (h @ self.proj).squeeze(-1) / norm
        if num_nodes >= self.k:
            top_vals, top_idx = torch.topk(scores, k=self.k, sorted=True)
        else:
            sorted_vals, sorted_idx = torch.sort(scores, descending=True)
            rep = (self.k + num_nodes - 1) // num_nodes
            top_vals = sorted_vals.repeat(rep)[: self.k]
            top_idx = sorted_idx.repeat(rep)[: self.k]
        gates = torch.tanh(top_vals).unsqueeze(-1)
        return h[top_idx] * gates


def bound_weight_matrix(w, max_fro_norm=15.0):
    """Smooth Frobenius-norm ceiling preventing temporal weight-norm explosion."""
    if max_fro_norm is None or max_fro_norm <= 0:
        return w
    norm = torch.norm(w, p="fro")
    scale = torch.clamp(norm / max_fro_norm, min=1.0)
    return w / scale


class StabilizedEvolveGCNLayer(nn.Module):
    """
    Unified stabilized EvolveGCN layer supporting:
      - variant='H': GRU input is TopKSummarizer(H_t) (EvolveGCN-H)
      - variant='O': GRU input is W_{t-1}^T itself, independent of H_t (EvolveGCN-O)
    Includes Frobenius norm bounding and optional residual anchoring toward W_init.
    """
    def __init__(self, in_channels, out_channels, variant="H", max_fro_norm=15.0, anchor_alpha=1.0):
        super().__init__()
        if variant not in ("H", "O"):
            raise ValueError(f"variant must be 'H' or 'O', got {variant}")
        self.variant = variant
        self.max_fro_norm = max_fro_norm
        self.anchor_alpha = float(anchor_alpha)

        if variant == "H":
            self.summarizer = TopKSummarizer(in_channels, k=out_channels)
        else:
            self.summarizer = None

        self.gru_cell = nn.GRUCell(input_size=in_channels, hidden_size=in_channels)
        self.W_init = nn.Parameter(torch.empty(in_channels, out_channels))
        nn.init.xavier_uniform_(self.W_init)

    def initial_weight(self):
        return bound_weight_matrix(self.W_init, self.max_fro_norm)

    def forward(self, h, adj_norm, w_prev):
        w_prev_T = w_prev.t()
        gru_in = self.summarizer(h) if self.variant == "H" else w_prev_T
        w_gru = self.gru_cell(gru_in, w_prev_T).t()

        if self.anchor_alpha < 1.0:
            w_t = (1.0 - self.anchor_alpha) * self.W_init + self.anchor_alpha * w_gru
        else:
            w_t = w_gru

        w_t = bound_weight_matrix(w_t, self.max_fro_norm)
        out = torch.sparse.mm(adj_norm, h @ w_t)
        return out, w_t


# =============================================================================
# 4. EXPERIMENT 1 & 3 MODELS: STABILIZED EVOLVEGCN-H AND EVOLVEGCN-O
# =============================================================================

class StabilizedEvolveGCN(nn.Module):
    """
    2-Layer Stabilized EvolveGCN-H or EvolveGCN-O with Dual-Branch LayerNorm,
    gated/concatenated residual skip preservation, and bounded Frobenius weights.
    """
    def __init__(
        self,
        in_channels=182,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        variant="H",
        residual_mode="gated",
        max_fro_norm=15.0,
        anchor_alpha=0.75,
    ):
        super().__init__()
        self.variant = variant
        self.egcn1 = StabilizedEvolveGCNLayer(
            in_channels, hidden_channels, variant=variant, max_fro_norm=max_fro_norm, anchor_alpha=anchor_alpha
        )
        self.res1 = DualNormGatedResidualBlock(
            in_channels, hidden_channels, dropout_rate=dropout_rate, mode=residual_mode
        )
        self.egcn2 = StabilizedEvolveGCNLayer(
            hidden_channels, out_channels, variant=variant, max_fro_norm=max_fro_norm, anchor_alpha=anchor_alpha
        )
        self.res2 = DualNormGatedResidualBlock(
            hidden_channels, out_channels, dropout_rate=dropout_rate * 0.7, mode=residual_mode
        )
        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_state(self):
        return (self.egcn1.initial_weight(), self.egcn2.initial_weight())

    def forward_step(self, x, adj_norm, state):
        w1_prev, w2_prev = state
        h1_conv, w1_next = self.egcn1(x, adj_norm, w1_prev)
        h1, c1_norm, s1_norm = self.res1(h1_conv, x)

        h2_conv, w2_next = self.egcn2(h1, adj_norm, w2_prev)
        h2, _, _ = self.res2(h2_conv, h1)

        logits = self.head(h2).squeeze(-1)
        telemetry = {
            "w1_norm": float(w1_next.detach().norm(p="fro").item()),
            "w2_norm": float(w2_next.detach().norm(p="fro").item()),
            "h1_conv_norm": float(h1_conv.detach().norm(dim=1).mean().item()),
            "h1_skip_norm": float(self.res1.skip_proj(x).detach().norm(dim=1).mean().item()),
        }
        return logits, (w1_next, w2_next), telemetry


# =============================================================================
# 5. EXPERIMENT 4 MODELS: STRICTLY CAUSAL TEMPORAL ATTENTION ARCHITECTURES
# =============================================================================

class CausalTemporalAttnGCN(nn.Module):
    """
    Model 4A: Dual-Norm Residual GCN + Strictly Causal Historical Snapshot Attention + FiLM.
    At step t:
      - Spatial backbone encodes X_t, A_t -> H_t.
      - Query summary q_t = MeanMaxPool(H_t) attends ONLY to causal prior summaries
        {g_{max(0, t-W)}, ..., g_{t-1}} (plus learned t=0 prior token) with relative lag embeddings.
      - FiLM modulation applies bounded scale/shift (1 + 0.5*tanh(gamma)) * LN(H_t) + 0.5*tanh(beta).
      - History bank is updated AFTER step t logits are computed (strictly causal).
    """
    def __init__(
        self,
        in_channels=182,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        window_size=8,
        num_heads=4,
        residual_mode="gated",
    ):
        super().__init__()
        self.window_size = window_size
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.summary_dim = out_channels * 2  # mean + max

        self.conv_w1 = nn.Linear(in_channels, hidden_channels, bias=False)
        self.res1 = DualNormGatedResidualBlock(in_channels, hidden_channels, dropout_rate=dropout_rate, mode=residual_mode)

        self.conv_w2 = nn.Linear(hidden_channels, out_channels, bias=False)
        self.res2 = DualNormGatedResidualBlock(hidden_channels, out_channels, dropout_rate=dropout_rate * 0.7, mode=residual_mode)

        self.init_summary = nn.Parameter(torch.zeros(1, self.summary_dim))
        self.lag_emb = nn.Embedding(window_size + 2, self.summary_dim)

        self.attn = nn.MultiheadAttention(
            embed_dim=self.summary_dim, num_heads=num_heads, dropout=dropout_rate * 0.5, batch_first=True
        )
        self.attn_ln = nn.LayerNorm(self.summary_dim)
        self.film_gamma = nn.Linear(self.summary_dim, out_channels)
        self.film_beta = nn.Linear(self.summary_dim, out_channels)
        self.post_ln = nn.LayerNorm(out_channels)

        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_state(self):
        # State is a tuple of strictly historical snapshot summaries (empty at t=0)
        return ()

    def forward_step(self, x, adj_norm, history_bank):
        h1_conv = torch.sparse.mm(adj_norm, self.conv_w1(x))
        h1, _, _ = self.res1(h1_conv, x)

        h2_conv = torch.sparse.mm(adj_norm, self.conv_w2(h1))
        h2, _, _ = self.res2(h2_conv, h1)

        # Unsupervised snapshot summary for step t
        g_mean = h2.mean(dim=0, keepdim=True)
        g_max = h2.max(dim=0, keepdim=True)[0]
        g_t = torch.cat([g_mean, g_max], dim=-1)  # [1, summary_dim]

        # Build strictly causal Keys/Values from prior steps ONLY (< t)
        if len(history_bank) == 0:
            hist_seq = self.init_summary  # [1, summary_dim]
            lags = torch.zeros(1, dtype=torch.long, device=x.device)
        else:
            recent = history_bank[-self.window_size :]
            hist_seq = torch.cat(recent, dim=0)  # [L, summary_dim]
            L = hist_seq.shape[0]
            lags = torch.arange(L, 0, -1, dtype=torch.long, device=x.device)

        k_seq = (hist_seq + self.lag_emb(lags)).unsqueeze(0)  # [1, L, summary_dim]
        v_seq = hist_seq.unsqueeze(0)
        q_t = g_t.unsqueeze(0)  # [1, 1, summary_dim]

        ctx, _ = self.attn(q_t, k_seq, v_seq)
        ctx = self.attn_ln(ctx.squeeze(0))  # [1, summary_dim]

        gamma = 0.5 * torch.tanh(self.film_gamma(ctx))  # [1, out_channels]
        beta = 0.5 * torch.tanh(self.film_beta(ctx))
        h2_mod = (1.0 + gamma) * self.post_ln(h2) + beta

        logits = self.head(h2_mod).squeeze(-1)

        # Update causal bank strictly AFTER logits
        next_bank = tuple(list(history_bank[-(self.window_size - 1) :]) + [g_t]) if self.window_size > 1 else (g_t,)
        telemetry = {
            "w1_norm": float(self.conv_w1.weight.detach().norm(p="fro").item()),
            "w2_norm": float(self.conv_w2.weight.detach().norm(p="fro").item()),
            "h1_conv_norm": float(h1_conv.detach().norm(dim=1).mean().item()),
            "h1_skip_norm": float(self.res1.skip_proj(x).detach().norm(dim=1).mean().item()),
        }
        return logits, next_bank, telemetry


class CausalAttnEvolveGCN(nn.Module):
    """
    Model 4B: Causal Attention over Historical Weight Bank + Stabilized EvolveGCN-H.
    At step t:
      - Summarizes X_t via TopK and computes attention weights over causal past
        weight matrices {W_{t-W}, ..., W_{t-1}} before running the GRU step.
    """
    def __init__(
        self,
        in_channels=182,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        window_size=6,
        max_fro_norm=15.0,
        residual_mode="gated",
    ):
        super().__init__()
        self.window_size = window_size
        self.max_fro_norm = max_fro_norm

        self.egcn1 = StabilizedEvolveGCNLayer(
            in_channels, hidden_channels, variant="H", max_fro_norm=max_fro_norm, anchor_alpha=0.75
        )
        self.res1 = DualNormGatedResidualBlock(in_channels, hidden_channels, dropout_rate=dropout_rate, mode=residual_mode)
        self.egcn2 = StabilizedEvolveGCNLayer(
            hidden_channels, out_channels, variant="H", max_fro_norm=max_fro_norm, anchor_alpha=0.75
        )
        self.res2 = DualNormGatedResidualBlock(hidden_channels, out_channels, dropout_rate=dropout_rate * 0.7, mode=residual_mode)

        # Lightweight bilinear/dot attention scorer over causal weight bank using layer-1 summary
        self.w_query1 = nn.Linear(in_channels, in_channels)
        self.w_key1 = nn.Linear(in_channels, in_channels)
        self.w_query2 = nn.Linear(hidden_channels, hidden_channels)
        self.w_key2 = nn.Linear(hidden_channels, hidden_channels)

        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_state(self):
        w1_0 = self.egcn1.initial_weight()
        w2_0 = self.egcn2.initial_weight()
        return ([w1_0], [w2_0])

    def _attend_weights(self, h_in, w_hist_list, query_proj, key_proj, summarizer):
        if len(w_hist_list) == 1:
            return w_hist_list[0]
        z_cur = summarizer(h_in).mean(dim=0, keepdim=True)  # [1, d_in]
        q = query_proj(z_cur)  # [1, d_in]
        w_stack = torch.stack(w_hist_list, dim=0)  # [L, d_in, d_out]
        w_summaries = w_stack.mean(dim=-1)  # [L, d_in]
        k = key_proj(w_summaries)  # [L, d_in]
        scores = (q @ k.t()) / math.sqrt(k.shape[-1])  # [1, L]
        alphas = F.softmax(scores, dim=-1).view(-1, 1, 1)
        return (alphas * w_stack).sum(dim=0)

    def forward_step(self, x, adj_norm, state):
        w1_bank, w2_bank = state
        w1_prior = self._attend_weights(x, w1_bank, self.w_query1, self.w_key1, self.egcn1.summarizer)
        h1_conv, w1_next = self.egcn1(x, adj_norm, w1_prior)
        h1, _, _ = self.res1(h1_conv, x)

        w2_prior = self._attend_weights(h1, w2_bank, self.w_query2, self.w_key2, self.egcn2.summarizer)
        h2_conv, w2_next = self.egcn2(h1, adj_norm, w2_prior)
        h2, _, _ = self.res2(h2_conv, h1)

        logits = self.head(h2).squeeze(-1)
        next_w1_bank = (w1_bank + [w1_next])[-self.window_size :]
        next_w2_bank = (w2_bank + [w2_next])[-self.window_size :]
        telemetry = {
            "w1_norm": float(w1_next.detach().norm(p="fro").item()),
            "w2_norm": float(w2_next.detach().norm(p="fro").item()),
            "h1_conv_norm": float(h1_conv.detach().norm(dim=1).mean().item()),
            "h1_skip_norm": float(self.res1.skip_proj(x).detach().norm(dim=1).mean().item()),
        }
        return logits, (next_w1_bank, next_w2_bank), telemetry


# =============================================================================
# 6. EXPERIMENT 6 MODEL: GRAPH VALUE-ADDED TEST (A=I vs GCN-Avg vs SAGE-Concat)
# =============================================================================

class SeparatedGraphNet(nn.Module):
    """
    Controlled architecture for Experiment 6 testing whether graph message passing
    adds value beyond tabular features (93 Local-Only vs 182 Full Elliptic++):
      - graph_mode='no_graph'   : Pure MLP baseline (self-projection only, A = I)
      - graph_mode='gcn_avg'    : Symmetric GCNConv averaging + DualNormGatedResidualBlock
      - graph_mode='sage_concat': GraphSAGE-style [LN(W_self h) || LN(A_neigh h W_neigh)],
                                  keeping node self-features unmixed with neighbor features.
    """
    def __init__(
        self,
        in_channels=182,
        hidden_channels=128,
        out_channels=64,
        dropout_rate=0.3,
        graph_mode="sage_concat",
    ):
        super().__init__()
        if graph_mode not in ("no_graph", "gcn_avg", "sage_concat"):
            raise ValueError(f"Invalid graph_mode: {graph_mode}")
        self.graph_mode = graph_mode
        self.dropout_rate = dropout_rate

        if graph_mode == "no_graph":
            self.net = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_channels, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout_rate * 0.7),
            )
        elif graph_mode == "gcn_avg":
            self.w1 = nn.Linear(in_channels, hidden_channels, bias=False)
            self.res1 = DualNormGatedResidualBlock(in_channels, hidden_channels, dropout_rate=dropout_rate, mode="gated")
            self.w2 = nn.Linear(hidden_channels, out_channels, bias=False)
            self.res2 = DualNormGatedResidualBlock(hidden_channels, out_channels, dropout_rate=dropout_rate * 0.7, mode="gated")
        else:  # sage_concat
            half_h = hidden_channels // 2
            half_o = out_channels // 2
            self.self1 = nn.Linear(in_channels, half_h)
            self.neigh1 = nn.Linear(in_channels, half_h, bias=False)
            self.ln_s1 = nn.LayerNorm(half_h)
            self.ln_n1 = nn.LayerNorm(half_h)
            self.mix1 = nn.Linear(hidden_channels, hidden_channels)
            self.ln_m1 = nn.LayerNorm(hidden_channels)

            self.self2 = nn.Linear(hidden_channels, half_o)
            self.neigh2 = nn.Linear(hidden_channels, half_o, bias=False)
            self.ln_s2 = nn.LayerNorm(half_o)
            self.ln_n2 = nn.LayerNorm(half_o)
            self.mix2 = nn.Linear(out_channels, out_channels)
            self.ln_m2 = nn.LayerNorm(out_channels)

        self.head = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(32, 1),
        )

    def init_state(self):
        return None

    def forward_step(self, x, adj_norm, state=None):
        if self.graph_mode == "no_graph":
            h2 = self.net(x)
            telemetry = {"w1_norm": 0.0, "w2_norm": 0.0, "h1_conv_norm": 0.0, "h1_skip_norm": float(h2.detach().norm(dim=1).mean().item())}
        elif self.graph_mode == "gcn_avg":
            h1_c = torch.sparse.mm(adj_norm, self.w1(x))
            h1, _, _ = self.res1(h1_c, x)
            h2_c = torch.sparse.mm(adj_norm, self.w2(h1))
            h2, _, _ = self.res2(h2_c, h1)
            telemetry = {
                "w1_norm": float(self.w1.weight.detach().norm(p="fro").item()),
                "w2_norm": float(self.w2.weight.detach().norm(p="fro").item()),
                "h1_conv_norm": float(h1_c.detach().norm(dim=1).mean().item()),
                "h1_skip_norm": float(self.res1.skip_proj(x).detach().norm(dim=1).mean().item()),
            }
        else:
            s1 = self.ln_s1(self.self1(x))
            n1 = self.ln_n1(torch.sparse.mm(adj_norm, self.neigh1(x)))
            h1 = F.relu(self.ln_m1(self.mix1(torch.cat([s1, n1], dim=-1))))
            h1 = F.dropout(h1, p=self.dropout_rate, training=self.training)

            s2 = self.ln_s2(self.self2(h1))
            n2 = self.ln_n2(torch.sparse.mm(adj_norm, self.neigh2(h1)))
            h2 = F.relu(self.ln_m2(self.mix2(torch.cat([s2, n2], dim=-1))))
            h2 = F.dropout(h2, p=self.dropout_rate * 0.7, training=self.training)
            telemetry = {
                "w1_norm": float(self.neigh1.weight.detach().norm(p="fro").item()),
                "w2_norm": float(self.neigh2.weight.detach().norm(p="fro").item()),
                "h1_conv_norm": float(n1.detach().norm(dim=1).mean().item()),
                "h1_skip_norm": float(s1.detach().norm(dim=1).mean().item()),
            }

        logits = self.head(h2).squeeze(-1)
        return logits, state, telemetry
