"""Stage 5C suspicious directed subgraph and fraud-ring candidate analysis service."""

from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np

from src.stage5.stage5b_individual_explanations import gt_label_str
from src.stage5.stage5c_fraud_ring_analysis import classify_directed_motif


def analyze_uploaded_snapshot_clusters(
    timestep: int,
    tx_ids: np.ndarray,
    src_local: np.ndarray,
    dst_local: np.ndarray,
    probs: np.ndarray,
    logits: np.ndarray,
    y_opt: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Identifies weakly connected predicted suspicious clusters (|C| >= 2) within a single
    temporal snapshot using strictly frozen model predictions and intra-snapshot directed edges.
    """
    num_nodes = len(tx_ids)
    in_deg = np.bincount(dst_local, minlength=num_nodes) if num_nodes > 0 else np.zeros(0, dtype=np.int64)
    out_deg = np.bincount(src_local, minlength=num_nodes) if num_nodes > 0 else np.zeros(0, dtype=np.int64)

    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for s_idx, d_idx in zip(src_local.tolist(), dst_local.tolist()):
        G.add_edge(int(s_idx), int(d_idx))

    sus_nodes = {u for u in range(num_nodes) if float(probs[u]) >= threshold}
    G_sus = G.subgraph(sus_nodes)

    wccs = sorted(
        [list(c) for c in nx.weakly_connected_components(G_sus) if len(c) >= 2],
        key=lambda c: (-len(c), -float(np.mean(probs[c])), min(int(tx_ids[u]) for u in c)),
    )

    cluster_rows: List[Dict[str, Any]] = []
    edge_rows: List[Dict[str, Any]] = []
    node_to_cid: Dict[int, str] = {}
    potential_candidates = 0

    for idx, nodes in enumerate(wccs, start=1):
        cid = f"C_T{timestep:02d}_{idx:03d}"
        for u in nodes:
            node_to_cid[u] = cid

        sub = G_sus.subgraph(nodes)
        n_nodes = len(nodes)
        n_edges = int(sub.number_of_edges())
        sub_probs = probs[nodes]
        mean_p = float(sub_probs.mean())
        max_p = float(sub_probs.max())
        min_p = float(sub_probs.min())
        n_high_conf = int((sub_probs >= 0.85).sum())

        motif = classify_directed_motif(sub)
        max_int_in = max((d for _, d in sub.in_degree()), default=0)
        max_int_out = max((d for _, d in sub.out_degree()), default=0)
        edge_density = float(n_edges / max(n_nodes * (n_nodes - 1), 1))
        has_cycle = not nx.is_directed_acyclic_graph(sub)
        try:
            max_dir_path = int(nx.dag_longest_path_length(sub)) if not has_cycle else int(n_edges)
        except Exception:
            max_dir_path = int(n_edges)

        # Tier A Ring Candidacy Rule (ZERO labels used)
        is_candidate = bool(n_nodes >= 3 and mean_p >= 0.75 and n_high_conf >= 2 and n_edges >= 2)
        if is_candidate:
            structural_tier = "Potential_Fraud_Ring_Candidate"
            potential_candidates += 1
        elif n_nodes >= 3:
            structural_tier = "Multi_Node_Suspicious_Cluster"
        else:
            structural_tier = "Suspicious_Pair"

        ring_score = float(mean_p * np.log2(1.0 + n_nodes) * (1.0 + (n_edges / float(n_nodes))))

        c_set = set(nodes)
        ext_in_nodes = set()
        ext_out_nodes = set()
        for u in nodes:
            ext_in_nodes.update(set(G.predecessors(u)) - c_set)
            ext_out_nodes.update(set(G.successors(u)) - c_set)

        ys = y_opt[nodes]
        gt_illicit = int((ys == 1).sum())
        gt_licit = int((ys == 0).sum())
        gt_unknown = int((ys == -1).sum())
        gt_labeled = gt_illicit + gt_licit
        post_hoc_prec = float(gt_illicit / gt_labeled) if gt_labeled > 0 else 0.0
        post_hoc_status = "Labeled_Mixed_or_Pure" if gt_labeled > 0 else "ALL_UNKNOWN_MINUS1"

        hub_u = sorted(
            nodes,
            key=lambda u: (-(sub.in_degree(u) + sub.out_degree(u)), -float(probs[u]), int(tx_ids[u])),
        )[0]

        cluster_rows.append(
            {
                "cluster_id": cid,
                "timestep": int(timestep),
                "structural_tier": structural_tier,
                "is_ring_candidate": int(is_candidate),
                "motif_type": motif,
                "ring_score": round(ring_score, 6),
                "num_nodes": int(n_nodes),
                "num_internal_edges": int(n_edges),
                "directed_edge_density": round(edge_density, 6),
                "has_directed_cycle": int(has_cycle),
                "max_directed_path_len": int(max_dir_path),
                "max_internal_in_deg": int(max_int_in),
                "max_internal_out_deg": int(max_int_out),
                "ext_1hop_in_neighbors": int(len(ext_in_nodes)),
                "ext_1hop_out_neighbors": int(len(ext_out_nodes)),
                "mean_pred_prob": round(mean_p, 6),
                "max_pred_prob": round(max_p, 6),
                "min_pred_prob": round(min_p, 6),
                "num_high_conf_ge_085": int(n_high_conf),
                "hub_tx_id": int(tx_ids[hub_u]),
                "hub_node_idx": int(hub_u),
                "hub_pred_prob": round(float(probs[hub_u]), 6),
                "post_hoc_gt_illicit": gt_illicit,
                "post_hoc_gt_licit": gt_licit,
                "post_hoc_gt_unknown": gt_unknown,
                "post_hoc_gt_labeled": gt_labeled,
                "has_labeled_gt": int(gt_labeled > 0),
                "post_hoc_labeled_illicit_precision": round(post_hoc_prec, 6),
                "post_hoc_eval_status": post_hoc_status,
            }
        )

        for e_src, e_dst in sorted(sub.edges(), key=lambda x: (int(tx_ids[x[0]]), int(tx_ids[x[1]]))):
            edge_rows.append(
                {
                    "cluster_id": cid,
                    "timestep": int(timestep),
                    "src_tx_id": int(tx_ids[e_src]),
                    "dst_tx_id": int(tx_ids[e_dst]),
                    "edge_type": "internal_suspicious",
                    "src_pred_prob": round(float(probs[e_src]), 6),
                    "dst_pred_prob": round(float(probs[e_dst]), 6),
                }
            )

    # Shared Unflagged Bridge Analysis: unflagged node b adjacent to >= 2 distinct predicted suspicious nodes
    shared_bridges_count = 0
    bridge_motif_counts = {"relay_sus_to_b_to_sus": 0, "shared_input_b_to_sus": 0, "shared_output_sus_to_b": 0}
    bridge_samples: List[Dict[str, Any]] = []

    for b in range(num_nodes):
        if b in sus_nodes:
            continue
        in_sus = [u for u in G.predecessors(b) if u in sus_nodes]
        out_sus = [u for u in G.successors(b) if u in sus_nodes]
        distinct_sus = set(in_sus) | set(out_sus)
        if len(distinct_sus) >= 2:
            shared_bridges_count += 1
            if len(in_sus) >= 1 and len(out_sus) >= 1:
                b_motif = "relay_sus_to_b_to_sus"
            elif len(out_sus) >= 2:
                b_motif = "shared_input_b_to_sus"
            else:
                b_motif = "shared_output_sus_to_b"
            bridge_motif_counts[b_motif] += 1
            if len(bridge_samples) < 25:
                bridge_samples.append(
                    {
                        "timestep": int(timestep),
                        "bridge_tx_id": int(tx_ids[b]),
                        "bridge_pred_prob": round(float(probs[b]), 6),
                        "bridge_gt_label": gt_label_str(int(y_opt[b])),
                        "bridge_motif": b_motif,
                        "sus_in_count": len(in_sus),
                        "sus_out_count": len(out_sus),
                        "distinct_sus_touched": len(distinct_sus),
                    }
                )

    return {
        "in_deg": in_deg,
        "out_deg": out_deg,
        "node_to_cid": node_to_cid,
        "cluster_rows": cluster_rows,
        "edge_rows": edge_rows,
        "direct_clusters_count": len(cluster_rows),
        "potential_candidates_count": potential_candidates,
        "shared_bridges_count": shared_bridges_count,
        "bridge_motif_counts": bridge_motif_counts,
        "bridge_samples": bridge_samples,
    }
