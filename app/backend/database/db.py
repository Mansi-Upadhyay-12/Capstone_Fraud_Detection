"""SQLite persistence layer (`app/data/fraud_app.db`) with parameterized queries and retention cleanup."""

import json
import os
import shutil
import sqlite3
import time
from typing import Any, Dict, List, Optional

from app.backend.config import DATA_DIR, DB_PATH, MAX_CACHED_RUNS, RETENTION_DAYS, RUNS_CACHE_DIR


def get_connection() -> sqlite3.Connection:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RUNS_CACHE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                created_epoch REAL NOT NULL,
                threshold_used REAL NOT NULL,
                inference_time_sec REAL NOT NULL,
                node_count INTEGER NOT NULL,
                edge_count INTEGER NOT NULL,
                flagged_count INTEGER NOT NULL,
                direct_clusters_count INTEGER NOT NULL,
                potential_candidates_count INTEGER NOT NULL,
                shared_bridges_count INTEGER NOT NULL,
                has_optional_class_column INTEGER NOT NULL,
                summary_json TEXT NOT NULL,
                cache_available INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS transactions (
                run_id TEXT NOT NULL,
                node_order INTEGER NOT NULL,
                tx_id INTEGER NOT NULL,
                timestep INTEGER NOT NULL,
                snapshot_node_idx INTEGER NOT NULL,
                pred_prob REAL NOT NULL,
                logit REAL NOT NULL,
                pred_class INTEGER NOT NULL,
                pred_label TEXT NOT NULL,
                risk_category TEXT NOT NULL,
                in_deg INTEGER NOT NULL,
                out_deg INTEGER NOT NULL,
                cluster_id TEXT NOT NULL,
                optional_gt_code INTEGER NOT NULL,
                optional_gt_label TEXT NOT NULL,
                PRIMARY KEY (run_id, tx_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_tx_run_prob ON transactions (run_id, pred_prob DESC);
            CREATE INDEX IF NOT EXISTS idx_tx_run_step ON transactions (run_id, timestep);

            CREATE TABLE IF NOT EXISTS clusters (
                run_id TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                timestep INTEGER NOT NULL,
                structural_tier TEXT NOT NULL,
                is_ring_candidate INTEGER NOT NULL,
                motif_type TEXT NOT NULL,
                ring_score REAL NOT NULL,
                num_nodes INTEGER NOT NULL,
                num_internal_edges INTEGER NOT NULL,
                mean_pred_prob REAL NOT NULL,
                max_pred_prob REAL NOT NULL,
                hub_tx_id INTEGER NOT NULL,
                hub_node_idx INTEGER NOT NULL,
                details_json TEXT NOT NULL,
                PRIMARY KEY (run_id, cluster_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS cluster_edges (
                run_id TEXT NOT NULL,
                cluster_id TEXT NOT NULL,
                timestep INTEGER NOT NULL,
                src_tx_id INTEGER NOT NULL,
                dst_tx_id INTEGER NOT NULL,
                edge_type TEXT NOT NULL,
                src_pred_prob REAL NOT NULL,
                dst_pred_prob REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_ce_run_cid ON cluster_edges (run_id, cluster_id);

            CREATE TABLE IF NOT EXISTS explanations (
                run_id TEXT NOT NULL,
                tx_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                method TEXT NOT NULL,
                completeness_abs_err REAL NOT NULL,
                completeness_rel_pct REAL NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (run_id, tx_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
            );
            """
        )


def save_analysis_run(
    run_meta: Dict[str, Any],
    tx_rows: List[Dict[str, Any]],
    cluster_rows: List[Dict[str, Any]],
    edge_rows: List[Dict[str, Any]],
) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, dataset_name, created_at, created_epoch, threshold_used,
                inference_time_sec, node_count, edge_count, flagged_count,
                direct_clusters_count, potential_candidates_count, shared_bridges_count,
                has_optional_class_column, summary_json, cache_available
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_meta["run_id"],
                run_meta["dataset_name"],
                run_meta["created_at"],
                run_meta["created_epoch"],
                run_meta["threshold_used"],
                run_meta["inference_time_sec"],
                run_meta["node_count"],
                run_meta["edge_count"],
                run_meta["flagged_count"],
                run_meta["direct_clusters_count"],
                run_meta["potential_candidates_count"],
                run_meta["shared_bridges_count"],
                int(run_meta["has_optional_class_column"]),
                json.dumps(run_meta["summary"]),
                1,
            ),
        )
        conn.executemany(
            """
            INSERT INTO transactions (
                run_id, node_order, tx_id, timestep, snapshot_node_idx,
                pred_prob, logit, pred_class, pred_label, risk_category,
                in_deg, out_deg, cluster_id, optional_gt_code, optional_gt_label
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_meta["run_id"], r["node_order"], r["tx_id"], r["timestep"],
                    r["snapshot_node_idx"], r["pred_prob"], r["logit"], r["pred_class"],
                    r["pred_label"], r["risk_category"], r["in_deg"], r["out_deg"],
                    r["cluster_id"], r["optional_gt_code"], r["optional_gt_label"],
                )
                for r in tx_rows
            ],
        )
        if cluster_rows:
            conn.executemany(
                """
                INSERT INTO clusters (
                    run_id, cluster_id, timestep, structural_tier, is_ring_candidate,
                    motif_type, ring_score, num_nodes, num_internal_edges,
                    mean_pred_prob, max_pred_prob, hub_tx_id, hub_node_idx, details_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_meta["run_id"], c["cluster_id"], c["timestep"], c["structural_tier"],
                        c["is_ring_candidate"], c["motif_type"], c["ring_score"], c["num_nodes"],
                        c["num_internal_edges"], c["mean_pred_prob"], c["max_pred_prob"],
                        c["hub_tx_id"], c["hub_node_idx"], json.dumps(c),
                    )
                    for c in cluster_rows
                ],
            )
        if edge_rows:
            conn.executemany(
                """
                INSERT INTO cluster_edges (
                    run_id, cluster_id, timestep, src_tx_id, dst_tx_id,
                    edge_type, src_pred_prob, dst_pred_prob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_meta["run_id"], e["cluster_id"], e["timestep"], e["src_tx_id"],
                        e["dst_tx_id"], e["edge_type"], e["src_pred_prob"], e["dst_pred_prob"],
                    )
                    for e in edge_rows
                ],
            )
    prune_old_run_caches()


def prune_old_run_caches() -> int:
    """
    Retention policy: removes cached tensor bundles older than RETENTION_DAYS
    or exceeding MAX_CACHED_RUNS while keeping metadata & predictions in SQLite.
    """
    now = time.time()
    cutoff = now - RETENTION_DAYS * 86400
    pruned = 0
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT run_id, created_epoch FROM runs WHERE cache_available = 1 ORDER BY created_epoch DESC"
        ).fetchall()
        for idx, row in enumerate(rows):
            rid = row["run_id"]
            if idx >= MAX_CACHED_RUNS or row["created_epoch"] < cutoff:
                cdir = os.path.join(RUNS_CACHE_DIR, rid)
                if os.path.exists(cdir):
                    shutil.rmtree(cdir, ignore_errors=True)
                conn.execute("UPDATE runs SET cache_available = 0 WHERE run_id = ?", (rid,))
                pruned += 1
    return pruned


def list_runs() -> List[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT run_id, dataset_name, created_at, node_count, edge_count,
                   flagged_count, direct_clusters_count, potential_candidates_count,
                   threshold_used, inference_time_sec, has_optional_class_column
            FROM runs ORDER BY created_epoch DESC
            """
        ).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["summary"] = json.loads(d.pop("summary_json"))
    d["has_optional_class_column"] = bool(d["has_optional_class_column"])
    return d


def query_transactions(
    run_id: str,
    timestep: Optional[int] = None,
    flagged_only: bool = False,
    min_prob: float = 0.0,
    search_tx_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    init_db()
    clauses = ["run_id = ?", "pred_prob >= ?"]
    params: List[Any] = [run_id, min_prob]
    if timestep is not None:
        clauses.append("timestep = ?")
        params.append(timestep)
    if flagged_only:
        clauses.append("pred_class = 1")
    if search_tx_id is not None:
        clauses.append("CAST(tx_id AS TEXT) LIKE ?")
        params.append(f"%{int(search_tx_id)}%")

    where_sql = " AND ".join(clauses)
    with get_connection() as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM transactions WHERE {where_sql}", params).fetchone()[0]
        rows = conn.execute(
            f"""
            SELECT * FROM transactions WHERE {where_sql}
            ORDER BY pred_prob DESC, tx_id ASC LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        ).fetchall()
    return {"total": int(total), "limit": limit, "offset": offset, "items": [dict(r) for r in rows]}


def get_transaction(run_id: str, tx_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM transactions WHERE run_id = ? AND tx_id = ?", (run_id, tx_id)
        ).fetchone()
    return dict(row) if row is not None else None


def query_clusters(run_id: str, candidates_only: bool = False, timestep: Optional[int] = None) -> List[Dict[str, Any]]:
    init_db()
    clauses = ["run_id = ?"]
    params: List[Any] = [run_id]
    if candidates_only:
        clauses.append("is_ring_candidate = 1")
    if timestep is not None:
        clauses.append("timestep = ?")
        params.append(timestep)
    where_sql = " AND ".join(clauses)
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT details_json FROM clusters WHERE {where_sql}
            ORDER BY is_ring_candidate DESC, ring_score DESC, num_nodes DESC, timestep ASC
            """,
            params,
        ).fetchall()
    return [json.loads(r["details_json"]) for r in rows]


def get_cluster(run_id: str, cluster_id: str) -> Optional[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT details_json FROM clusters WHERE run_id = ? AND cluster_id = ?",
            (run_id, cluster_id),
        ).fetchone()
    return json.loads(row["details_json"]) if row is not None else None


def get_cluster_nodes_and_edges(run_id: str, cluster_id: str) -> Dict[str, Any]:
    init_db()
    with get_connection() as conn:
        nodes = conn.execute(
            "SELECT * FROM transactions WHERE run_id = ? AND cluster_id = ? ORDER BY pred_prob DESC, tx_id ASC",
            (run_id, cluster_id),
        ).fetchall()
        edges = conn.execute(
            "SELECT * FROM cluster_edges WHERE run_id = ? AND cluster_id = ?",
            (run_id, cluster_id),
        ).fetchall()
    return {"nodes": [dict(n) for n in nodes], "edges": [dict(e) for e in edges]}


def get_cached_explanation(run_id: str, tx_id: int) -> Optional[Dict[str, Any]]:
    init_db()
    with get_connection() as conn:
        row = conn.execute(
            "SELECT payload_json FROM explanations WHERE run_id = ? AND tx_id = ?",
            (run_id, tx_id),
        ).fetchone()
    return json.loads(row["payload_json"]) if row is not None else None


def save_explanation(run_id: str, tx_id: int, created_at: str, method: str, abs_err: float, rel_err: float, payload: Dict[str, Any]) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO explanations (
                run_id, tx_id, created_at, method, completeness_abs_err, completeness_rel_pct, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, tx_id, created_at, method, abs_err, rel_err, json.dumps(payload)),
        )
