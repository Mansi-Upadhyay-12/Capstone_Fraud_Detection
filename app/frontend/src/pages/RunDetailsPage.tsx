import React, { useEffect, useMemo, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Network,
  ShieldAlert,
  Table2,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { BackendRequestError, fetchRunDetail } from '../api/fraudApi';
import { formatNumber, formatTimestamp } from '../components/RiskBadge';
import type { RunDetail } from '../types/api';

export function RunDetailsPage(): React.ReactElement {
  const { runId = '' } = useParams<{ runId: string }>();
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadDetail = () => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    fetchRunDetail(runId)
      .then((res) => setDetail(res))
      .catch((err) => {
        setError(err instanceof BackendRequestError ? err.message : 'Unable to load run details.');
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadDetail();
  }, [runId]);

  const riskBands = useMemo(() => {
    if (!detail) return [];
    const r = detail.summary.risk_band_counts;
    return [
      { label: 'High (>=0.85)', count: r.HighPredictedRisk, color: '#ef4444' },
      { label: 'Elevated (0.57-0.85)', count: r.ElevatedRisk, color: '#f97316' },
      { label: 'Moderate (0.35-0.57)', count: r.ModerateMonitor, color: '#eab308' },
      { label: 'Low (<0.35)', count: r.LowPredictedRisk, color: '#10b981' },
    ];
  }, [detail]);

  if (loading) {
    return (
      <div className="card">
        <div className="empty-state">Loading run metadata and risk summary...</div>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="card">
        <div className="banner banner-error">
          <AlertTriangle size={18} />
          <div style={{ flex: 1 }}>{error || 'Run not found.'}</div>
          <button type="button" className="btn btn-secondary" onClick={loadDetail}>
            Retry
          </button>
        </div>
        <div style={{ marginTop: 14 }}>
          <Link to="/" className="btn btn-ghost">
            <ArrowLeft size={14} /> Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <>
      <div className="page-header">
        <div>
          <Link to="/" style={{ fontSize: '0.82rem', display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <ArrowLeft size={14} /> Back to Dashboard
          </Link>
          <h2 className="page-title" style={{ marginTop: 6 }}>
            Run Details: {detail.dataset_name}
          </h2>
          <p className="page-desc mono">
            {detail.run_id} • {formatTimestamp(detail.created_at)}
          </p>
        </div>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          <Link to={`/runs/${encodeURIComponent(detail.run_id)}/transactions`} className="btn btn-primary">
            <Table2 size={16} /> Explore Transactions
          </Link>
          <Link to={`/runs/${encodeURIComponent(detail.run_id)}/clusters`} className="btn btn-secondary">
            <Network size={16} /> Inspect Suspicious Clusters
          </Link>
        </div>
      </div>

      <div className="kpi-grid">
        <div className="kpi-card">
          <div className="kpi-top">
            <span>Transactions Analysed</span>
            <Activity size={16} color="var(--accent-blue)" />
          </div>
          <div className="kpi-value">{formatNumber(detail.node_count)}</div>
          <div className="kpi-meta">{formatNumber(detail.edge_count)} deduplicated directed edges</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <span>Flagged Transactions</span>
            <ShieldAlert size={16} color="var(--risk-elevated)" />
          </div>
          <div className="kpi-value">{formatNumber(detail.flagged_count)}</div>
          <div className="kpi-meta">
            {detail.summary.flagged_rate_pct.toFixed(2)}% at threshold {detail.threshold_used.toFixed(2)}
          </div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <span>Suspicious Clusters</span>
            <Network size={16} color="var(--accent-teal)" />
          </div>
          <div className="kpi-value">{formatNumber(detail.direct_clusters_count)}</div>
          <div className="kpi-meta">{formatNumber(detail.potential_candidates_count)} Tier-A candidate(s)</div>
        </div>

        <div className="kpi-card">
          <div className="kpi-top">
            <span>Shared Unflagged Bridges</span>
            <Activity size={16} color="var(--accent-indigo)" />
          </div>
          <div className="kpi-value">{formatNumber(detail.shared_bridges_count)}</div>
          <div className="kpi-meta">Inference time: {detail.inference_time_sec.toFixed(3)}s</div>
        </div>
      </div>

      <div className="banner banner-info">
        <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>{detail.summary.disclaimer}</div>
      </div>

      {detail.summary.warnings.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {detail.summary.warnings.map((w, idx) => (
            <div key={idx} className="banner banner-warning">
              <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
              <span>{w}</span>
            </div>
          ))}
        </div>
      )}

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Risk Distribution</h3>
              <p className="card-subtitle">
                Mean P={detail.summary.mean_pred_prob.toFixed(4)} • Max P={detail.summary.max_pred_prob.toFixed(4)}
              </p>
            </div>
            <span className="badge badge-teal">Status: Completed</span>
          </div>
          <div style={{ width: '100%', height: 250 }}>
            <ResponsiveContainer>
              <BarChart data={riskBands} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                <XAxis dataKey="label" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#0d1628',
                    borderColor: 'rgba(148,163,184,0.25)',
                    color: '#f1f5f9',
                  }}
                />
                <Bar dataKey="count" radius={[6, 6, 0, 0]} isAnimationActive={false}>
                  {riskBands.map((b) => (
                    <Cell key={b.label} fill={b.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div>
              <h3 className="card-title">Execution &amp; Pipeline Metadata</h3>
              <p className="card-subtitle">Frozen Stage 4H execution details</p>
            </div>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, fontSize: '0.86rem' }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Model ID: </span>
              <span className="mono">{detail.summary.model_id}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Preprocessing Pipeline: </span>
              <span>{detail.summary.preprocessing_order.join(' -> ')}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Timesteps Uploaded: </span>
              <span className="mono">{detail.summary.timesteps.join(', ')}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Duplicate Edges Deduplicated: </span>
              <span>{formatNumber(detail.summary.duplicate_edges_removed)}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Median-Imputed NaN Cells: </span>
              <span>{formatNumber(detail.summary.imputed_nan_cells)}</span>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Snapshot Tensor Cache (for IG &amp; Cluster Graphs): </span>
              <span className={`badge ${detail.cache_available ? 'badge-low' : 'badge-neutral'}`}>
                {detail.cache_available ? 'Available' : 'Expired / Evicted'}
              </span>
            </div>
          </div>

          {detail.summary.post_hoc_eval && (
            <div
              style={{
                marginTop: 16,
                padding: 12,
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-canvas)',
                border: '1px solid var(--border-subtle)',
                fontSize: '0.82rem',
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: 6 }}>
                Post-Hoc Optional Label Comparison (Not Used in Inference)
              </div>
              <div style={{ color: 'var(--text-secondary)', marginBottom: 6 }}>
                Labeled: {detail.summary.post_hoc_eval.labeled_count} | Unknown:{' '}
                {detail.summary.post_hoc_eval.unknown_count}
              </div>
              <div className="mono" style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
                <span>TP={detail.summary.post_hoc_eval.tp}</span>
                <span>FP={detail.summary.post_hoc_eval.fp}</span>
                <span>FN={detail.summary.post_hoc_eval.fn}</span>
                <span>TN={detail.summary.post_hoc_eval.tn}</span>
                <span>Prec={(detail.summary.post_hoc_eval.precision * 100).toFixed(2)}%</span>
                <span>Rec={(detail.summary.post_hoc_eval.recall * 100).toFixed(2)}%</span>
                <span>F1={(detail.summary.post_hoc_eval.f1 * 100).toFixed(2)}%</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Per-Timestep Snapshot Summary</h3>
            <p className="card-subtitle">Strict intra-timestep inference breakdown</p>
          </div>
        </div>
        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Time Step</th>
                <th>Nodes</th>
                <th>Edges</th>
                <th>Flagged</th>
                <th>Flag Rate</th>
                <th>Mean Prob</th>
                <th>Max Prob</th>
                <th>Clusters</th>
                <th>Candidates</th>
                <th>Bridges</th>
              </tr>
            </thead>
            <tbody>
              {detail.summary.per_timestep.map((s) => (
                <tr key={s.timestep}>
                  <td className="mono">{s.timestep}</td>
                  <td>{formatNumber(s.node_count)}</td>
                  <td>{formatNumber(s.edge_count)}</td>
                  <td>{formatNumber(s.flagged_count)}</td>
                  <td>{s.flagged_rate_pct.toFixed(2)}%</td>
                  <td className="mono">{s.mean_pred_prob.toFixed(4)}</td>
                  <td className="mono">{s.max_pred_prob.toFixed(4)}</td>
                  <td>{formatNumber(s.direct_clusters_count)}</td>
                  <td>{formatNumber(s.potential_candidates_count)}</td>
                  <td>{formatNumber(s.shared_bridges_count)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
