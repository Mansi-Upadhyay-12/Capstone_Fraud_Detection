import React, { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  AlertTriangle,
  ArrowRight,
  Database,
  FileUp,
  Layers,
  Network,
  ShieldAlert,
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
import { fetchRunDetail } from '../api/fraudApi';
import { formatNumber, formatTimestamp } from '../components/RiskBadge';
import type { HealthResponse, RunDetail, RunSummaryItem } from '../types/api';

interface DashboardPageProps {
  runs: RunSummaryItem[];
  runsLoading: boolean;
  runsError: string | null;
  onRetryRuns: () => void;
  health: HealthResponse | null;
}

export function DashboardPage({
  runs,
  runsLoading,
  runsError,
  onRetryRuns,
  health,
}: DashboardPageProps): React.ReactElement {
  const [latestDetail, setLatestDetail] = useState<RunDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);

  const latestRunId = runs[0]?.run_id || null;

  useEffect(() => {
    if (!latestRunId) {
      setLatestDetail(null);
      return;
    }
    let cancelled = false;
    setDetailLoading(true);
    fetchRunDetail(latestRunId)
      .then((d) => {
        if (!cancelled) setLatestDetail(d);
      })
      .catch(() => {
        if (!cancelled) setLatestDetail(null);
      })
      .finally(() => {
        if (!cancelled) setDetailLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [latestRunId]);

  const totals = useMemo(() => {
    let nodes = 0;
    let flagged = 0;
    let clusters = 0;
    let candidates = 0;
    for (const r of runs) {
      nodes += r.node_count;
      flagged += r.flagged_count;
      clusters += r.direct_clusters_count;
      candidates += r.potential_candidates_count;
    }
    return { nodes, flagged, clusters, candidates };
  }, [runs]);

  const riskBandChartData = useMemo(() => {
    if (!latestDetail) return [];
    const b = latestDetail.summary.risk_band_counts;
    return [
      { band: 'High (>=0.85)', count: b.HighPredictedRisk, color: '#ef4444' },
      { band: 'Elevated (>=0.57)', count: b.ElevatedRisk, color: '#f97316' },
      { band: 'Moderate (0.35-0.57)', count: b.ModerateMonitor, color: '#eab308' },
      { band: 'Low (<0.35)', count: b.LowPredictedRisk, color: '#10b981' },
    ];
  }, [latestDetail]);

  const timestepChartData = useMemo(() => {
    if (!latestDetail) return [];
    return latestDetail.summary.per_timestep.map((s) => ({
      step: `T${s.timestep}`,
      flagged: s.flagged_count,
      clusters: s.direct_clusters_count,
    }));
  }, [latestDetail]);

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Analytics Overview</h2>
          <p className="page-desc">
            Aggregated batch inference insights from persisted CSV analysis runs.
          </p>
        </div>
        <Link to="/upload" className="btn btn-primary">
          <FileUp size={16} />
          <span>New CSV Analysis</span>
        </Link>
      </div>

      <div className="banner banner-info">
        <Activity size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <strong>Batch Offline Evaluation Engine:</strong> Predictions are generated on uploaded transaction
          and directed-edge CSVs using the frozen Stage 4H <span className="mono">{health?.model_id || 'ExpA2_RankGauss182_BiDirGCN'}</span>{' '}
          checkpoint at threshold <span className="mono">{(health?.threshold_used ?? 0.57).toFixed(2)}</span>.
          This workspace is not connected to a live blockchain stream.
        </div>
      </div>

      {runsError && (
        <div className="banner banner-error">
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
          <div style={{ flex: 1 }}>
            <strong>Failed to load analysis runs:</strong> {runsError}
          </div>
          <button type="button" className="btn btn-secondary" onClick={onRetryRuns}>
            Retry
          </button>
        </div>
      )}

      {runsLoading ? (
        <div className="card">
          <div className="empty-state">Loading persisted analysis runs from backend SQLite storage...</div>
        </div>
      ) : runs.length === 0 ? (
        <div className="card">
          <div className="empty-state">
            <Database size={40} color="var(--accent-blue)" />
            <h3 style={{ margin: 0, color: 'var(--text-primary)' }}>No Analysis Runs Recorded Yet</h3>
            <p style={{ margin: 0, maxWidth: 520, lineHeight: 1.55 }}>
              Upload a validated pair of <span className="mono">transactions.csv</span> (with all 182 features)
              and <span className="mono">edges.csv</span> files to score transactions, discover directed suspicious
              subgraphs, and explore explanations.
            </p>
            <Link to="/upload" className="btn btn-primary" style={{ marginTop: 6 }}>
              <FileUp size={16} />
              <span>Upload CSV Bundle</span>
            </Link>
          </div>
        </div>
      ) : (
        <>
          <div className="kpi-grid">
            <div className="kpi-card">
              <div className="kpi-top">
                <span>Total Analysed Transactions</span>
                <Layers size={16} color="var(--accent-blue)" />
              </div>
              <div className="kpi-value">{formatNumber(totals.nodes)}</div>
              <div className="kpi-meta">Across {runs.length} persisted batch run(s)</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-top">
                <span>High-Risk / Flagged Transactions</span>
                <ShieldAlert size={16} color="var(--risk-elevated)" />
              </div>
              <div className="kpi-value">{formatNumber(totals.flagged)}</div>
              <div className="kpi-meta">
                {totals.nodes > 0 ? ((totals.flagged / totals.nodes) * 100).toFixed(2) : '0.00'}% flagged at threshold 0.57
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-top">
                <span>Suspicious Clusters</span>
                <Network size={16} color="var(--accent-teal)" />
              </div>
              <div className="kpi-value">{formatNumber(totals.clusters)}</div>
              <div className="kpi-meta">{formatNumber(totals.candidates)} Tier-A ring candidate(s)</div>
            </div>

            <div className="kpi-card">
              <div className="kpi-top">
                <span>Analysis Run Count</span>
                <Activity size={16} color="var(--accent-indigo)" />
              </div>
              <div className="kpi-value">{formatNumber(runs.length)}</div>
              <div className="kpi-meta">
                Latest latency: {runs[0].inference_time_sec.toFixed(2)}s
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <h3 className="card-title">Latest Analysis Summary</h3>
                <p className="card-subtitle">
                  Run <span className="mono">{runs[0].run_id}</span> • Dataset: <strong>{runs[0].dataset_name}</strong> •{' '}
                  {formatTimestamp(runs[0].created_at)}
                </p>
              </div>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                <Link to={`/runs/${encodeURIComponent(runs[0].run_id)}`} className="btn btn-secondary">
                  Run Details <ArrowRight size={14} />
                </Link>
                <Link
                  to={`/runs/${encodeURIComponent(runs[0].run_id)}/transactions`}
                  className="btn btn-secondary"
                >
                  Transactions
                </Link>
                <Link
                  to={`/runs/${encodeURIComponent(runs[0].run_id)}/clusters`}
                  className="btn btn-secondary"
                >
                  Clusters
                </Link>
              </div>
            </div>

            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, fontSize: '0.86rem', color: 'var(--text-secondary)' }}>
              <span>
                Nodes: <strong style={{ color: 'var(--text-primary)' }}>{formatNumber(runs[0].node_count)}</strong>
              </span>
              <span>
                Unique Edges: <strong style={{ color: 'var(--text-primary)' }}>{formatNumber(runs[0].edge_count)}</strong>
              </span>
              <span>
                Flagged (&ge; {runs[0].threshold_used.toFixed(2)}):{' '}
                <strong style={{ color: 'var(--risk-elevated)' }}>{formatNumber(runs[0].flagged_count)}</strong>
              </span>
              <span>
                Suspicious Clusters:{' '}
                <strong style={{ color: 'var(--accent-teal)' }}>{formatNumber(runs[0].direct_clusters_count)}</strong>
              </span>
              <span>
                Optional Labels Present:{' '}
                <strong style={{ color: 'var(--text-primary)' }}>
                  {runs[0].has_optional_class_column ? 'Yes (Post-Hoc Only)' : 'No'}
                </strong>
              </span>
            </div>
          </div>

          <div className="grid-2">
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Risk Band Distribution (Latest Run)</h3>
                  <p className="card-subtitle">Exact backend risk classification counts</p>
                </div>
              </div>
              {detailLoading || !latestDetail ? (
                <div className="empty-state" style={{ padding: 28 }}>Loading risk breakdown...</div>
              ) : (
                <div style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <BarChart data={riskBandChartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                      <XAxis dataKey="band" stroke="#94a3b8" fontSize={12} />
                      <YAxis stroke="#94a3b8" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0d1628',
                          borderColor: 'rgba(148,163,184,0.25)',
                          color: '#f1f5f9',
                        }}
                      />
                      <Bar dataKey="count" radius={[6, 6, 0, 0]} isAnimationActive={false}>
                        {riskBandChartData.map((entry) => (
                          <Cell key={entry.band} fill={entry.color} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>

            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Per-Timestep Flagged &amp; Cluster Counts</h3>
                  <p className="card-subtitle">Snapshot-by-snapshot temporal summary (Latest Run)</p>
                </div>
              </div>
              {detailLoading || !latestDetail ? (
                <div className="empty-state" style={{ padding: 28 }}>Loading snapshot timeline...</div>
              ) : (
                <div style={{ width: '100%', height: 260 }}>
                  <ResponsiveContainer>
                    <BarChart data={timestepChartData} margin={{ top: 8, right: 16, left: 0, bottom: 8 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                      <XAxis dataKey="step" stroke="#94a3b8" fontSize={12} />
                      <YAxis stroke="#94a3b8" fontSize={12} />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#0d1628',
                          borderColor: 'rgba(148,163,184,0.25)',
                          color: '#f1f5f9',
                        }}
                      />
                      <Bar dataKey="flagged" name="Flagged Txs" fill="#38bdf8" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                      <Bar dataKey="clusters" name="Clusters" fill="#2dd4bf" radius={[4, 4, 0, 0]} isAnimationActive={false} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>

          <div className="card">
            <div className="card-header">
              <div>
                <h3 className="card-title">Recent Analysis History</h3>
                <p className="card-subtitle">Persisted batch runs in SQLite (newest first)</p>
              </div>
            </div>
            <div className="table-wrap">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Run ID</th>
                    <th>Dataset</th>
                    <th>Created</th>
                    <th>Transactions</th>
                    <th>Flagged</th>
                    <th>Clusters</th>
                    <th>Runtime</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {runs.map((r) => (
                    <tr key={r.run_id}>
                      <td className="mono">{r.run_id}</td>
                      <td>{r.dataset_name}</td>
                      <td style={{ color: 'var(--text-secondary)' }}>{formatTimestamp(r.created_at)}</td>
                      <td>{formatNumber(r.node_count)}</td>
                      <td>
                        <span className="badge badge-elevated">{formatNumber(r.flagged_count)}</span>
                      </td>
                      <td>
                        {formatNumber(r.direct_clusters_count)}{' '}
                        <span style={{ color: 'var(--text-muted)', fontSize: '0.76rem' }}>
                          ({r.potential_candidates_count} cand.)
                        </span>
                      </td>
                      <td className="mono">{r.inference_time_sec.toFixed(2)}s</td>
                      <td>
                        <div style={{ display: 'flex', gap: 8 }}>
                          <Link to={`/runs/${encodeURIComponent(r.run_id)}`} className="btn btn-ghost">
                            Overview
                          </Link>
                          <Link
                            to={`/runs/${encodeURIComponent(r.run_id)}/transactions`}
                            className="btn btn-ghost"
                          >
                            Txs
                          </Link>
                          <Link
                            to={`/runs/${encodeURIComponent(r.run_id)}/clusters`}
                            className="btn btn-ghost"
                          >
                            Clusters
                          </Link>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </>
  );
}
