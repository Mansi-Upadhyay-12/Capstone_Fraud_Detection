import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { AlertTriangle, ArrowLeft, GitBranch, Network } from 'lucide-react';
import { BackendRequestError, fetchRunClusters, fetchRunDetail } from '../api/fraudApi';
import { formatNumber } from '../components/RiskBadge';
import type { ClusterItem, RunDetail } from '../types/api';

export function ClustersPage(): React.ReactElement {
  const { runId = '' } = useParams<{ runId: string }>();
  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [clusters, setClusters] = useState<ClusterItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [candidatesOnly, setCandidatesOnly] = useState(false);
  const [timestepFilter, setTimestepFilter] = useState<number | null>(null);

  useEffect(() => {
    if (!runId) return;
    fetchRunDetail(runId)
      .then((d) => setRunDetail(d))
      .catch(() => setRunDetail(null));
  }, [runId]);

  const loadClusters = () => {
    if (!runId) return;
    setLoading(true);
    setError(null);
    fetchRunClusters(runId, candidatesOnly, timestepFilter)
      .then((res) => setClusters(res.clusters))
      .catch((err) => {
        setError(err instanceof BackendRequestError ? err.message : 'Unable to fetch suspicious clusters.');
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadClusters();
  }, [runId, candidatesOnly, timestepFilter]);

  return (
    <>
      <div className="page-header">
        <div>
          <Link
            to={`/runs/${encodeURIComponent(runId)}`}
            style={{ fontSize: '0.82rem', display: 'inline-flex', alignItems: 'center', gap: 5 }}
          >
            <ArrowLeft size={14} /> Back to Run Details
          </Link>
          <h2 className="page-title" style={{ marginTop: 6 }}>
            Model-Identified Suspicious Clusters
          </h2>
          <p className="page-desc">
            Weakly connected subgraphs of flagged transactions (&ge; 2 nodes) discovered in run{' '}
            <span className="mono">{runId}</span>.
          </p>
        </div>
        <Link to={`/runs/${encodeURIComponent(runId)}/transactions`} className="btn btn-secondary">
          Explore Transactions
        </Link>
      </div>

      <div className="banner banner-warning">
        <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <strong>Investigative Caution:</strong> These clusters are <strong>model-identified suspicious groups</strong>{' '}
          extracted from frozen BiDirGCN probabilities (&ge; 0.57) and intra-timestep transaction edges. They are
          prioritized triage groups and are <strong>not confirmed criminal fraud rings</strong>.
        </div>
      </div>

      <div className="card">
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: 14,
            marginBottom: 16,
          }}
        >
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: '0.86rem', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={candidatesOnly}
              onChange={(e) => setCandidatesOnly(e.target.checked)}
            />
            <span>Potential Fraud-Ring Candidates Only (Tier A)</span>
          </label>

          <select
            aria-label="Filter clusters by timestep"
            className="select"
            value={timestepFilter ?? ''}
            onChange={(e) => setTimestepFilter(e.target.value ? Number(e.target.value) : null)}
          >
            <option value="">All Timesteps</option>
            {(runDetail?.summary.timesteps || []).map((t) => (
              <option key={t} value={t}>
                Timestep {t}
              </option>
            ))}
          </select>

          <span style={{ marginLeft: 'auto', fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
            Showing <strong>{formatNumber(clusters.length)}</strong> cluster(s)
          </span>
        </div>

        {error && (
          <div className="banner banner-error" style={{ marginBottom: 14 }}>
            <AlertTriangle size={18} />
            <div style={{ flex: 1 }}>{error}</div>
            <button type="button" className="btn btn-secondary" onClick={loadClusters}>
              Retry
            </button>
          </div>
        )}

        {loading ? (
          <div className="empty-state">Loading suspicious clusters from backend...</div>
        ) : clusters.length === 0 ? (
          <div className="empty-state">
            <Network size={36} color="var(--text-muted)" />
            <div>No suspicious clusters matched the current filter criteria.</div>
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Cluster ID</th>
                  <th>Step</th>
                  <th>Nodes / Edges</th>
                  <th>Risk Summary</th>
                  <th>Directed Structure</th>
                  <th>Candidate Status</th>
                  <th>Hub Tx</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {clusters.map((c) => (
                  <tr key={c.cluster_id}>
                    <td className="mono">{c.cluster_id}</td>
                    <td className="mono">{c.timestep}</td>
                    <td>
                      <strong>{c.num_nodes}</strong> nodes / <strong>{c.num_internal_edges}</strong> edges
                    </td>
                    <td>
                      <div>
                        Mean <span className="mono">{c.mean_pred_prob.toFixed(3)}</span> • Max{' '}
                        <span className="mono">{c.max_pred_prob.toFixed(3)}</span>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Ring Score: {c.ring_score.toFixed(2)} • &ge;0.85: {c.num_high_conf_ge_085}
                      </div>
                    </td>
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                        <GitBranch size={13} color="var(--accent-teal)" />
                        <span className="mono" style={{ fontSize: '0.8rem' }}>
                          {c.motif_type}
                        </span>
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        Int in/out max: {c.max_internal_in_deg}/{c.max_internal_out_deg} • Ext 1-hop:{' '}
                        {c.ext_1hop_in_neighbors} in / {c.ext_1hop_out_neighbors} out
                      </div>
                    </td>
                    <td>
                      {c.is_ring_candidate === 1 ? (
                        <span className="badge badge-critical">Ring Candidate</span>
                      ) : (
                        <span className="badge badge-neutral">{c.structural_tier.replace(/_/g, ' ')}</span>
                      )}
                    </td>
                    <td className="mono">{c.hub_tx_id}</td>
                    <td>
                      <Link
                        to={`/runs/${encodeURIComponent(runId)}/clusters/${encodeURIComponent(c.cluster_id)}`}
                        className="btn btn-secondary"
                      >
                        Open Cluster
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
