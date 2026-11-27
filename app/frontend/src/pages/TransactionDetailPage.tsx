import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import {
  AlertCircle,
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Cpu,
  Database,
  GitFork,
  Loader2,
  Network,
  RefreshCw,
  Sparkles,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  BackendRequestError,
  explainTransaction,
  fetchRunTransactions,
} from '../api/fraudApi';
import { GtBadge, ProbBar, RiskBadge } from '../components/RiskBadge';
import type { TransactionExplanationResponse, TransactionItem } from '../types/api';

const GROUP_DISPLAY_LABEL: Record<string, string> = {
  Local93: 'Local Features (93)',
  Aggregate72: 'Aggregate Features (72)',
  Domain17: 'Domain Features (17)',
};

export function TransactionDetailPage(): React.ReactElement {
  const { runId = '', txId = '' } = useParams<{ runId: string; txId: string }>();
  const [tx, setTx] = useState<TransactionItem | null>(null);
  const [txLoading, setTxLoading] = useState(true);
  const [txError, setTxError] = useState<string | null>(null);

  const [explanation, setExplanation] = useState<TransactionExplanationResponse | null>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainError, setExplainError] = useState<string | null>(null);
  const explainInFlightRef = useRef(false);

  useEffect(() => {
    if (!runId || !txId) return;
    let active = true;
    setTxLoading(true);
    setTxError(null);
    setExplanation(null);
    setExplainError(null);

    fetchRunTransactions(runId, { searchTxId: txId, limit: 50 })
      .then((res) => {
        if (!active) return;
        const exact = res.items.find((row) => String(row.tx_id) === String(txId));
        if (!exact) {
          setTxError(`Transaction ${txId} was not found in run ${runId}.`);
        } else {
          setTx(exact);
        }
      })
      .catch((err) => {
        if (!active) return;
        setTxError(err instanceof BackendRequestError ? err.message : 'Unable to load transaction metadata.');
      })
      .finally(() => {
        if (active) setTxLoading(false);
      });

    return () => {
      active = false;
    };
  }, [runId, txId]);

  const handleRequestExplanation = async () => {
    if (!runId || !txId || explainInFlightRef.current) return;
    explainInFlightRef.current = true;
    setExplainLoading(true);
    setExplainError(null);

    try {
      const res = await explainTransaction(runId, txId);
      setExplanation(res);
    } catch (err) {
      setExplainError(
        err instanceof BackendRequestError
          ? err.message
          : 'Unable to compute transaction explanation.',
      );
    } finally {
      explainInFlightRef.current = false;
      setExplainLoading(false);
    }
  };

  const combinedChartData = useMemo(() => {
    if (!explanation) return [];
    const positives = explanation.top_positive_features.slice(0, 6).map((f) => ({
      name: f.feature_name,
      attribution: f.graph_ig_attribution,
      group: f.feature_group,
      direction: 'Positive (Increases Fraud Score)',
    }));
    const negatives = explanation.top_negative_features.slice(0, 6).map((f) => ({
      name: f.feature_name,
      attribution: f.graph_ig_attribution,
      group: f.feature_group,
      direction: 'Negative (Decreases Fraud Score)',
    }));
    // Order highest positive down to most negative
    return [...positives.sort((a, b) => b.attribution - a.attribution), ...negatives.sort((a, b) => b.attribution - a.attribution)];
  }, [explanation]);

  if (txLoading) {
    return (
      <div className="card">
        <div className="empty-state">Loading transaction risk record...</div>
      </div>
    );
  }

  if (txError || !tx) {
    return (
      <div className="card">
        <div className="banner banner-error">
          <AlertTriangle size={18} />
          <div>{txError || 'Transaction not found.'}</div>
        </div>
        <div style={{ marginTop: 14 }}>
          <Link to={`/runs/${encodeURIComponent(runId)}/transactions`} className="btn btn-secondary">
            <ArrowLeft size={14} /> Back to Transactions
          </Link>
        </div>
      </div>
    );
  }

  const integrationStepsLabel = explanation?.method.includes('M200')
    ? 'M = 200 (3-zone Composite Midpoint Grid)'
    : explanation?.method || 'Integrated Gradients';

  return (
    <>
      <div className="page-header">
        <div>
          <Link
            to={`/runs/${encodeURIComponent(runId)}/transactions`}
            style={{ fontSize: '0.82rem', display: 'inline-flex', alignItems: 'center', gap: 5 }}
          >
            <ArrowLeft size={14} /> Back to Transaction Risk Table
          </Link>
          <h2 className="page-title" style={{ marginTop: 6 }}>
            Transaction <span className="mono">{tx.tx_id}</span>
          </h2>
          <p className="page-desc">
            Run <span className="mono">{runId}</span> • Snapshot Time Step {tx.timestep}
          </p>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
          {tx.cluster_id && (
            <Link
              to={`/runs/${encodeURIComponent(runId)}/clusters/${encodeURIComponent(tx.cluster_id)}`}
              className="btn btn-secondary"
            >
              <Network size={16} /> View Cluster {tx.cluster_id}
            </Link>
          )}
          <button
            type="button"
            className="btn btn-primary"
            disabled={explainLoading}
            onClick={handleRequestExplanation}
          >
            {explainLoading ? <Loader2 size={16} className="spin" /> : <Sparkles size={16} />}
            <span>
              {explainLoading
                ? 'Computing IG Attribution...'
                : explanation
                  ? 'Refresh Explanation'
                  : 'Explain Prediction (IG)'}
            </span>
          </button>
        </div>
      </div>

      {/* A. Transaction Risk Summary */}
      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Transaction Risk Summary</h3>
            <p className="card-subtitle">
              Model prediction at frozen threshold 0.57 — strictly separated from optional ground-truth label
            </p>
          </div>
          <RiskBadge prob={tx.pred_prob} riskCategory={tx.risk_category} />
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(175px, 1fr))',
            gap: 16,
          }}
        >
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              Predicted Fraud Probability
            </div>
            <ProbBar prob={tx.pred_prob} />
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4 }}>Raw Logit Score</div>
            <div className="mono" style={{ fontWeight: 600 }}>
              {tx.logit !== undefined ? tx.logit.toFixed(6) : '—'}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              Model Predicted Class
            </div>
            <span className={`badge ${tx.pred_class === 1 ? 'badge-elevated' : 'badge-low'}`}>
              {tx.pred_label} (Predicted)
            </span>
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              Actual Ground-Truth Class (Optional)
            </div>
            <GtBadge label={tx.optional_gt_label} />
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4 }}>
              Directed Degree (In / Out)
            </div>
            <div className="mono" style={{ fontWeight: 600 }}>
              {tx.in_deg} / {tx.out_deg}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4 }}>Cluster Association</div>
            {tx.cluster_id ? (
              <Link
                to={`/runs/${encodeURIComponent(runId)}/clusters/${encodeURIComponent(tx.cluster_id)}`}
                className="mono"
              >
                {tx.cluster_id}
              </Link>
            ) : (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.86rem' }}>Singleton / None</span>
            )}
          </div>
        </div>
      </div>

      {/* E. On-Demand Explanation States */}
      {!explanation && !explainLoading && !explainError && (
        <div className="card">
          <div className="empty-state">
            <Cpu size={36} color="var(--accent-blue)" />
            <h3 style={{ margin: 0, color: 'var(--text-primary)' }}>On-Demand Feature Attribution Ready</h3>
            <p style={{ margin: 0, maxWidth: 580, lineHeight: 1.55 }}>
              Integrated Gradients (<span className="mono">M=200</span>) attributes the prediction logit across all
              182 features and snapshot neighbors relative to the RankGauss median baseline. Click{' '}
              <strong>Explain Prediction (IG)</strong> above to compute or retrieve cached attributions for transaction{' '}
              <span className="mono">{tx.tx_id}</span>.
            </p>
            <button type="button" className="btn btn-primary" onClick={handleRequestExplanation}>
              <Sparkles size={15} /> Explain Prediction (IG)
            </button>
          </div>
        </div>
      )}

      {explainLoading && (
        <div className="card">
          <div className="empty-state">
            <Loader2 size={32} color="var(--accent-teal)" className="spin" />
            <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
              Running On-Demand 3-Zone Composite Midpoint Integrated Gradients (M=200)...
            </div>
            <div style={{ fontSize: '0.84rem', color: 'var(--text-secondary)' }}>
              First-time explanations evaluate 200 gradient steps on the snapshot graph and are cached in SQLite.
            </div>
          </div>
        </div>
      )}

      {explainError && (
        <div className="card" style={{ borderColor: 'rgba(239, 68, 68, 0.4)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
            <AlertCircle size={20} color="var(--risk-critical)" style={{ flexShrink: 0, marginTop: 2 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, color: '#fca5a5' }}>Explanation Request Failed</div>
              <p style={{ margin: '6px 0 12px', color: 'var(--text-primary)', fontSize: '0.9rem' }}>
                {explainError}
              </p>
              <button type="button" className="btn btn-secondary" onClick={handleRequestExplanation}>
                <RefreshCw size={14} /> Retry Explanation
              </button>
            </div>
          </div>
        </div>
      )}

      {explanation && (
        <>
          <div className="banner banner-info">
            <CheckCircle2 size={18} style={{ flexShrink: 0, marginTop: 2 }} />
            <div>{explanation.disclaimer}</div>
          </div>

          {/* B. Feature Contribution Analysis */}
          <div className="card">
            <div className="card-header">
              <div>
                <h3 className="card-title">Top Positive &amp; Negative Feature Contributions</h3>
                <p className="card-subtitle">
                  Graph Integrated Gradients logit sensitivity (positive increases predicted fraud risk; negative pushes toward licit)
                </p>
              </div>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                <span className={`badge ${explanation.cache_hit ? 'badge-low' : 'badge-teal'}`}>
                  <Database size={12} /> {explanation.cache_hit ? 'Cached SQLite Result' : 'Freshly Computed'}
                </span>
              </div>
            </div>

            {combinedChartData.length === 0 ? (
              <div className="empty-state">No feature attribution records returned.</div>
            ) : (
              <div style={{ width: '100%', height: 340 }}>
                <ResponsiveContainer>
                  <BarChart
                    data={combinedChartData}
                    layout="vertical"
                    margin={{ top: 8, right: 24, left: 28, bottom: 8 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.12)" />
                    <XAxis type="number" stroke="#94a3b8" fontSize={12} />
                    <YAxis
                      type="category"
                      dataKey="name"
                      width={150}
                      stroke="#cbd5e1"
                      fontSize={11}
                    />
                    <ReferenceLine x={0} stroke="#64748b" />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#0d1628',
                        borderColor: 'rgba(148,163,184,0.28)',
                        color: '#f1f5f9',
                      }}
                      formatter={(val: number) => [val.toFixed(6), 'IG Logit Contribution']}
                    />
                    <Bar dataKey="attribution" radius={[4, 4, 4, 4]} isAnimationActive={false}>
                      {combinedChartData.map((entry) => (
                        <Cell
                          key={`${entry.direction}_${entry.name}`}
                          fill={entry.attribution >= 0 ? '#ef4444' : '#10b981'}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="grid-2" style={{ marginTop: 18 }}>
              <div>
                <h4 style={{ margin: '0 0 10px', fontSize: '0.9rem', color: '#fca5a5' }}>
                  Top Positive Contributors (Increase Fraud Score)
                </h4>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Group</th>
                        <th>Graph IG</th>
                        <th>Self / Nbr</th>
                      </tr>
                    </thead>
                    <tbody>
                      {explanation.top_positive_features.map((f) => (
                        <tr key={`pos_${f.feature_index}`}>
                          <td className="mono">{f.feature_name}</td>
                          <td>
                            <span className="badge badge-neutral">{f.feature_group}</span>
                          </td>
                          <td className="mono" style={{ color: '#fca5a5' }}>
                            +{f.graph_ig_attribution.toFixed(5)}
                          </td>
                          <td className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                            {f.self_ig_attribution.toFixed(4)} / {f.neighbor_ig_attribution.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              <div>
                <h4 style={{ margin: '0 0 10px', fontSize: '0.9rem', color: '#6ee7b7' }}>
                  Top Negative Contributors (Decrease Fraud Score)
                </h4>
                <div className="table-wrap">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Group</th>
                        <th>Graph IG</th>
                        <th>Self / Nbr</th>
                      </tr>
                    </thead>
                    <tbody>
                      {explanation.top_negative_features.map((f) => (
                        <tr key={`neg_${f.feature_index}`}>
                          <td className="mono">{f.feature_name}</td>
                          <td>
                            <span className="badge badge-neutral">{f.feature_group}</span>
                          </td>
                          <td className="mono" style={{ color: '#6ee7b7' }}>
                            {f.graph_ig_attribution.toFixed(5)}
                          </td>
                          <td className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                            {f.self_ig_attribution.toFixed(4)} / {f.neighbor_ig_attribution.toFixed(4)}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          {/* C. Feature Group Analysis & Self-vs-Neighbor Breakdown */}
          <div className="grid-2">
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Feature Group Attribution Summary</h3>
                  <p className="card-subtitle">
                    Grouped across Local (93), Aggregate (72), and Domain (17) features
                  </p>
                </div>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {explanation.feature_group_breakdown.map((g) => (
                  <div
                    key={g.group}
                    style={{
                      padding: 12,
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-canvas)',
                      border: '1px solid var(--border-subtle)',
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      flexWrap: 'wrap',
                      gap: 10,
                    }}
                  >
                    <div>
                      <div style={{ fontWeight: 600 }}>
                        {GROUP_DISPLAY_LABEL[g.group] || `${g.group} (${g.feature_count})`}
                      </div>
                      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                        Self-node signed sum: <span className="mono">{g.self_signed_sum.toFixed(5)}</span>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div
                        className="mono"
                        style={{
                          fontWeight: 700,
                          color: g.signed_sum >= 0 ? '#fca5a5' : '#6ee7b7',
                        }}
                      >
                        Signed: {g.signed_sum >= 0 ? `+${g.signed_sum.toFixed(5)}` : g.signed_sum.toFixed(5)}
                      </div>
                      <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                        |IG| Magnitude: {g.abs_sum.toFixed(5)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* D. Explanation Quality and Completeness + Topology Partition */}
            <div className="card">
              <div className="card-header">
                <div>
                  <h3 className="card-title">Explanation Quality &amp; Completeness</h3>
                  <p className="card-subtitle">Numerical fidelity relative to RankGauss median baseline</p>
                </div>
                <span className={`badge ${explanation.completeness.all_finite ? 'badge-low' : 'badge-critical'}`}>
                  {explanation.completeness.all_finite ? 'All Finite' : 'Non-Finite Detected'}
                </span>
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: 10,
                  fontSize: '0.84rem',
                  marginBottom: 14,
                }}
              >
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Method</div>
                  <div className="mono">{explanation.method}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Integration Steps</div>
                  <div>{integrationStepsLabel}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Completeness Residual</div>
                  <div className="mono">{explanation.completeness.abs_error.toFixed(6)} logits</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Relative Error</div>
                  <div className="mono">{explanation.completeness.rel_error_pct.toFixed(4)}%</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Logit &Delta; (x - base)</div>
                  <div className="mono">{explanation.completeness.logit_delta.toFixed(5)}</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Total IG Attribution Sum</div>
                  <div className="mono">{explanation.completeness.ig_sum.toFixed(5)}</div>
                </div>
              </div>

              <div
                style={{
                  padding: 10,
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-canvas)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '0.8rem',
                  color: 'var(--text-secondary)',
                  lineHeight: 1.45,
                }}
              >
                <strong>Plain-English Check:</strong> The completeness residual measures how closely all feature
                contributions sum to the actual change in model logit between this transaction and the baseline
                (&Delta;logit). Lower error means high numerical reliability. Self-features account for{' '}
                <strong>{(explanation.self_vs_neighbor.self_abs_share * 100).toFixed(1)}%</strong> of absolute
                attribution while graph neighbors contribute{' '}
                <strong>{(explanation.self_vs_neighbor.neighbor_abs_share * 100).toFixed(1)}%</strong>.
              </div>
            </div>
          </div>

          {/* 1-Hop & 2-Hop Neighbor Attribution */}
          <div className="card">
            <div className="card-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <GitFork size={18} color="var(--accent-teal)" />
                <div>
                  <h3 className="card-title">
                    Directed 1-Hop Neighborhood Influence ({explanation.one_hop_neighbors.length})
                  </h3>
                  <p className="card-subtitle">
                    2-hop snapshot context: {explanation.two_hop_summary.count} node(s),{' '}
                    {explanation.two_hop_summary.pred_illicit_count} predicted illicit (mean P=
                    {explanation.two_hop_summary.mean_pred_prob.toFixed(3)})
                  </p>
                </div>
              </div>
            </div>

            {explanation.one_hop_neighbors.length === 0 ? (
              <div className="empty-state" style={{ padding: 24 }}>
                This transaction is isolated (0 incoming/outgoing edges within timestep {explanation.timestep}).
              </div>
            ) : (
              <div className="table-wrap">
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Direction</th>
                      <th>Directed Edge</th>
                      <th>Neighbor Tx</th>
                      <th>Neighbor Prob</th>
                      <th>Neighbor GT</th>
                      <th>Signed IG Logit</th>
                      <th>|IG| Logit</th>
                    </tr>
                  </thead>
                  <tbody>
                    {explanation.one_hop_neighbors.map((nbr, idx) => (
                      <tr key={`${nbr.direction}_${nbr.neighbor_tx_id}_${idx}`}>
                        <td>
                          <span className="badge badge-neutral">{nbr.direction}</span>
                        </td>
                        <td className="mono">{nbr.edge_representation}</td>
                        <td className="mono">{nbr.neighbor_tx_id}</td>
                        <td>
                          <ProbBar prob={nbr.neighbor_pred_prob} />
                        </td>
                        <td>
                          <GtBadge label={nbr.neighbor_gt_label} />
                        </td>
                        <td
                          className="mono"
                          style={{ color: nbr.neighbor_signed_ig_logit >= 0 ? '#fca5a5' : '#6ee7b7' }}
                        >
                          {nbr.neighbor_signed_ig_logit >= 0
                            ? `+${nbr.neighbor_signed_ig_logit.toFixed(5)}`
                            : nbr.neighbor_signed_ig_logit.toFixed(5)}
                        </td>
                        <td className="mono">{nbr.neighbor_abs_ig_logit.toFixed(5)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
    </>
  );
}
