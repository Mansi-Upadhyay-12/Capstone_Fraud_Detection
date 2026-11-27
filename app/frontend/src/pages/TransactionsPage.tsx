import React, { useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import {
  AlertTriangle,
  ArrowLeft,
  ArrowUpDown,
  ChevronLeft,
  ChevronRight,
  Search,
} from 'lucide-react';
import {
  BackendRequestError,
  fetchRunDetail,
  fetchRunTransactions,
} from '../api/fraudApi';
import { formatNumber, GtBadge, ProbBar, RiskBadge } from '../components/RiskBadge';
import type { RunDetail, TransactionItem } from '../types/api';

type SortKey = 'pred_prob' | 'tx_id' | 'timestep' | 'in_deg' | 'out_deg';

export function TransactionsPage(): React.ReactElement {
  const { runId = '' } = useParams<{ runId: string }>();
  const navigate = useNavigate();

  const [runDetail, setRunDetail] = useState<RunDetail | null>(null);
  const [items, setItems] = useState<TransactionItem[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [searchInput, setSearchInput] = useState('');
  const [searchApplied, setSearchApplied] = useState('');
  const [searchNotice, setSearchNotice] = useState<string | null>(null);
  const [timestepFilter, setTimestepFilter] = useState<number | null>(null);
  const [riskBandFilter, setRiskBandFilter] = useState<'all' | 'flagged' | 'high' | 'moderate'>('all');
  const [limit, setLimit] = useState(50);
  const [offset, setOffset] = useState(0);

  const [sortKey, setSortKey] = useState<SortKey>('pred_prob');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  useEffect(() => {
    if (!runId) return;
    fetchRunDetail(runId)
      .then((d) => setRunDetail(d))
      .catch(() => setRunDetail(null));
  }, [runId]);

  const loadTransactions = () => {
    if (!runId) return;
    setLoading(true);
    setError(null);

    let minProb = 0;
    let flaggedOnly = false;
    if (riskBandFilter === 'flagged') {
      flaggedOnly = true;
    } else if (riskBandFilter === 'high') {
      minProb = 0.85;
    } else if (riskBandFilter === 'moderate') {
      minProb = 0.35;
    }

    fetchRunTransactions(runId, {
      timestep: timestepFilter,
      flaggedOnly,
      minProb,
      searchTxId: searchApplied,
      limit,
      offset,
    })
      .then((res) => {
        setItems(res.items);
        setTotal(res.total);
      })
      .catch((err) => {
        setError(err instanceof BackendRequestError ? err.message : 'Unable to fetch transactions.');
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadTransactions();
  }, [runId, timestepFilter, riskBandFilter, searchApplied, limit, offset]);

  const sortedItems = useMemo(() => {
    const copy = [...items];
    copy.sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av === bv) return 0;
      return sortDir === 'asc' ? (av < bv ? -1 : 1) : av > bv ? -1 : 1;
    });
    return copy;
  }, [items, sortKey, sortDir]);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir((d) => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const cleaned = searchInput.trim();
    if (cleaned && !/^\d+$/.test(cleaned)) {
      setSearchNotice('Transaction ID search expects digits only (e.g. 71989375).');
      return;
    }
    setSearchNotice(null);
    setOffset(0);
    setSearchApplied(cleaned);
  };

  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.max(1, Math.ceil(total / limit));

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
            Transaction Risk Table
          </h2>
          <p className="page-desc">
            Server-filtered model predictions for run <span className="mono">{runId}</span>
            {runDetail ? ` (${runDetail.dataset_name})` : ''}. Probabilities reflect frozen BiDirGCN outputs.
          </p>
        </div>
        <Link to={`/runs/${encodeURIComponent(runId)}/clusters`} className="btn btn-secondary">
          View Suspicious Clusters
        </Link>
      </div>

      <div className="card">
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: 12,
            marginBottom: 16,
          }}
        >
          <form onSubmit={handleSearchSubmit} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div style={{ position: 'relative' }}>
              <input
                className="input"
                type="text"
                placeholder="Search numeric txId..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                style={{ paddingLeft: 32, width: 210 }}
              />
              <Search
                size={15}
                color="var(--text-muted)"
                style={{ position: 'absolute', left: 10, top: 10, pointerEvents: 'none' }}
              />
            </div>
            <button type="submit" className="btn btn-secondary">
              Search
            </button>
            {searchApplied && (
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => {
                  setSearchInput('');
                  setSearchApplied('');
                  setSearchNotice(null);
                  setOffset(0);
                }}
              >
                Clear
              </button>
            )}
          </form>

          <select
            aria-label="Filter by risk category"
            className="select"
            value={riskBandFilter}
            onChange={(e) => {
              setRiskBandFilter(e.target.value as 'all' | 'flagged' | 'high' | 'moderate');
              setOffset(0);
            }}
          >
            <option value="all">All Risk Levels</option>
            <option value="flagged">Flagged Illicit (&ge; 0.57)</option>
            <option value="high">High Predicted Risk (&ge; 0.85)</option>
            <option value="moderate">Moderate or Higher (&ge; 0.35)</option>
          </select>

          <select
            aria-label="Filter by timestep"
            className="select"
            value={timestepFilter ?? ''}
            onChange={(e) => {
              setTimestepFilter(e.target.value ? Number(e.target.value) : null);
              setOffset(0);
            }}
          >
            <option value="">All Timesteps</option>
            {(runDetail?.summary.timesteps || []).map((t) => (
              <option key={t} value={t}>
                Timestep {t}
              </option>
            ))}
          </select>

          <select
            aria-label="Rows per page"
            className="select"
            value={limit}
            onChange={(e) => {
              setLimit(Number(e.target.value));
              setOffset(0);
            }}
          >
            <option value={25}>25 / page</option>
            <option value={50}>50 / page</option>
            <option value={100}>100 / page</option>
          </select>

          <span style={{ marginLeft: 'auto', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Matching: <strong>{formatNumber(total)}</strong> transaction(s)
          </span>
        </div>

        {searchNotice && (
          <div className="banner banner-warning" style={{ marginBottom: 14 }}>
            <AlertTriangle size={16} />
            <span>{searchNotice}</span>
          </div>
        )}

        {error && (
          <div className="banner banner-error" style={{ marginBottom: 14 }}>
            <AlertTriangle size={18} />
            <div style={{ flex: 1 }}>{error}</div>
            <button type="button" className="btn btn-secondary" onClick={loadTransactions}>
              Retry
            </button>
          </div>
        )}

        {loading ? (
          <div className="empty-state">Loading transactions from backend SQLite index...</div>
        ) : sortedItems.length === 0 ? (
          <div className="empty-state">
            No transactions match the selected filters in this run.
          </div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>
                    <button type="button" className="btn btn-ghost" onClick={() => toggleSort('tx_id')}>
                      Transaction ID <ArrowUpDown size={13} />
                    </button>
                  </th>
                  <th>
                    <button type="button" className="btn btn-ghost" onClick={() => toggleSort('timestep')}>
                      Timestep <ArrowUpDown size={13} />
                    </button>
                  </th>
                  <th>
                    <button type="button" className="btn btn-ghost" onClick={() => toggleSort('pred_prob')}>
                      Fraud Probability <ArrowUpDown size={13} />
                    </button>
                  </th>
                  <th>Predicted Class</th>
                  <th>Risk Category</th>
                  <th>
                    <button type="button" className="btn btn-ghost" onClick={() => toggleSort('in_deg')}>
                      In/Out Deg <ArrowUpDown size={13} />
                    </button>
                  </th>
                  <th>Cluster</th>
                  <th>Optional Label</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {sortedItems.map((tx) => (
                  <tr key={tx.tx_id}>
                    <td className="mono">{tx.tx_id}</td>
                    <td className="mono">{tx.timestep}</td>
                    <td>
                      <ProbBar prob={tx.pred_prob} />
                    </td>
                    <td>
                      <span className={`badge ${tx.pred_class === 1 ? 'badge-elevated' : 'badge-low'}`}>
                        {tx.pred_label}
                      </span>
                    </td>
                    <td>
                      <RiskBadge prob={tx.pred_prob} riskCategory={tx.risk_category} />
                    </td>
                    <td className="mono">
                      {tx.in_deg} / {tx.out_deg}
                    </td>
                    <td>
                      {tx.cluster_id ? (
                        <Link
                          to={`/runs/${encodeURIComponent(runId)}/clusters/${encodeURIComponent(tx.cluster_id)}`}
                          className="mono"
                        >
                          {tx.cluster_id}
                        </Link>
                      ) : (
                        <span style={{ color: 'var(--text-muted)' }}>—</span>
                      )}
                    </td>
                    <td>
                      <GtBadge label={tx.optional_gt_label} />
                    </td>
                    <td>
                      <button
                        type="button"
                        className="btn btn-ghost"
                        onClick={() =>
                          navigate(`/runs/${encodeURIComponent(runId)}/transactions/${tx.tx_id}`)
                        }
                      >
                        Detail
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            marginTop: 16,
            fontSize: '0.84rem',
            color: 'var(--text-secondary)',
          }}
        >
          <span>
            Page {currentPage} of {totalPages}
          </span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={offset === 0 || loading}
              onClick={() => setOffset(Math.max(0, offset - limit))}
            >
              <ChevronLeft size={15} /> Previous
            </button>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={offset + limit >= total || loading}
              onClick={() => setOffset(offset + limit)}
            >
              Next <ChevronRight size={15} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
