import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import cytoscape from 'cytoscape';
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Crosshair,
  Maximize2,
  Network,
  RotateCcw,
  ShieldAlert,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import { BackendRequestError, fetchClusterGraph } from '../api/fraudApi';
import { formatNumber, GtBadge, ProbBar, RiskBadge } from '../components/RiskBadge';
import type { ClusterGraphEdge, ClusterGraphResponse, TransactionItem } from '../types/api';

type LayoutMode = 'breadthfirst' | 'cose' | 'circle';

function deriveNodeClusterRole(
  node: TransactionItem,
  hubTxId: number,
  internalEdges: ClusterGraphEdge[],
): string {
  if (node.is_external_context) return '1-Hop External Boundary';
  if (node.tx_id === hubTxId) return 'Cluster Hub';
  let intIn = 0;
  let intOut = 0;
  for (const e of internalEdges) {
    if (e.dst_tx_id === node.tx_id) intIn += 1;
    if (e.src_tx_id === node.tx_id) intOut += 1;
  }
  if (intIn === 0 && intOut >= 1) return 'Cluster Source';
  if (intOut === 0 && intIn >= 1) return 'Cluster Sink';
  return 'Internal Relay / Member';
}

export function ClusterDetailPage(): React.ReactElement {
  const { runId = '', clusterId = '' } = useParams<{ runId: string; clusterId: string }>();
  const [graphData, setGraphData] = useState<ClusterGraphResponse | null>(null);
  const [selectedNode, setSelectedNode] = useState<TransactionItem | null>(null);
  const [layoutMode, setLayoutMode] = useState<LayoutMode>('breadthfirst');
  const [showExternalContext, setShowExternalContext] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const cyContainerRef = useRef<HTMLDivElement | null>(null);
  const cyInstanceRef = useRef<cytoscape.Core | null>(null);

  const loadGraph = useCallback(() => {
    if (!runId || !clusterId) return;
    let active = true;
    setLoading(true);
    setError(null);
    fetchClusterGraph(runId, clusterId)
      .then((res) => {
        if (!active) return;
        const hub = res.nodes.find((n) => n.tx_id === res.cluster.hub_tx_id) || res.nodes[0] || null;
        setGraphData(res);
        setSelectedNode(hub);
        setLoading(false);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof BackendRequestError ? err.message : 'Unable to load cluster graph.');
        setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [runId, clusterId]);

  useEffect(() => {
    const cleanup = loadGraph();
    return () => {
      if (cleanup) cleanup();
    };
  }, [loadGraph]);

  const visibleNodes = useMemo(() => {
    if (!graphData) return [];
    return showExternalContext
      ? graphData.nodes
      : graphData.nodes.filter((n) => !n.is_external_context);
  }, [graphData, showExternalContext]);

  const visibleEdges = useMemo(() => {
    if (!graphData) return [];
    if (showExternalContext) return graphData.edges;
    return graphData.edges.filter((e) => e.edge_type === 'internal_suspicious');
  }, [graphData, showExternalContext]);

  const internalEdges = useMemo(() => {
    if (!graphData) return [];
    return graphData.edges.filter((e) => e.edge_type === 'internal_suspicious');
  }, [graphData]);

  useEffect(() => {
    if (loading || !graphData || !cyContainerRef.current || visibleNodes.length === 0) {
      if (cyInstanceRef.current) {
        cyInstanceRef.current.removeAllListeners();
        cyInstanceRef.current.destroy();
        cyInstanceRef.current = null;
      }
      return;
    }

    const elements: cytoscape.ElementDefinition[] = [
      ...visibleNodes.map((n) => ({
        data: {
          id: String(n.tx_id),
          label: String(n.tx_id),
          prob: n.pred_prob,
          isExt: Boolean(n.is_external_context),
          isHub: n.tx_id === graphData.cluster.hub_tx_id,
        },
      })),
      ...visibleEdges.map((e, idx) => ({
        data: {
          id: `e_${idx}_${e.src_tx_id}_${e.dst_tx_id}`,
          source: String(e.src_tx_id),
          target: String(e.dst_tx_id),
          edgeType: e.edge_type,
        },
      })),
    ];

    const cy = cytoscape({
      container: cyContainerRef.current,
      elements,
      minZoom: 0.25,
      maxZoom: 3.0,
      style: [
        {
          selector: 'node',
          style: {
            label: 'data(label)',
            'font-size': '10px',
            'font-family': 'JetBrains Mono, monospace',
            color: '#e2e8f0',
            'text-valign': 'bottom',
            'text-margin-y': 5,
            width: 28,
            height: 28,
            'background-color': '#eab308',
            'border-width': 2,
            'border-color': '#0f172a',
          },
        },
        {
          selector: 'node[prob < 0.35]',
          style: {
            'background-color': '#10b981',
          },
        },
        {
          selector: 'node[prob >= 0.57][prob < 0.85]',
          style: {
            'background-color': '#f97316',
          },
        },
        {
          selector: 'node[prob >= 0.85]',
          style: {
            'background-color': '#ef4444',
          },
        },
        {
          selector: 'node[?isExt]',
          style: {
            'background-color': '#334155',
            width: 20,
            height: 20,
            'border-style': 'dashed',
            'border-color': '#94a3b8',
          },
        },
        {
          selector: 'node[?isHub]',
          style: {
            'border-color': '#2dd4bf',
            'border-width': 3.5,
            width: 36,
            height: 36,
          },
        },
        {
          selector: 'node:selected',
          style: {
            'border-color': '#f8fafc',
            'border-width': 3.5,
            'overlay-color': '#38bdf8',
            'overlay-opacity': 0.22,
          },
        },
        {
          selector: 'edge',
          style: {
            width: 2.2,
            'line-color': '#f97316',
            'target-arrow-color': '#f97316',
            'target-arrow-shape': 'triangle',
            'arrow-scale': 1.15,
            'curve-style': 'bezier',
          },
        },
        {
          selector: 'edge[edgeType != "internal_suspicious"]',
          style: {
            width: 1.5,
            'line-style': 'dashed',
            'line-color': '#64748b',
            'target-arrow-color': '#64748b',
          },
        },
      ],
      layout: {
        name: layoutMode,
        directed: true,
        padding: 28,
        animate: false,
      } as cytoscape.LayoutOptions,
    });

    cy.on('tap', 'node', (evt) => {
      const id = evt.target.id();
      const found = graphData.nodes.find((n) => String(n.tx_id) === id);
      if (found) setSelectedNode(found);
    });

    cyInstanceRef.current = cy;

    return () => {
      cy.removeAllListeners();
      cy.destroy();
      cyInstanceRef.current = null;
    };
  }, [loading, graphData, visibleNodes, visibleEdges, layoutMode]);

  const handleSelectNode = (node: TransactionItem) => {
    setSelectedNode(node);
    const cy = cyInstanceRef.current;
    if (cy) {
      cy.nodes().unselect();
      cy.getElementById(String(node.tx_id)).select();
    }
  };

  const handleZoom = (factor: number) => {
    const cy = cyInstanceRef.current;
    if (!cy) return;
    cy.zoom({
      level: cy.zoom() * factor,
      renderedPosition: {
        x: cy.width() / 2,
        y: cy.height() / 2,
      },
    });
  };

  const handleFit = () => {
    const cy = cyInstanceRef.current;
    if (cy) cy.fit(undefined, 28);
  };

  const handleResetGraph = () => {
    const cy = cyInstanceRef.current;
    if (!cy) return;
    cy.layout({
      name: layoutMode,
      directed: true,
      padding: 28,
      animate: false,
    } as cytoscape.LayoutOptions).run();
    cy.fit(undefined, 28);
  };

  if (loading) {
    return (
      <div className="card">
        <div className="empty-state">Loading cluster graph and member nodes from backend...</div>
      </div>
    );
  }

  if (error || !graphData) {
    return (
      <div className="card">
        <div className="banner banner-error">
          <AlertTriangle size={18} />
          <div style={{ flex: 1 }}>{error || 'Cluster graph not found.'}</div>
          <button type="button" className="btn btn-secondary" onClick={loadGraph}>
            Retry
          </button>
        </div>
        <div style={{ marginTop: 14 }}>
          <Link to={`/runs/${encodeURIComponent(runId)}/clusters`} className="btn btn-ghost">
            <ArrowLeft size={14} /> Back to Suspicious Clusters
          </Link>
        </div>
      </div>
    );
  }

  const { cluster, nodes, edges } = graphData;
  const internalCount = nodes.filter((n) => !n.is_external_context).length;
  const externalCount = nodes.length - internalCount;

  const selectedIncoming = selectedNode
    ? edges.filter((e) => e.dst_tx_id === selectedNode.tx_id)
    : [];
  const selectedOutgoing = selectedNode
    ? edges.filter((e) => e.src_tx_id === selectedNode.tx_id)
    : [];
  const selectedRole = selectedNode
    ? deriveNodeClusterRole(selectedNode, cluster.hub_tx_id, internalEdges)
    : '';

  return (
    <>
      <div className="page-header">
        <div>
          <Link
            to={`/runs/${encodeURIComponent(runId)}/clusters`}
            style={{ fontSize: '0.82rem', display: 'inline-flex', alignItems: 'center', gap: 5 }}
          >
            <ArrowLeft size={14} /> Back to Suspicious Clusters
          </Link>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 10, marginTop: 6 }}>
            <h2 className="page-title" style={{ margin: 0 }}>
              Cluster <span className="mono">{cluster.cluster_id}</span>
            </h2>
            {cluster.is_ring_candidate === 1 ? (
              <span className="badge badge-critical">Potential Fraud-Ring Candidate</span>
            ) : (
              <span className="badge badge-neutral">{cluster.structural_tier.replace(/_/g, ' ')}</span>
            )}
          </div>
          <p className="page-desc">
            Timestep <span className="mono">{cluster.timestep}</span> • Motif:{' '}
            <span className="mono">{cluster.motif_type}</span> • Ring Score:{' '}
            <span className="mono">{cluster.ring_score.toFixed(3)}</span> •{' '}
            {formatNumber(cluster.num_nodes)} core nodes, {formatNumber(cluster.num_internal_edges)} internal edges
            {externalCount > 0 ? `, +${externalCount} 1-hop context node(s)` : ''}
          </p>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 10 }}>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6, fontSize: '0.83rem', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={showExternalContext}
              onChange={(e) => setShowExternalContext(e.target.checked)}
            />
            <span>Show 1-Hop Context ({externalCount})</span>
          </label>
          <select
            aria-label="Graph layout selector"
            className="select"
            value={layoutMode}
            onChange={(e) => setLayoutMode(e.target.value as LayoutMode)}
          >
            <option value="breadthfirst">Directed Flow Layout</option>
            <option value="cose">Force-Directed Layout</option>
            <option value="circle">Circular Layout</option>
          </select>
        </div>
      </div>

      <div className="banner banner-warning">
        <ShieldAlert size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <strong>Model-Identified Suspicious Subgraph:</strong> Directed links (source &rarr; target) and risk
          probabilities reflect frozen <span className="mono">ExpA2_RankGauss182_BiDirGCN</span> outputs for this
          cluster only. Predicted patterns identify suspicious structural activity and are{' '}
          <strong>not proof of confirmed criminal activity</strong>.
        </div>
      </div>

      <div className="grid-2">
        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Network size={18} color="var(--accent-teal)" />
              <div>
                <h3 className="card-title">Interactive Directed Subgraph</h3>
                <p className="card-subtitle">Drag canvas to pan, scroll to zoom, or click a node to inspect</p>
              </div>
            </div>
            <div style={{ display: 'flex', gap: 6 }} role="toolbar" aria-label="Graph zoom and layout controls">
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => handleZoom(1.25)}
                aria-label="Zoom in"
                title="Zoom in"
              >
                <ZoomIn size={15} />
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => handleZoom(0.8)}
                aria-label="Zoom out"
                title="Zoom out"
              >
                <ZoomOut size={15} />
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={handleFit}
                aria-label="Fit graph to screen"
                title="Fit to screen"
              >
                <Maximize2 size={15} />
              </button>
              <button
                type="button"
                className="btn btn-ghost"
                onClick={handleResetGraph}
                aria-label="Reset graph view"
                title="Reset layout and zoom"
              >
                <RotateCcw size={15} />
              </button>
            </div>
          </div>

          {visibleNodes.length === 0 ? (
            <div className="empty-state" style={{ height: 390, justifyContent: 'center' }}>
              <Network size={36} color="var(--text-muted)" />
              <div>No graph nodes available for this cluster view.</div>
            </div>
          ) : (
            <div
              ref={cyContainerRef}
              data-testid="cluster-cytoscape-canvas"
              style={{
                width: '100%',
                height: 390,
                backgroundColor: '#070d19',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
              }}
            />
          )}

          <div
            aria-label="Graph visual legend"
            style={{
              marginTop: 12,
              padding: 10,
              borderRadius: 'var(--radius-sm)',
              background: 'var(--bg-canvas)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexWrap: 'wrap',
              gap: 14,
              fontSize: '0.76rem',
              color: 'var(--text-secondary)',
            }}
          >
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#ef4444' }} /> High Risk (&ge; 0.85)
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#f97316' }} /> Flagged (&ge; 0.57)
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', background: '#eab308' }} /> Moderate / Context
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              <span
                style={{
                  width: 11,
                  height: 11,
                  borderRadius: '50%',
                  border: '2px solid #2dd4bf',
                }}
              />{' '}
              Cluster Hub
            </span>
            <span style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
              <span style={{ width: 10, height: 10, borderRadius: '50%', border: '1px dashed #94a3b8', background: '#334155' }} /> 1-Hop Context
            </span>
            <span>Solid Arrow = Internal Suspicious Edge • Dashed Arrow = Boundary Flow</span>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Crosshair size={18} color="var(--accent-blue)" />
              <div>
                <h3 className="card-title">Selected Transaction Node</h3>
                <p className="card-subtitle">Click any node in the graph or member list</p>
              </div>
            </div>
            {selectedNode && <RiskBadge prob={selectedNode.pred_prob} riskCategory={selectedNode.risk_category} />}
          </div>

          {selectedNode ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
                <span className="mono" style={{ fontSize: '1.1rem', fontWeight: 700 }} data-testid="selected-node-txid">
                  txId: {selectedNode.tx_id}
                </span>
                <span className="badge badge-teal">{selectedRole}</span>
              </div>

              <div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: 4 }}>
                  Predicted Fraud Probability
                </div>
                <ProbBar prob={selectedNode.pred_prob} />
              </div>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                  gap: 10,
                  fontSize: '0.84rem',
                  padding: 12,
                  borderRadius: 'var(--radius-md)',
                  background: 'var(--bg-canvas)',
                  border: '1px solid var(--border-subtle)',
                }}
              >
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Predicted Class</div>
                  <strong>{selectedNode.pred_label}</strong>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Snapshot Degree (In / Out)</div>
                  <strong className="mono">
                    {selectedNode.in_deg} / {selectedNode.out_deg}
                  </strong>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Raw Model Logit</div>
                  <strong className="mono">
                    {selectedNode.logit !== undefined ? selectedNode.logit.toFixed(5) : 'N/A (Context Node)'}
                  </strong>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: '0.74rem' }}>Optional Ground Truth</div>
                  <GtBadge label={selectedNode.optional_gt_label} />
                </div>
              </div>

              <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <strong>Connected Subgraph Flows:</strong> {selectedIncoming.length} incoming edge(s),{' '}
                {selectedOutgoing.length} outgoing edge(s)
              </div>

              {!selectedNode.is_external_context ? (
                <Link
                  to={`/runs/${encodeURIComponent(runId)}/transactions/${selectedNode.tx_id}`}
                  className="btn btn-primary"
                >
                  <span>Inspect &amp; Explain Transaction (IG)</span>
                  <ArrowRight size={15} />
                </Link>
              ) : (
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  External 1-hop context node adjacent to cluster <span className="mono">{cluster.cluster_id}</span>.{' '}
                  <Link to={`/runs/${encodeURIComponent(runId)}/transactions/${selectedNode.tx_id}`}>
                    Open Transaction Detail
                  </Link>
                </div>
              )}
            </div>
          ) : (
            <div className="empty-state">Select a node to inspect transaction details.</div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Subgraph Nodes ({visibleNodes.length})</h3>
            <p className="card-subtitle">Core members ({internalCount}) and 1-hop context ({externalCount})</p>
          </div>
        </div>
        {visibleNodes.length === 0 ? (
          <div className="empty-state">No nodes in graph response.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Tx ID</th>
                  <th>Role</th>
                  <th>Probability</th>
                  <th>In / Out</th>
                  <th>Optional GT</th>
                </tr>
              </thead>
              <tbody>
                {visibleNodes.map((n) => {
                  const role = deriveNodeClusterRole(n, cluster.hub_tx_id, internalEdges);
                  const isSelected = selectedNode?.tx_id === n.tx_id;
                  return (
                    <tr
                      key={`${n.tx_id}_${n.is_external_context ? 'ext' : 'int'}`}
                      style={{
                        cursor: 'pointer',
                        background: isSelected ? 'rgba(56, 189, 248, 0.12)' : undefined,
                      }}
                      onClick={() => handleSelectNode(n)}
                    >
                      <td className="mono">{n.tx_id}</td>
                      <td>
                        <span
                          className={`badge ${
                            n.is_external_context
                              ? 'badge-neutral'
                              : n.tx_id === cluster.hub_tx_id
                                ? 'badge-teal'
                                : 'badge-elevated'
                          }`}
                        >
                          {role}
                        </span>
                      </td>
                      <td>
                        <ProbBar prob={n.pred_prob} />
                      </td>
                      <td className="mono">
                        {n.in_deg} / {n.out_deg}
                      </td>
                      <td>
                        <GtBadge label={n.optional_gt_label} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">Directed Edges ({visibleEdges.length})</h3>
            <p className="card-subtitle">Explicit source &rarr; destination payment links</p>
          </div>
        </div>
        {visibleEdges.length === 0 ? (
          <div className="empty-state">No directed edges recorded for this cluster view.</div>
        ) : (
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Directed Flow (Src &rarr; Dst)</th>
                  <th>Edge Type</th>
                  <th>Src Prob</th>
                  <th>Dst Prob</th>
                </tr>
              </thead>
              <tbody>
                {visibleEdges.map((e, idx) => (
                  <tr key={`${e.src_tx_id}_${e.dst_tx_id}_${idx}`}>
                    <td className="mono">
                      {e.src_tx_id} &rarr; {e.dst_tx_id}
                    </td>
                    <td>
                      <span
                        className={`badge ${
                          e.edge_type === 'internal_suspicious' ? 'badge-elevated' : 'badge-neutral'
                        }`}
                      >
                        {e.edge_type}
                      </span>
                    </td>
                    <td className="mono">{e.src_pred_prob.toFixed(4)}</td>
                    <td className="mono">{e.dst_pred_prob.toFixed(4)}</td>
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
