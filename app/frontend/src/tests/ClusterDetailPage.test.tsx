import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { ClusterDetailPage } from '../pages/ClusterDetailPage';
import * as fraudApi from '../api/fraudApi';
import type { ClusterGraphResponse } from '../types/api';

const cySpies = vi.hoisted(() => ({
  zoom: vi.fn(() => 1),
  width: vi.fn(() => 800),
  height: vi.fn(() => 400),
  fit: vi.fn(),
  layoutRun: vi.fn(),
  unselect: vi.fn(),
  select: vi.fn(),
  removeAllListeners: vi.fn(),
  destroy: vi.fn(),
  on: vi.fn(),
}));

vi.mock('cytoscape', () => ({
  default: vi.fn(() => ({
    zoom: cySpies.zoom,
    width: cySpies.width,
    height: cySpies.height,
    fit: cySpies.fit,
    layout: () => ({ run: cySpies.layoutRun }),
    nodes: () => ({ unselect: cySpies.unselect }),
    getElementById: () => ({ select: cySpies.select }),
    on: cySpies.on,
    removeAllListeners: cySpies.removeAllListeners,
    destroy: cySpies.destroy,
  })),
}));

const SAMPLE_GRAPH: ClusterGraphResponse = {
  run_id: 'run_step42_demo',
  cluster: {
    cluster_id: 'C_T42_001',
    timestep: 42,
    structural_tier: 'Potential_Fraud_Ring_Candidate',
    is_ring_candidate: 1,
    motif_type: 'fan_out_dispersion',
    ring_score: 4.912,
    num_nodes: 2,
    num_internal_edges: 1,
    directed_edge_density: 0.5,
    has_directed_cycle: 0,
    max_directed_path_len: 1,
    max_internal_in_deg: 1,
    max_internal_out_deg: 1,
    ext_1hop_in_neighbors: 0,
    ext_1hop_out_neighbors: 0,
    mean_pred_prob: 0.89,
    max_pred_prob: 0.94,
    min_pred_prob: 0.84,
    num_high_conf_ge_085: 1,
    hub_tx_id: 71989375,
    hub_node_idx: 10,
    hub_pred_prob: 0.94,
    post_hoc_gt_illicit: 1,
    post_hoc_gt_licit: 0,
    post_hoc_gt_unknown: 1,
    post_hoc_gt_labeled: 1,
    has_labeled_gt: 1,
    post_hoc_labeled_illicit_precision: 1.0,
    post_hoc_eval_status: 'Labeled_Mixed_or_Pure',
  },
  nodes: [
    {
      tx_id: 71989375,
      timestep: 42,
      snapshot_node_idx: 10,
      pred_prob: 0.94,
      logit: 2.751,
      pred_class: 1,
      pred_label: 'Illicit',
      risk_category: 'High Predicted Risk (P >= 0.85)',
      in_deg: 0,
      out_deg: 2,
      cluster_id: 'C_T42_001',
      optional_gt_code: 1,
      optional_gt_label: 'Illicit',
      is_external_context: false,
    },
    {
      tx_id: 71989399,
      timestep: 42,
      snapshot_node_idx: 11,
      pred_prob: 0.84,
      logit: 1.658,
      pred_class: 1,
      pred_label: 'Illicit',
      risk_category: 'Elevated Risk (Flagged >= 0.57)',
      in_deg: 1,
      out_deg: 0,
      cluster_id: 'C_T42_001',
      optional_gt_code: -1,
      optional_gt_label: 'Unknown',
      is_external_context: false,
    },
  ],
  edges: [
    {
      cluster_id: 'C_T42_001',
      timestep: 42,
      src_tx_id: 71989375,
      dst_tx_id: 71989399,
      edge_type: 'internal_suspicious',
      src_pred_prob: 0.94,
      dst_pred_prob: 0.84,
    },
  ],
};

function renderPage(): void {
  render(
    <MemoryRouter initialEntries={['/runs/run_step42_demo/clusters/C_T42_001']}>
      <Routes>
        <Route path="/runs/:runId/clusters/:clusterId" element={<ClusterDetailPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('ClusterDetailPage', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
    Object.values(cySpies).forEach((fn) => fn.mockClear());
  });

  it('shows loading state and then renders directed graph, legend, controls, and node selection', async () => {
    vi.spyOn(fraudApi, 'fetchClusterGraph').mockResolvedValueOnce(SAMPLE_GRAPH);
    renderPage();

    expect(screen.getByText(/Loading cluster graph/i)).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('C_T42_001')).toBeInTheDocument();
    });

    expect(screen.getByText('Potential Fraud-Ring Candidate')).toBeInTheDocument();
    expect(screen.getByText(/not proof of confirmed criminal activity/i)).toBeInTheDocument();
    expect(screen.getByTestId('cluster-cytoscape-canvas')).toBeInTheDocument();
    expect(screen.getByLabelText('Graph visual legend')).toBeInTheDocument();

    // Default selected hub node
    expect(screen.getByTestId('selected-node-txid')).toHaveTextContent('txId: 71989375');

    // Directed edge displayed explicitly with source -> destination
    expect(screen.getByText('71989375 → 71989399')).toBeInTheDocument();

    // Wait for Cytoscape passive useEffect initialization before clicking graph toolbar controls
    await waitFor(() => {
      expect(cySpies.on).toHaveBeenCalled();
    });

    // Zoom, fit, reset buttons invoke Cytoscape methods
    fireEvent.click(screen.getByLabelText('Zoom in'));
    expect(cySpies.zoom).toHaveBeenCalled();

    fireEvent.click(screen.getByLabelText('Fit graph to screen'));
    expect(cySpies.fit).toHaveBeenCalled();

    fireEvent.click(screen.getByLabelText('Reset graph view'));
    expect(cySpies.layoutRun).toHaveBeenCalled();

    // Clicking secondary node in table updates Selected Transaction Node panel
    fireEvent.click(screen.getByText('71989399'));
    expect(screen.getByTestId('selected-node-txid')).toHaveTextContent('txId: 71989399');
    expect(screen.getAllByText('Cluster Sink').length).toBeGreaterThanOrEqual(2);

    cleanup();
    expect(cySpies.destroy).toHaveBeenCalled();
  });

  it('handles API error and allows retrying', async () => {
    const spy = vi
      .spyOn(fraudApi, 'fetchClusterGraph')
      .mockRejectedValueOnce(new fraudApi.BackendRequestError('Cluster not found.', 404, 'cluster_not_found'))
      .mockResolvedValueOnce(SAMPLE_GRAPH);

    renderPage();

    await waitFor(() => {
      expect(screen.getByText('Cluster not found.')).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /Retry/i }));
    await waitFor(() => {
      expect(screen.getByText('C_T42_001')).toBeInTheDocument();
    });
    expect(spy).toHaveBeenCalledTimes(2);
  });

  it('renders clean empty graph state when backend returns 0 nodes and 0 edges', async () => {
    vi.spyOn(fraudApi, 'fetchClusterGraph').mockResolvedValueOnce({
      ...SAMPLE_GRAPH,
      nodes: [],
      edges: [],
    });

    renderPage();

    await waitFor(() => {
      expect(screen.getByText(/No graph nodes available/i)).toBeInTheDocument();
    });
    expect(screen.getByText(/No directed edges recorded/i)).toBeInTheDocument();
  });
});
