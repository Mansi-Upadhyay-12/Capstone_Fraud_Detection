import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent, cleanup } from '@testing-library/react';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { TransactionDetailPage } from '../pages/TransactionDetailPage';
import * as fraudApi from '../api/fraudApi';
import type { TransactionExplanationResponse, TransactionItem } from '../types/api';

const SAMPLE_TX: TransactionItem = {
  run_id: 'run_step42_demo',
  node_order: 1,
  tx_id: 71989375,
  timestep: 42,
  snapshot_node_idx: 10,
  pred_prob: 0.999976,
  logit: 10.625143,
  pred_class: 1,
  pred_label: 'Illicit',
  risk_category: 'High Predicted Risk (P >= 0.85)',
  in_deg: 1,
  out_deg: 2,
  cluster_id: 'C_T42_001',
  optional_gt_code: -1,
  optional_gt_label: 'Unknown',
};

const SAMPLE_EXPLANATION: TransactionExplanationResponse = {
  run_id: 'run_step42_demo',
  tx_id: 71989375,
  timestep: 42,
  created_at: '2026-11-27T01:00:00Z',
  cache_hit: false,
  method: 'Composite_Midpoint_IG_3Zone_M200',
  baseline: 'RankGauss zero vector (training median baseline)',
  model_id: 'ExpA2_RankGauss182_BiDirGCN',
  threshold_used: 0.57,
  prediction: {
    pred_prob: 0.999976,
    logit: 10.625143,
    pred_label: 'Illicit',
    risk_category: 'High Predicted Risk (P >= 0.85)',
    cluster_id: 'C_T42_001',
    optional_gt_label: 'Unknown',
  },
  completeness: {
    logit_x: 10.625143,
    logit_baseline: 0.0,
    logit_delta: 10.625143,
    ig_sum: 10.622905,
    abs_error: 0.002238,
    rel_error_pct: 0.0211,
    all_finite: true,
  },
  self_vs_neighbor: {
    self_signed_logit: 7.51,
    neighbor_signed_logit: 3.112905,
    self_abs_share: 0.707,
    neighbor_abs_share: 0.293,
    one_hop_unique_count: 1,
    incoming_count: 1,
    outgoing_count: 0,
  },
  feature_group_breakdown: [
    { group: 'Local93', feature_count: 93, signed_sum: 6.12, abs_sum: 7.45, self_signed_sum: 4.5 },
    { group: 'Aggregate72', feature_count: 72, signed_sum: 3.21, abs_sum: 4.1, self_signed_sum: 2.2 },
    { group: 'Domain17', feature_count: 17, signed_sum: 1.292905, abs_sum: 1.85, self_signed_sum: 0.81 },
  ],
  top_positive_features: [
    {
      feature_index: 110,
      feature_name: 'Aggregate_feature_18',
      feature_group: 'Aggregate72',
      target_rankgauss_value: 2.15,
      graph_ig_attribution: 0.81234,
      self_ig_attribution: 0.601,
      neighbor_ig_attribution: 0.21134,
    },
  ],
  top_negative_features: [
    {
      feature_index: 3,
      feature_name: 'Local_feature_4',
      feature_group: 'Local93',
      target_rankgauss_value: -1.42,
      graph_ig_attribution: -0.3125,
      self_ig_attribution: -0.25,
      neighbor_ig_attribution: -0.0625,
    },
  ],
  one_hop_neighbors: [
    {
      direction: 'incoming',
      edge_representation: '71989300 -> 71989375',
      neighbor_tx_id: 71989300,
      neighbor_pred_prob: 0.88,
      neighbor_pred_class: 'Illicit',
      neighbor_gt_label: 'Unknown',
      neighbor_in_deg: 1,
      neighbor_out_deg: 1,
      neighbor_signed_ig_logit: 1.94,
      neighbor_abs_ig_logit: 2.12,
    },
  ],
  two_hop_summary: {
    count: 3,
    pred_illicit_count: 2,
    pred_licit_count: 1,
    pred_illicit_rate: 0.6667,
    mean_pred_prob: 0.65,
    gt_illicit_count: 0,
    gt_licit_count: 0,
    gt_unknown_count: 3,
  },
  disclaimer: 'Integrated Gradients explains model logit sensitivity in RankGauss-normalized space.',
};

function renderPage(): void {
  render(
    <MemoryRouter initialEntries={['/runs/run_step42_demo/transactions/71989375']}>
      <Routes>
        <Route path="/runs/:runId/transactions/:txId" element={<TransactionDetailPage />} />
      </Routes>
    </MemoryRouter>,
  );
}

describe('TransactionDetailPage', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('loads risk summary without auto-calling explain, then fetches and displays IG attributions on demand', async () => {
    vi.spyOn(fraudApi, 'fetchRunTransactions').mockResolvedValueOnce({
      total: 1,
      limit: 50,
      offset: 0,
      items: [SAMPLE_TX],
    });
    const explainSpy = vi
      .spyOn(fraudApi, 'explainTransaction')
      .mockResolvedValueOnce(SAMPLE_EXPLANATION);

    renderPage();

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Transaction 71989375/i })).toBeInTheDocument();
    });

    // Must NOT automatically compute explanations before button click
    expect(explainSpy).not.toHaveBeenCalled();
    expect(screen.getByText(/On-Demand Feature Attribution Ready/i)).toBeInTheDocument();
    expect(screen.getByText(/Illicit \(Predicted\)/i)).toBeInTheDocument();
    expect(screen.getByText('Unknown')).toBeInTheDocument();

    // Request explanation on demand
    const explainButtons = screen.getAllByRole('button', { name: /Explain Prediction \(IG\)/i });
    fireEvent.click(explainButtons[0]);

    await waitFor(() => {
      expect(screen.getByText('Composite_Midpoint_IG_3Zone_M200')).toBeInTheDocument();
    });

    expect(explainSpy).toHaveBeenCalledTimes(1);
    expect(explainSpy).toHaveBeenCalledWith('run_step42_demo', '71989375');

    // Positive & Negative features rendered (may appear in both chart & table)
    expect(screen.getAllByText('Aggregate_feature_18').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Local_feature_4').length).toBeGreaterThanOrEqual(1);

    // Feature groups rendered
    expect(screen.getByText('Local Features (93)')).toBeInTheDocument();
    expect(screen.getByText('Aggregate Features (72)')).toBeInTheDocument();
    expect(screen.getByText('Domain Features (17)')).toBeInTheDocument();

    // Completeness metrics rendered
    expect(screen.getByText(/M = 200/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.002238 logits/i)).toBeInTheDocument();
    expect(screen.getByText(/0\.0211%/i)).toBeInTheDocument();

    // 1-hop directed neighbor rendered
    expect(screen.getByText('71989300 -> 71989375')).toBeInTheDocument();
  });

  it('displays API explanation errors cleanly and allows retry', async () => {
    vi.spyOn(fraudApi, 'fetchRunTransactions').mockResolvedValueOnce({
      total: 1,
      limit: 50,
      offset: 0,
      items: [SAMPLE_TX],
    });
    const explainSpy = vi
      .spyOn(fraudApi, 'explainTransaction')
      .mockRejectedValueOnce(
        new fraudApi.BackendRequestError(
          'Tensor cache expired for timestep 42.',
          409,
          'snapshot_cache_expired',
        ),
      )
      .mockResolvedValueOnce({ ...SAMPLE_EXPLANATION, cache_hit: true });

    renderPage();

    await waitFor(() => {
      expect(screen.getByRole('heading', { name: /Transaction 71989375/i })).toBeInTheDocument();
    });

    fireEvent.click(screen.getAllByRole('button', { name: /Explain Prediction \(IG\)/i })[0]);

    await waitFor(() => {
      expect(screen.getByText(/Tensor cache expired/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole('button', { name: /Retry Explanation/i }));

    await waitFor(() => {
      expect(screen.getByText(/Cached SQLite Result/i)).toBeInTheDocument();
    });
    expect(explainSpy).toHaveBeenCalledTimes(2);
  });
});
