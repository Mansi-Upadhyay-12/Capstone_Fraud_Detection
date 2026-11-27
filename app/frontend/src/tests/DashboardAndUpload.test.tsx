import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { DashboardPage } from '../pages/DashboardPage';
import { UploadPage } from '../pages/UploadPage';
import * as fraudApi from '../api/fraudApi';

describe('DashboardPage empty and error states', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('renders clean empty state with upload CTA when 0 runs exist', () => {
    render(
      <MemoryRouter>
        <DashboardPage
          runs={[]}
          runsLoading={false}
          runsError={null}
          onRetryRuns={vi.fn()}
          health={null}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText(/No Analysis Runs Recorded Yet/i)).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /Upload CSV Bundle/i })).toHaveAttribute('href', '/upload');
    expect(screen.getByText(/not connected to a live blockchain stream/i)).toBeInTheDocument();
  });

  it('renders error banner and triggers retry when loading runs fails', () => {
    const onRetry = vi.fn();
    render(
      <MemoryRouter>
        <DashboardPage
          runs={[]}
          runsLoading={false}
          runsError="Unable to reach backend"
          onRetryRuns={onRetry}
          health={null}
        />
      </MemoryRouter>,
    );

    expect(screen.getByText(/Unable to reach backend/i)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Retry/i }));
    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});

describe('UploadPage drag-and-drop and validation gating', () => {
  beforeEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it('supports drag-and-drop selection, rejects wrong extensions, and keeps Analyze disabled until validated', async () => {
    const valSpy = vi.spyOn(fraudApi, 'validateCsvBundle').mockResolvedValueOnce({
      valid: true,
      node_count: 10,
      edge_count: 4,
      unique_edges_after_dedup: 4,
      duplicate_edges_removed: 0,
      timesteps: [42],
      imputed_nan_cells: 0,
      has_optional_class_column: false,
      class_counts: { illicit: 0, licit: 0, unknown: 10 },
      warnings: [],
    });

    render(
      <MemoryRouter>
        <UploadPage onAnalysisCompleted={vi.fn()} />
      </MemoryRouter>,
    );

    const txZone = screen.getByTestId('transactions-file-input').parentElement!;
    const edgeZone = screen.getByTestId('edges-file-input').parentElement!;

    // Wrong file extension via drop shows user-friendly warning
    const badFile = new File(['hello'], 'notes.json', { type: 'application/json' });
    fireEvent.drop(txZone, { dataTransfer: { files: [badFile] } });
    expect(screen.getByText(/Please upload a \.csv or \.txt file/i)).toBeInTheDocument();

    // Valid files via drop
    const txCsv = new File(['txId,Time step\n1,42'], 'txs.csv', { type: 'text/csv' });
    const edgeCsv = new File(['txId1,txId2\n1,1'], 'edges.csv', { type: 'text/csv' });
    fireEvent.drop(txZone, { dataTransfer: { files: [txCsv] } });
    fireEvent.drop(edgeZone, { dataTransfer: { files: [edgeCsv] } });

    const analyzeBtn = screen.getByRole('button', { name: /Run Fraud Analysis/i });
    expect(analyzeBtn).toBeDisabled();

    fireEvent.click(screen.getByRole('button', { name: /Validate CSV Bundle/i }));
    await waitFor(() => {
      expect(screen.getByText(/Validation Passed/i)).toBeInTheDocument();
    });
    expect(valSpy).toHaveBeenCalledTimes(1);
    expect(analyzeBtn).toBeEnabled();
  });
});
