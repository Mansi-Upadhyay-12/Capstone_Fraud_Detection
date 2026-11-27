import React, { lazy, Suspense, useCallback, useEffect, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { BackendRequestError, fetchHealth, fetchRuns } from './api/fraudApi';
import { AppLayout } from './components/AppLayout';
import type { HealthResponse, RunSummaryItem } from './types/api';

const DashboardPage = lazy(() =>
  import('./pages/DashboardPage').then((m) => ({ default: m.DashboardPage })),
);
const UploadPage = lazy(() =>
  import('./pages/UploadPage').then((m) => ({ default: m.UploadPage })),
);
const RunDetailsPage = lazy(() =>
  import('./pages/RunDetailsPage').then((m) => ({ default: m.RunDetailsPage })),
);
const TransactionsPage = lazy(() =>
  import('./pages/TransactionsPage').then((m) => ({ default: m.TransactionsPage })),
);
const TransactionDetailPage = lazy(() =>
  import('./pages/TransactionDetailPage').then((m) => ({ default: m.TransactionDetailPage })),
);
const ClustersPage = lazy(() =>
  import('./pages/ClustersPage').then((m) => ({ default: m.ClustersPage })),
);
const ClusterDetailPage = lazy(() =>
  import('./pages/ClusterDetailPage').then((m) => ({ default: m.ClusterDetailPage })),
);

function RouteLoadingFallback(): React.ReactElement {
  return (
    <div className="card">
      <div className="empty-state">Loading view...</div>
    </div>
  );
}

export default function App(): React.ReactElement {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [runs, setRuns] = useState<RunSummaryItem[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [runsError, setRunsError] = useState<string | null>(null);

  const refreshHealth = useCallback(() => {
    setHealthLoading(true);
    setHealthError(null);
    fetchHealth()
      .then((res) => setHealth(res))
      .catch((err) => {
        setHealth(null);
        setHealthError(err instanceof BackendRequestError ? err.message : 'Backend connection failed.');
      })
      .finally(() => setHealthLoading(false));
  }, []);

  const refreshRuns = useCallback(() => {
    setRunsLoading(true);
    setRunsError(null);
    fetchRuns()
      .then((list) => setRuns(list))
      .catch((err) => {
        setRunsError(err instanceof BackendRequestError ? err.message : 'Unable to fetch analysis runs.');
      })
      .finally(() => setRunsLoading(false));
  }, []);

  useEffect(() => {
    refreshHealth();
    refreshRuns();
  }, [refreshHealth, refreshRuns]);

  return (
    <BrowserRouter>
      <AppLayout
        health={health}
        healthError={healthError}
        healthLoading={healthLoading}
        onRefreshHealth={refreshHealth}
        runs={runs}
      >
        <Suspense fallback={<RouteLoadingFallback />}>
          <Routes>
            <Route
              path="/"
              element={
                <DashboardPage
                  runs={runs}
                  runsLoading={runsLoading}
                  runsError={runsError}
                  onRetryRuns={refreshRuns}
                  health={health}
                />
              }
            />
            <Route path="/upload" element={<UploadPage onAnalysisCompleted={refreshRuns} />} />
            <Route path="/runs/:runId" element={<RunDetailsPage />} />
            <Route path="/runs/:runId/transactions" element={<TransactionsPage />} />
            <Route path="/runs/:runId/transactions/:txId" element={<TransactionDetailPage />} />
            <Route path="/runs/:runId/clusters" element={<ClustersPage />} />
            <Route path="/runs/:runId/clusters/:clusterId" element={<ClusterDetailPage />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </AppLayout>
    </BrowserRouter>
  );
}
