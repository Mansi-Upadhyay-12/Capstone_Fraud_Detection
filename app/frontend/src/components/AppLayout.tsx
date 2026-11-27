import React, { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Activity,
  FileUp,
  LayoutDashboard,
  Menu,
  Network,
  RefreshCw,
  ShieldAlert,
  Table2,
  X,
} from 'lucide-react';
import type { HealthResponse, RunSummaryItem } from '../types/api';

interface AppLayoutProps {
  children: React.ReactNode;
  health: HealthResponse | null;
  healthError: string | null;
  healthLoading: boolean;
  onRefreshHealth: () => void;
  runs: RunSummaryItem[];
}

export function AppLayout({
  children,
  health,
  healthError,
  healthLoading,
  onRefreshHealth,
  runs,
}: AppLayoutProps): React.ReactElement {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const location = useLocation();

  const matchRun = location.pathname.match(/^\/runs\/([^/]+)/);
  const activeRunId = matchRun ? decodeURIComponent(matchRun[1]) : runs[0]?.run_id || null;

  const statusDotClass = healthLoading ? 'checking' : health ? 'online' : 'offline';
  const statusText = healthLoading
    ? 'Checking API...'
    : health
      ? `API Online • Threshold ${health.threshold_used.toFixed(2)}`
      : 'Backend Offline';

  const closeMobile = () => setSidebarOpen(false);

  return (
    <div className="app-shell">
      {sidebarOpen && (
        <div
          className="mobile-backdrop"
          onClick={closeMobile}
          aria-hidden="true"
        />
      )}
      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`} aria-label="Primary Navigation">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div className="brand-block">
            <div className="brand-icon">
              <ShieldAlert size={20} />
            </div>
            <div>
              <h1 className="brand-title">FraudLens</h1>
              <p className="brand-subtitle">Graph-Based Fraud Detection &amp; Explainable AI</p>
            </div>
          </div>
          <button
            type="button"
            className="mobile-menu-btn"
            onClick={closeMobile}
            aria-label="Close navigation menu"
          >
            <X size={16} />
          </button>
        </div>

        <div>
          <div className="nav-section-label">Workspace</div>
          <nav className="nav-list">
            <NavLink
              to="/"
              end
              onClick={closeMobile}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <LayoutDashboard size={17} />
              <span>Dashboard</span>
            </NavLink>
            <NavLink
              to="/upload"
              onClick={closeMobile}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
            >
              <FileUp size={17} />
              <span>Upload CSV Bundle</span>
            </NavLink>
          </nav>
        </div>

        <div>
          <div className="nav-section-label">
            {matchRun ? 'Selected Run' : runs.length > 0 ? 'Latest Run' : 'Investigation Views'}
          </div>
          {activeRunId ? (
            <nav className="nav-list">
              <NavLink
                to={`/runs/${encodeURIComponent(activeRunId)}`}
                end
                onClick={closeMobile}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                <Activity size={17} />
                <span>Run Overview</span>
              </NavLink>
              <NavLink
                to={`/runs/${encodeURIComponent(activeRunId)}/transactions`}
                onClick={closeMobile}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                <Table2 size={17} />
                <span>Transaction Risk Table</span>
              </NavLink>
              <NavLink
                to={`/runs/${encodeURIComponent(activeRunId)}/clusters`}
                onClick={closeMobile}
                className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              >
                <Network size={17} />
                <span>Suspicious Clusters</span>
              </NavLink>
            </nav>
          ) : (
            <div style={{ padding: '8px 10px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Upload and analyse a CSV bundle to unlock run exploration.
            </div>
          )}
        </div>

        <div className="sidebar-footer">
          <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
            Model: <span className="mono">{health?.model_id || 'ExpA2_RankGauss182_BiDirGCN'}</span>
          </div>
          <div>
            Mode: Batch CSV upload only. Not connected to a live blockchain stream.
          </div>
        </div>
      </aside>

      <div className="main-column">
        <header className="topbar">
          <div className="topbar-left">
            <button
              type="button"
              className="mobile-menu-btn"
              onClick={() => setSidebarOpen((v) => !v)}
              aria-label="Toggle navigation menu"
            >
              <Menu size={18} />
            </button>
            <div className="topbar-title-group">
              <div className="topbar-app-name">
                FraudLens — Intelligent Transaction Risk Analysis
              </div>
              <div className="topbar-app-sub">
                Graph-Based Fraud Detection &amp; Explainable AI
              </div>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              className="status-pill"
              title={
                healthError ||
                (health ? `Device: ${health.device} | Features: ${health.feature_count}` : '')
              }
            >
              <span className={`status-dot ${statusDotClass}`} />
              <span>{statusText}</span>
            </div>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={onRefreshHealth}
              disabled={healthLoading}
              aria-label="Refresh backend connection status"
            >
              <RefreshCw size={14} />
            </button>
          </div>
        </header>

        <main className="page-content">{children}</main>
      </div>
    </div>
  );
}
