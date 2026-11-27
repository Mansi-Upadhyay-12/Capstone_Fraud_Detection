import React from 'react';

export function formatNumber(value: number | undefined | null): string {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return value.toLocaleString('en-US');
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatTimestamp(iso: string): string {
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return iso;
  return dt.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function RiskBadge({
  prob,
  riskCategory,
}: {
  prob: number;
  riskCategory?: string;
}): React.ReactElement {
  let cls = 'badge-low';
  if (prob >= 0.85) {
    cls = 'badge-critical';
  } else if (prob >= 0.57) {
    cls = 'badge-elevated';
  } else if (prob >= 0.35) {
    cls = 'badge-moderate';
  }
  return <span className={`badge ${cls}`}>{riskCategory || `${(prob * 100).toFixed(1)}%`}</span>;
}

export function ProbBar({ prob }: { prob: number }): React.ReactElement {
  const pct = Math.max(0, Math.min(100, prob * 100));
  let color = 'var(--risk-low)';
  if (prob >= 0.85) color = 'var(--risk-critical)';
  else if (prob >= 0.57) color = 'var(--risk-elevated)';
  else if (prob >= 0.35) color = 'var(--risk-moderate)';

  return (
    <div className="prob-bar-wrap">
      <div className="prob-bar-track" aria-hidden="true">
        <div className="prob-bar-fill" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <span className="mono" style={{ fontSize: '0.82rem' }}>
        {prob.toFixed(4)}
      </span>
    </div>
  );
}

export function GtBadge({ label }: { label: string }): React.ReactElement {
  const normalized = label.toLowerCase();
  if (normalized === 'illicit') {
    return <span className="badge badge-critical">Illicit</span>;
  }
  if (normalized === 'licit') {
    return <span className="badge badge-low">Licit</span>;
  }
  return <span className="badge badge-neutral">Unknown</span>;
}
