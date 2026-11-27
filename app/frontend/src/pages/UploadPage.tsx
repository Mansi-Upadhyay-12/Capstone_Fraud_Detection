import React, { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AlertCircle,
  AlertTriangle,
  CheckCircle2,
  FileCheck2,
  FileSpreadsheet,
  Info,
  Loader2,
  Play,
  Trash2,
  UploadCloud,
} from 'lucide-react';
import { analyzeCsvBundle, BackendRequestError, validateCsvBundle } from '../api/fraudApi';
import { formatBytes, formatNumber } from '../components/RiskBadge';
import type { ValidationResponse } from '../types/api';

interface UploadPageProps {
  onAnalysisCompleted: () => void;
}

const MAX_BYTES = 100 * 1024 * 1024;

function isCsvLikeFile(file: File): boolean {
  const lower = file.name.toLowerCase();
  return lower.endsWith('.csv') || lower.endsWith('.txt');
}

interface DropZoneProps {
  title: string;
  subtitle: string;
  required?: boolean;
  file: File | null;
  disabled: boolean;
  onSelect: (file: File | null) => void;
  onClientError: (msg: string) => void;
  inputTestId: string;
}

function FileDropZone({
  title,
  subtitle,
  required = false,
  file,
  disabled,
  onSelect,
  onClientError,
  inputTestId,
}: DropZoneProps): React.ReactElement {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const acceptFile = (candidate: File | undefined) => {
    if (!candidate) return;
    if (!isCsvLikeFile(candidate)) {
      onClientError(`Invalid file type for ${title}: "${candidate.name}". Please upload a .csv or .txt file.`);
      return;
    }
    if (candidate.size === 0) {
      onClientError(`Selected file "${candidate.name}" is empty (0 bytes).`);
      return;
    }
    if (candidate.size > MAX_BYTES) {
      onClientError(`Selected file "${candidate.name}" exceeds the 100 MB limit.`);
      return;
    }
    onSelect(candidate);
  };

  return (
    <div
      className={`dropzone ${dragging ? 'dragging' : ''} ${file ? 'ready' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled) acceptFile(e.dataTransfer.files[0]);
      }}
      onClick={() => {
        if (!disabled) inputRef.current?.click();
      }}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if ((e.key === 'Enter' || e.key === ' ') && !disabled) {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
    >
      <input
        ref={inputRef}
        data-testid={inputTestId}
        type="file"
        accept=".csv,.txt,text/csv"
        style={{ display: 'none' }}
        disabled={disabled}
        onChange={(e) => {
          acceptFile(e.target.files?.[0]);
          e.target.value = '';
        }}
      />

      {file ? (
        <>
          <FileCheck2 size={28} color="var(--accent-teal)" />
          <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{title}</div>
          <div className="mono" style={{ fontSize: '0.84rem', color: 'var(--accent-teal)' }}>
            {file.name}
          </div>
          <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
            Size: {formatBytes(file.size)}
          </div>
          <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={disabled}
              onClick={(e) => {
                e.stopPropagation();
                inputRef.current?.click();
              }}
            >
              Replace
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              disabled={disabled}
              onClick={(e) => {
                e.stopPropagation();
                onSelect(null);
              }}
            >
              <Trash2 size={14} /> Remove
            </button>
          </div>
        </>
      ) : (
        <>
          <UploadCloud size={28} color="var(--accent-blue)" />
          <div style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
            {title}{' '}
            {required ? (
              <span style={{ color: 'var(--accent-teal)', fontSize: '0.76rem' }}>(Required)</span>
            ) : (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.76rem' }}>(Optional)</span>
            )}
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', maxWidth: 320 }}>{subtitle}</div>
          <span className="btn btn-secondary" style={{ marginTop: 4 }}>
            Browse File
          </span>
        </>
      )}
    </div>
  );
}

export function UploadPage({ onAnalysisCompleted }: UploadPageProps): React.ReactElement {
  const navigate = useNavigate();

  const [txFile, setTxFile] = useState<File | null>(null);
  const [edgeFile, setEdgeFile] = useState<File | null>(null);
  const [classFile, setClassFile] = useState<File | null>(null);
  const [datasetName, setDatasetName] = useState('');

  const [validation, setValidation] = useState<ValidationResponse | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [phase, setPhase] = useState<'idle' | 'validating' | 'analyzing'>('idle');
  const [errorState, setErrorState] = useState<BackendRequestError | null>(null);
  const [clientNotice, setClientNotice] = useState<string | null>(null);
  const submitLockRef = useRef(false);

  const resetValidation = () => {
    setValidation(null);
    setErrorState(null);
    setClientNotice(null);
    setUploadProgress(0);
  };

  const handleValidate = async () => {
    if (!txFile || !edgeFile || phase !== 'idle' || submitLockRef.current) return;
    submitLockRef.current = true;
    setErrorState(null);
    setClientNotice(null);
    setValidation(null);
    setUploadProgress(0);
    setPhase('validating');

    try {
      const res = await validateCsvBundle(txFile, edgeFile, classFile, setUploadProgress);
      setValidation(res);
    } catch (err) {
      setErrorState(
        err instanceof BackendRequestError
          ? err
          : new BackendRequestError('Validation failed unexpectedly.'),
      );
    } finally {
      submitLockRef.current = false;
      setPhase('idle');
    }
  };

  const handleAnalyze = async () => {
    if (!txFile || !edgeFile || !validation?.valid || phase !== 'idle' || submitLockRef.current) return;
    submitLockRef.current = true;
    setErrorState(null);
    setClientNotice(null);
    setUploadProgress(0);
    setPhase('analyzing');

    try {
      const safeLabel = (datasetName || txFile.name).trim().slice(0, 120);
      const res = await analyzeCsvBundle(
        txFile,
        edgeFile,
        classFile,
        safeLabel,
        setUploadProgress,
      );
      onAnalysisCompleted();
      navigate(`/runs/${encodeURIComponent(res.run_id)}`);
    } catch (err) {
      setErrorState(
        err instanceof BackendRequestError
          ? err
          : new BackendRequestError('Analysis submission failed unexpectedly.'),
      );
      setPhase('idle');
    } finally {
      submitLockRef.current = false;
    }
  };

  const busy = phase !== 'idle';

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Upload &amp; Analyse Transaction CSVs</h2>
          <p className="page-desc">
            Two-stage workflow: schema &amp; edge integrity verification followed by frozen BiDirGCN inference.
          </p>
        </div>
      </div>

      <div className="banner banner-info">
        <Info size={18} style={{ flexShrink: 0, marginTop: 2 }} />
        <div>
          <strong>Optional Class Labels Isolation:</strong> If a <span className="mono">class</span> column is
          included inside <span className="mono">transactions.csv</span> or via an optional <span className="mono">classes.csv</span>,
          labels (<span className="mono">1=Illicit</span>, <span className="mono">2=Licit</span>,{' '}
          <span className="mono">3/Unknown</span>) are strictly ignored during Stage 2 preprocessing, RankGauss, BiDirGCN inference,
          and cluster scoring, and retained solely for post-hoc inspection.
        </div>
      </div>

      <div className="card">
        <div className="card-header">
          <div>
            <h3 className="card-title">1. Required CSV Inputs</h3>
            <p className="card-subtitle">Drag and drop files below or browse from disk (max 100 MB per file)</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <label htmlFor="dataset-name-input" style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
              Run Label:
            </label>
            <input
              id="dataset-name-input"
              className="input"
              type="text"
              placeholder="e.g. elliptic_step42_batch"
              value={datasetName}
              disabled={busy}
              onChange={(e) => setDatasetName(e.target.value)}
            />
          </div>
        </div>

        <div className="grid-2">
          <FileDropZone
            title="transactions.csv"
            subtitle="Must include columns: txId, Time step, and all exact 182 feature columns."
            required
            file={txFile}
            disabled={busy}
            inputTestId="transactions-file-input"
            onClientError={(m) => setClientNotice(m)}
            onSelect={(f) => {
              setTxFile(f);
              resetValidation();
            }}
          />

          <FileDropZone
            title="edges.csv"
            subtitle="Directed intra-timestep pairs with columns txId1 and txId2. No dangling or cross-step edges."
            required
            file={edgeFile}
            disabled={busy}
            inputTestId="edges-file-input"
            onClientError={(m) => setClientNotice(m)}
            onSelect={(f) => {
              setEdgeFile(f);
              resetValidation();
            }}
          />
        </div>

        <details style={{ marginTop: 16 }}>
          <summary
            style={{
              cursor: 'pointer',
              fontSize: '0.86rem',
              fontWeight: 600,
              color: 'var(--text-secondary)',
            }}
          >
            Optional File 3: Separate Ground-Truth classes.csv (Post-Hoc Display Only)
          </summary>
          <div style={{ marginTop: 12 }}>
            <FileDropZone
              title="classes.csv"
              subtitle="Optional lookup CSV with columns txId and class (1=Illicit, 2=Licit, 3/unknown)."
              file={classFile}
              disabled={busy}
              inputTestId="classes-file-input"
              onClientError={(m) => setClientNotice(m)}
              onSelect={(f) => {
                setClassFile(f);
                resetValidation();
              }}
            />
          </div>
        </details>

        <div
          style={{
            marginTop: 18,
            padding: 14,
            borderRadius: 'var(--radius-md)',
            background: 'var(--bg-canvas)',
            border: '1px solid var(--border-subtle)',
            fontSize: '0.82rem',
            color: 'var(--text-secondary)',
            display: 'flex',
            alignItems: 'flex-start',
            gap: 10,
          }}
        >
          <FileSpreadsheet size={18} color="var(--accent-blue)" style={{ flexShrink: 0, marginTop: 2 }} />
          <div>
            <strong>Required Feature Schema:</strong> <span className="mono">txId</span> (unique int),{' '}
            <span className="mono">Time step</span> (int &ge; 1), <span className="mono">Local_feature_1..93</span>,{' '}
            <span className="mono">Aggregate_feature_1..72</span>, and 17 Bitcoin flow domain columns (182 numeric features total;
            missing numeric cells allowed for Stage 2 median imputation). Dangling edges and cross-timestep edges trigger{' '}
            <span className="mono">HTTP 422</span> and are never silently dropped.
          </div>
        </div>

        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 12, marginTop: 18 }}>
          <button
            type="button"
            className="btn btn-secondary"
            disabled={!txFile || !edgeFile || busy}
            onClick={handleValidate}
          >
            {phase === 'validating' ? <Loader2 size={16} className="spin" /> : <CheckCircle2 size={16} />}
            <span>{phase === 'validating' ? 'Validating Bundle...' : '1. Validate CSV Bundle'}</span>
          </button>

          <button
            type="button"
            className="btn btn-primary"
            disabled={!validation?.valid || busy}
            onClick={handleAnalyze}
          >
            {phase === 'analyzing' ? <Loader2 size={16} className="spin" /> : <Play size={16} />}
            <span>{phase === 'analyzing' ? 'Running Frozen BiDirGCN Inference...' : '2. Run Fraud Analysis'}</span>
          </button>

          {!validation?.valid && txFile && edgeFile && !busy && (
            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
              Validate the CSV bundle first to enable frozen model inference.
            </span>
          )}
        </div>

        {busy && (
          <div style={{ marginTop: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', marginBottom: 6 }}>
              <span>
                {phase === 'validating'
                  ? 'Uploading & checking schema + directed edge integrity...'
                  : 'Running Stage 2 -> RankGauss -> BiDirGCN inference & clustering (preventing duplicate submissions)...'}
              </span>
              <span className="mono">{uploadProgress}%</span>
            </div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${Math.max(uploadProgress, 8)}%` }} />
            </div>
          </div>
        )}
      </div>

      {clientNotice && (
        <div className="banner banner-warning">
          <AlertTriangle size={18} style={{ flexShrink: 0, marginTop: 2 }} />
          <div>{clientNotice}</div>
        </div>
      )}

      {errorState && (
        <div className="card" style={{ borderColor: 'rgba(239, 68, 68, 0.4)' }}>
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
            <AlertCircle size={20} color="var(--risk-critical)" style={{ flexShrink: 0, marginTop: 2 }} />
            <div style={{ flex: 1 }}>
              <div style={{ fontWeight: 700, color: '#fca5a5' }}>
                Backend Rejected Upload ({errorState.code || `HTTP ${errorState.status}`})
              </div>
              <p style={{ margin: '6px 0 8px', fontSize: '0.9rem', color: 'var(--text-primary)' }}>
                {errorState.message}
              </p>
              {errorState.detail?.missing_sample && (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Missing columns sample:{' '}
                  <span className="mono">{errorState.detail.missing_sample.join(', ')}</span>
                </div>
              )}
              {errorState.detail?.columns && (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  Offending feature columns:{' '}
                  <span className="mono">{errorState.detail.columns.join(', ')}</span>
                </div>
              )}
              {errorState.detail?.sample && (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                  Offending sample:{' '}
                  <span className="mono">{JSON.stringify(errorState.detail.sample.slice(0, 3))}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {validation && (
        <div className="card" style={{ borderColor: 'rgba(16, 185, 129, 0.4)' }}>
          <div className="card-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <CheckCircle2 size={20} color="var(--risk-low)" />
              <div>
                <h3 className="card-title">Validation Passed — Ready for Fraud Analysis</h3>
                <p className="card-subtitle">Exact 182 feature columns and intra-timestep edge constraints verified</p>
              </div>
            </div>
            <span className="badge badge-low">VALID</span>
          </div>

          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))',
              gap: 12,
              marginBottom: 14,
            }}
          >
            <div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>Validated Transactions</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700 }}>{formatNumber(validation.node_count)}</div>
            </div>
            <div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>Unique Directed Edges</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700 }}>
                {formatNumber(validation.unique_edges_after_dedup)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>Timesteps Covered</div>
              <div className="mono" style={{ fontSize: '0.95rem', fontWeight: 600 }}>
                {validation.timesteps.join(', ')}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>Median-Imputed NaNs</div>
              <div style={{ fontSize: '1.15rem', fontWeight: 700 }}>
                {formatNumber(validation.imputed_nan_cells)}
              </div>
            </div>
            <div>
              <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)' }}>Optional Labels</div>
              <div style={{ fontSize: '0.9rem', fontWeight: 600 }}>
                {validation.has_optional_class_column
                  ? `${validation.class_counts.illicit} Illicit / ${validation.class_counts.licit} Licit / ${validation.class_counts.unknown} Unk`
                  : 'None uploaded'}
              </div>
            </div>
          </div>

          {validation.warnings.length > 0 && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
              {validation.warnings.map((w, i) => (
                <div key={i} className="banner banner-warning">
                  <AlertTriangle size={16} style={{ flexShrink: 0, marginTop: 2 }} />
                  <span>{w}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </>
  );
}
