import axios, { AxiosError } from 'axios';
import type {
  AnalyzeResponse,
  ApiErrorDetail,
  ClusterGraphResponse,
  ClustersResponse,
  HealthResponse,
  RunDetail,
  RunSummaryItem,
  TransactionExplanationResponse,
  TransactionsPageResponse,
  ValidationResponse,
} from '../types/api';

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') || 'http://127.0.0.1:8010';

const http = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180_000,
});

export class BackendRequestError extends Error {
  status: number;
  code: string;
  detail: ApiErrorDetail | null;

  constructor(message: string, status = 0, code = 'request_failed', detail: ApiErrorDetail | null = null) {
    super(message);
    this.name = 'BackendRequestError';
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

export function normalizeApiError(err: unknown): BackendRequestError {
  if (err instanceof BackendRequestError) {
    return err;
  }
  if (axios.isAxiosError(err)) {
    const axErr = err as AxiosError<{ detail?: ApiErrorDetail | string }>;
    if (!axErr.response) {
      return new BackendRequestError(
        `Unable to reach FraudLens backend at ${API_BASE_URL}. Ensure the FastAPI server is running.`,
        0,
        'network_error',
        null,
      );
    }
    const status = axErr.response.status;
    const rawDetail = axErr.response.data?.detail;
    if (rawDetail && typeof rawDetail === 'object') {
      return new BackendRequestError(
        rawDetail.message || `Request failed with status ${status}.`,
        status,
        rawDetail.error || 'api_error',
        rawDetail,
      );
    }
    if (typeof rawDetail === 'string') {
      return new BackendRequestError(rawDetail, status, 'api_error', { message: rawDetail });
    }
    return new BackendRequestError(
      axErr.message || `Backend responded with HTTP ${status}.`,
      status,
      'http_error',
      null,
    );
  }
  if (err instanceof Error) {
    return new BackendRequestError(err.message, 0, 'unexpected_error', null);
  }
  return new BackendRequestError('An unexpected error occurred.', 0, 'unexpected_error', null);
}

export async function fetchHealth(): Promise<HealthResponse> {
  try {
    const { data } = await http.get<HealthResponse>('/api/health', { timeout: 10_000 });
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export async function validateCsvBundle(
  transactionsFile: File,
  edgesFile: File,
  classesFile: File | null,
  onProgress?: (pct: number) => void,
): Promise<ValidationResponse> {
  const form = new FormData();
  form.append('transactions_file', transactionsFile);
  form.append('edges_file', edgesFile);
  if (classesFile) {
    form.append('classes_file', classesFile);
  }

  try {
    const { data } = await http.post<ValidationResponse>('/api/validate-csv', form, {
      onUploadProgress: (evt) => {
        if (onProgress && evt.total) {
          onProgress(Math.round((evt.loaded * 100) / evt.total));
        }
      },
    });
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export async function analyzeCsvBundle(
  transactionsFile: File,
  edgesFile: File,
  classesFile: File | null,
  datasetName: string,
  onProgress?: (pct: number) => void,
): Promise<AnalyzeResponse> {
  const form = new FormData();
  form.append('transactions_file', transactionsFile);
  form.append('edges_file', edgesFile);
  if (classesFile) {
    form.append('classes_file', classesFile);
  }
  if (datasetName.trim()) {
    form.append('dataset_name', datasetName.trim());
  }

  try {
    const { data } = await http.post<AnalyzeResponse>('/api/analyze', form, {
      onUploadProgress: (evt) => {
        if (onProgress && evt.total) {
          onProgress(Math.round((evt.loaded * 100) / evt.total));
        }
      },
    });
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export async function fetchRuns(): Promise<RunSummaryItem[]> {
  try {
    const { data } = await http.get<RunSummaryItem[]>('/api/runs');
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export async function fetchRunDetail(runId: string): Promise<RunDetail> {
  try {
    const { data } = await http.get<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`);
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export interface TransactionQueryParams {
  timestep?: number | null;
  flaggedOnly?: boolean;
  minProb?: number;
  searchTxId?: string;
  limit?: number;
  offset?: number;
}

export async function fetchRunTransactions(
  runId: string,
  params: TransactionQueryParams = {},
): Promise<TransactionsPageResponse> {
  const query: Record<string, string | number | boolean> = {
    limit: params.limit ?? 50,
    offset: params.offset ?? 0,
  };
  if (params.timestep !== undefined && params.timestep !== null) {
    query.timestep = params.timestep;
  }
  if (params.flaggedOnly) {
    query.flagged_only = true;
  }
  if (params.minProb !== undefined && params.minProb > 0) {
    query.min_prob = params.minProb;
  }
  if (params.searchTxId && /^\d+$/.test(params.searchTxId.trim())) {
    query.search_tx_id = Number(params.searchTxId.trim());
  }

  try {
    const { data } = await http.get<TransactionsPageResponse>(
      `/api/runs/${encodeURIComponent(runId)}/transactions`,
      { params: query },
    );
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export async function fetchRunClusters(
  runId: string,
  candidatesOnly = false,
  timestep: number | null = null,
): Promise<ClustersResponse> {
  const query: Record<string, number | boolean> = {};
  if (candidatesOnly) {
    query.candidates_only = true;
  }
  if (timestep !== null) {
    query.timestep = timestep;
  }
  try {
    const { data } = await http.get<ClustersResponse>(
      `/api/runs/${encodeURIComponent(runId)}/clusters`,
      { params: query },
    );
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export async function fetchClusterGraph(
  runId: string,
  clusterId: string,
): Promise<ClusterGraphResponse> {
  try {
    const { data } = await http.get<ClusterGraphResponse>(
      `/api/runs/${encodeURIComponent(runId)}/clusters/${encodeURIComponent(clusterId)}/graph`,
    );
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}

export async function explainTransaction(
  runId: string,
  txId: number | string,
): Promise<TransactionExplanationResponse> {
  try {
    const { data } = await http.post<TransactionExplanationResponse>(
      `/api/runs/${encodeURIComponent(runId)}/transactions/${encodeURIComponent(String(txId))}/explain`,
      null,
      { timeout: 180_000 },
    );
    return data;
  } catch (err) {
    throw normalizeApiError(err);
  }
}
