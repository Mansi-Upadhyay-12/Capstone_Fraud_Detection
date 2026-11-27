export interface HealthResponse {
  status: string;
  model_id: string;
  checkpoint_path: string;
  threshold_used: number;
  preprocessing_order: string[];
  device: string;
  stage2_artifact_ready: boolean;
  rankgauss_ready: boolean;
  feature_count: number;
  streaming_supported: boolean;
}

export interface ValidationResponse {
  valid: boolean;
  node_count: number;
  edge_count: number;
  unique_edges_after_dedup: number;
  duplicate_edges_removed: number;
  timesteps: number[];
  imputed_nan_cells: number;
  has_optional_class_column: boolean;
  class_counts: {
    illicit: number;
    licit: number;
    unknown: number;
  };
  warnings: string[];
}

export interface AnalyzeResponse {
  run_id: string;
  dataset_name: string;
  status: string;
  created_at: string;
  inference_time_sec: number;
  threshold_used: number;
  node_count: number;
  edge_count: number;
  timesteps: number[];
  flagged_count: number;
  flagged_rate_pct: number;
  direct_clusters_count: number;
  potential_candidates_count: number;
  shared_bridges_count: number;
  warnings: string[];
  disclaimer: string;
}

export interface RunSummaryItem {
  run_id: string;
  dataset_name: string;
  created_at: string;
  node_count: number;
  edge_count: number;
  flagged_count: number;
  direct_clusters_count: number;
  potential_candidates_count: number;
  threshold_used: number;
  inference_time_sec: number;
  has_optional_class_column: boolean;
}

export interface PerTimestepSummary {
  timestep: number;
  node_count: number;
  edge_count: number;
  flagged_count: number;
  flagged_rate_pct: number;
  mean_pred_prob: number;
  max_pred_prob: number;
  direct_clusters_count: number;
  potential_candidates_count: number;
  shared_bridges_count: number;
}

export interface BridgeSample {
  timestep: number;
  bridge_tx_id: number;
  bridge_pred_prob: number;
  bridge_gt_label: string;
  bridge_motif: string;
  sus_in_count: number;
  sus_out_count: number;
  distinct_sus_touched: number;
}

export interface PostHocEval {
  labeled_count: number;
  unknown_count: number;
  tp: number;
  fp: number;
  fn: number;
  tn: number;
  precision: number;
  recall: number;
  f1: number;
  note: string;
}

export interface RunDetail {
  run_id: string;
  dataset_name: string;
  created_at: string;
  created_epoch: number;
  threshold_used: number;
  inference_time_sec: number;
  node_count: number;
  edge_count: number;
  flagged_count: number;
  direct_clusters_count: number;
  potential_candidates_count: number;
  shared_bridges_count: number;
  has_optional_class_column: boolean;
  cache_available: number;
  summary: {
    model_id: string;
    preprocessing_order: string[];
    timesteps: number[];
    unique_edges_after_dedup: number;
    duplicate_edges_removed: number;
    imputed_nan_cells: number;
    flagged_rate_pct: number;
    mean_pred_prob: number;
    max_pred_prob: number;
    risk_band_counts: {
      HighPredictedRisk: number;
      ElevatedRisk: number;
      ModerateMonitor: number;
      LowPredictedRisk: number;
    };
    class_counts: {
      illicit: number;
      licit: number;
      unknown: number;
    };
    post_hoc_eval: PostHocEval | null;
    per_timestep: PerTimestepSummary[];
    bridge_samples: BridgeSample[];
    warnings: string[];
    disclaimer: string;
  };
}

export interface TransactionItem {
  run_id?: string;
  node_order?: number;
  tx_id: number;
  timestep: number;
  snapshot_node_idx: number;
  pred_prob: number;
  logit?: number;
  pred_class: number;
  pred_label: string;
  risk_category: string;
  in_deg: number;
  out_deg: number;
  cluster_id: string;
  optional_gt_code: number;
  optional_gt_label: string;
  is_external_context?: boolean;
}

export interface TransactionsPageResponse {
  total: number;
  limit: number;
  offset: number;
  items: TransactionItem[];
}

export interface ClusterItem {
  cluster_id: string;
  timestep: number;
  structural_tier: string;
  is_ring_candidate: number;
  motif_type: string;
  ring_score: number;
  num_nodes: number;
  num_internal_edges: number;
  directed_edge_density: number;
  has_directed_cycle: number;
  max_directed_path_len: number;
  max_internal_in_deg: number;
  max_internal_out_deg: number;
  ext_1hop_in_neighbors: number;
  ext_1hop_out_neighbors: number;
  mean_pred_prob: number;
  max_pred_prob: number;
  min_pred_prob: number;
  num_high_conf_ge_085: number;
  hub_tx_id: number;
  hub_node_idx: number;
  hub_pred_prob: number;
  post_hoc_gt_illicit: number;
  post_hoc_gt_licit: number;
  post_hoc_gt_unknown: number;
  post_hoc_gt_labeled: number;
  has_labeled_gt: number;
  post_hoc_labeled_illicit_precision: number;
  post_hoc_eval_status: string;
}

export interface ClustersResponse {
  run_id: string;
  total: number;
  clusters: ClusterItem[];
}

export interface ClusterGraphEdge {
  cluster_id: string;
  timestep: number;
  src_tx_id: number;
  dst_tx_id: number;
  edge_type: 'internal_suspicious' | 'external_boundary_in' | 'external_boundary_out' | string;
  src_pred_prob: number;
  dst_pred_prob: number;
}

export interface ClusterGraphResponse {
  run_id: string;
  cluster: ClusterItem;
  nodes: TransactionItem[];
  edges: ClusterGraphEdge[];
}

export interface FeatureAttributionItem {
  feature_index: number;
  feature_name: string;
  feature_group: 'Local93' | 'Aggregate72' | 'Domain17' | string;
  target_rankgauss_value: number;
  graph_ig_attribution: number;
  self_ig_attribution: number;
  neighbor_ig_attribution: number;
}

export interface FeatureGroupBreakdownItem {
  group: 'Local93' | 'Aggregate72' | 'Domain17' | string;
  feature_count: number;
  signed_sum: number;
  abs_sum: number;
  self_signed_sum: number;
}

export interface OneHopNeighborItem {
  direction: 'incoming' | 'outgoing' | string;
  edge_representation: string;
  neighbor_tx_id: number;
  neighbor_pred_prob: number;
  neighbor_pred_class: string;
  neighbor_gt_label: string;
  neighbor_in_deg: number;
  neighbor_out_deg: number;
  neighbor_signed_ig_logit: number;
  neighbor_abs_ig_logit: number;
}

export interface TransactionExplanationResponse {
  run_id: string;
  tx_id: number;
  timestep: number;
  created_at: string;
  cache_hit: boolean;
  method: string;
  baseline: string;
  model_id: string;
  threshold_used: number;
  prediction: {
    pred_prob: number;
    logit: number;
    pred_label: string;
    risk_category: string;
    cluster_id: string;
    optional_gt_label: string;
  };
  completeness: {
    logit_x: number;
    logit_baseline: number;
    logit_delta: number;
    ig_sum: number;
    abs_error: number;
    rel_error_pct: number;
    all_finite: boolean;
  };
  self_vs_neighbor: {
    self_signed_logit: number;
    neighbor_signed_logit: number;
    self_abs_share: number;
    neighbor_abs_share: number;
    one_hop_unique_count: number;
    incoming_count: number;
    outgoing_count: number;
  };
  feature_group_breakdown: FeatureGroupBreakdownItem[];
  top_positive_features: FeatureAttributionItem[];
  top_negative_features: FeatureAttributionItem[];
  one_hop_neighbors: OneHopNeighborItem[];
  two_hop_summary: {
    count: number;
    pred_illicit_count: number;
    pred_licit_count: number;
    pred_illicit_rate: number;
    mean_pred_prob: number;
    gt_illicit_count: number;
    gt_licit_count: number;
    gt_unknown_count: number;
  };
  disclaimer: string;
}

export interface ApiErrorDetail {
  error?: string;
  message: string;
  missing_count?: number;
  missing_sample?: string[];
  columns?: string[];
  duplicate_sample?: number[];
  dangling_edge_count?: number;
  cross_timestep_edge_count?: number;
  sample?: unknown[];
}
