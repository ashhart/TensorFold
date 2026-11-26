export type RdmaPort = {
  device: string
  iface: string
  state: string
  mtu: number
  speed_gbps: number
  ip: string | null
  subnet: string | null
  peer_ip: string | null
  rx_bytes: number
  tx_bytes: number
  rx_mbps: number
  tx_mbps: number
}

export type ClusterModelPresence = {
  node_id: string
  name: string
  present: boolean
  complete: boolean
  shards_present: number
  shards_expected: number
  bytes_on_disk: number
}

export type ClusterModel = {
  id: string
  name: string
  path: string
  model_type: string
  format: string
  num_hidden_layers: number | null
  num_attention_heads: number | null
  n_routed_experts: number | null
  hidden_size: number | null
  max_position_embeddings: number | null
  shards_present: number
  shards_expected: number
  bytes_on_disk: number
  complete: boolean
  download: {
    done_files: number
    total_files: number
    done_bytes: number
    total_bytes: number
    pct: number
    rate_mib_s: number
    complete: boolean
  } | null
  cluster_presence?: ClusterModelPresence[]
}

export type ClusterRankState = {
  state: 'idle' | 'loading' | 'running' | 'stopped' | 'failed' | 'offline'
  rank?: number | null
  world?: number | null
  tp?: number | null
  pp?: number | null
  tp_rank?: number | null
  pp_rank?: number | null
  model_path?: string | null
  port?: number | null
  pid?: number | null
  rss_bytes?: number
  layer_loaded?: number
  layer_total?: number
  phase?: string
  error?: string | null
  log_tail?: string[]
  stats?: Record<string, any>
}

export type ClusterNode = {
  id: string
  name: string
  hostname: string
  role: 'master' | 'slave'
  online: boolean
  ping_ms: number
  error: string | null
  enabled: boolean
  weight: number
  color: string
  ssh_host?: string | null
  slave_url?: string | null
  uptime_s?: number
  hardware: {
    chip: string
    cpu_cores?: number
    gpu_cores: number
    ram_total_bytes: number
    memory_bandwidth_gbs: number
    peak_tflops_fp16?: number
    os_version?: string
  }
  memory: {
    total_bytes: number
    used_bytes: number
    wired_bytes?: number
    free_bytes?: number
    available_bytes: number
    pressure_pct: number
  }
  gpu: {
    util_pct: number
    renderer_pct?: number
    tiler_pct?: number
    alloc_bytes: number
    in_use_bytes: number
  }
  cpu: {
    load_1m: number
    util_pct: number
    cores?: number
  }
  rdma: RdmaPort[]
  models: ClusterModel[]
  binaries?: Record<string, string | null>
  rank: ClusterRankState
}

export type ClusterLink = {
  id: string
  subnet: string
  kind: 'tb5_rdma'
  speed_gbps: number
  mtu: number
  state: 'active' | 'degraded'
  a_id: string
  a_name: string
  a_device: string
  a_iface: string
  a_ip: string
  b_id: string
  b_name: string
  b_device: string
  b_iface: string
  b_ip: string
  traffic_mbps: number
  a_tx_mbps: number
  b_tx_mbps: number
  latency_us: number | null
  bandwidth_gbs: number | null
  probed_at: number | null
}

export type ClusterRankPlan = {
  rank: number
  node_id: string
  name: string
  role: string
  color: string
  chip: string
  gpu_cores: number
  bandwidth_gbs: number
  tp_rank: number
  pp_rank: number
  layer_range: [number, number]
  layer_count: number
  head_range: [number, number]
  head_count: number
  moe_half_range: [number, number]
  moe_channels: number
  dense_range: [number, number]
  vocab_range: [number, number]
  share_pct: number
  weights_gib: number
  kv_gib: number
  runtime_gib: number
  total_needed_gib: number
  ram_total_gib: number
  load_limit_gib: number
  headroom_gib: number
  fits: boolean
  peers: { rank: number; device: string; via: string; port: number; peer_port: number; name: string; fallback?: boolean }[]
  missing_tb5_links: string[]
  rank_json: Record<string, any>
}

export type ClusterLayerAlloc = {
  layer: number
  kind: 'dense' | 'moe' | 'mtp'
  indexer: 'full' | 'shared'
  pp_stage: number
  shards: {
    rank: number
    node_id: string
    name: string
    color: string
    pct: number
    heads: string
    moe_ch: string
  }[]
}

export type ClusterPlan = {
  model: {
    name: string
    path: string
    model_type: string
    num_layers: number
    num_dense_layers: number
    num_heads: number
    hidden_size: number
    moe_intermediate: number
    dense_intermediate: number
    vocab_size: number
    n_experts: number
    has_mtp: boolean
    total_weight_bytes: number
  }
  mode: string
  resolved_mode: 'single' | 'tp' | 'pp' | 'hybrid'
  world: number
  tp: number
  pp: number
  heterogeneous: boolean
  context: number
  parallel: number
  all_fit: boolean
  effective_bandwidth_gbs: number
  total_cluster_ram_gib: number
  total_weights_gib: number
  all_reduce_per_token: number
  pp_handoffs_per_token: number
  ranks: ClusterRankPlan[]
  layers: ClusterLayerAlloc[]
}

export type ClusterBenchmarkPoint = {
  target_prompt_tokens: number
  prompt_tokens: number
  completion_tokens: number
  ttft_ms: number
  prefill_tps: number
  decode_tps: number
}

export type ClusterBenchmark = {
  id: string
  label: string
  state: 'running' | 'done' | 'failed'
  started_at: number
  finished_at?: number
  model: string
  world: number
  tp: number
  pp: number
  mode: string
  draft: boolean
  steps_total?: number
  steps_done?: number
  points: ClusterBenchmarkPoint[]
  peak_prefill_tps?: number
  avg_decode_tps?: number
  error?: string | null
}

export type ClusterDiscoverCandidate = {
  host: string
  ssh_host: string
  ssh_user?: string | null
  ssh_key?: string | null
  name: string
  hostname?: string
  source: string
  tb5_iface?: string
  tb5_device?: string
  already_added: boolean
  slave_running: boolean
  ssh_reachable: boolean
  chip: string | null
  ram_gb: number | null
}

export type ClusterSyncJob = {
  node_id: string
  name: string
  model: string
  state: 'running' | 'done' | 'failed'
  pct: number
  rate: string
  started_at: number
  finished_at?: number
  error?: string | null
}

export type ClusterState = {
  timestamp: number
  master_id: string
  master_lan_ip: string
  cluster_token: string
  slave_join_command: string
  summary: {
    nodes_total: number
    nodes_online: number
    links_total: number
    links_active: number
    total_ram_bytes: number
    used_ram_bytes: number
    available_ram_bytes: number
    total_gpu_cores: number
    total_gpu_alloc_bytes: number
    total_bandwidth_gbs: number
    total_tflops_fp16: number
    avg_gpu_util_pct: number
    total_rdma_mbps: number
    decode_tps: number
    prefill_tps: number
    mtp_accept_rate: number | null
    running_requests: number
    waiting_requests: number
    prompt_tokens_total: number
    generation_tokens_total: number
    ttft_p50_ms: number | null
  }
  nodes: ClusterNode[]
  links: ClusterLink[]
  models: ClusterModel[]
  deployment: {
    id: string
    started_at: number
    serve_port: number
    model_name: string
    model_path: string
    mode: string
    world: number
    tp: number
    pp: number
    context: number
    parallel: number
    node_ids: string[]
    plan: ClusterPlan
  } | null
  sync_jobs: Record<string, ClusterSyncJob>
  benchmark_job: ClusterBenchmark | null
  benchmarks: ClusterBenchmark[]
  probe_running: boolean
  history: { t: number; decode_tps: number; prefill_tps: number; gpu_util: number; rdma_mbps: number }[]
}
