/** Renderer-safe local benchmark contracts. No process paths, credentials or generated text. */
export type ModelBenchmarkState = 'preparing' | 'running' | 'restoring' | 'complete' | 'cancelled' | 'failed'
export type ModelBenchmarkFixture = 'code' | 'chat'
export type ModelBenchmarkSampleStatus = 'ok' | 'early_eos' | 'error' | 'unmeasured'
export interface ModelBenchmarkInput { id: string; model: string; serial?: boolean }
export interface ModelBenchmarkSpread { median: number; min: number; max: number; spread: number }
export interface ModelBenchmarkCell {
  fixture: ModelBenchmarkFixture
  temperature: 0 | 1
  attempted: number
  expected: 5
  successful: number
  earlyEos: number
  errors: number
  unmeasured: number
  failedWarmups: number
  complete: boolean
  deliveryTokensPerSecond: ModelBenchmarkSpread | null
  firstTextSeconds: ModelBenchmarkSpread | null
  endToEndSeconds: ModelBenchmarkSpread | null
  serverDecodeTokensPerSecond: ModelBenchmarkSpread | null
  prefillSeconds: ModelBenchmarkSpread | null
}
export interface ModelBenchmarkProgress {
  completedRequests: number | null
  totalRequests: 24
  fixture: ModelBenchmarkFixture | null
  temperature: 0 | 1 | null
  repeat: number | null
  phase: 'loading' | 'warmup' | 'measured' | null
}
export interface ModelBenchmarkSnapshot {
  id: string
  model: string
  serial: boolean
  state: ModelBenchmarkState
  revision: number
  startedAt: number
  finishedAt: number | null
  progress: ModelBenchmarkProgress
  cells: ModelBenchmarkCell[]
  receiptAvailable: boolean
  protocolComplete: boolean
  previousModel: string | null
  restored: boolean | null
  restoreError: string | null
  error: string | null
  storageError: string | null
  warnings: string[]
  saved: boolean
  metadata?: (Pick<ModelBenchmarkReceipt, 'runtime' | 'model' | 'hardware' | 'settings'> & Partial<Pick<ModelBenchmarkReceipt, 'suite'>>) | null
}
export interface ModelBenchmarkSample {
  fixture_id: ModelBenchmarkFixture
  temperature: number
  repeat: number
  seed: number
  status: ModelBenchmarkSampleStatus
  prompt_tokens: number | null
  completion_tokens: number | null
  cached_tokens: number | null
  delivery_seconds: number | null
  delivery_tps: number | null
  ttft_seconds: number | null
  end_to_end_seconds: number | null
  server_decode_tps: number | null
  server_decode_seconds: number | null
  prefill_seconds: number | null
  token_sha: string | null
  error_code: string | null
  /** Required by community-v2; absent in historical community-v1 receipts. */
  finish_reason?: 'stop' | 'length' | null
  route?: '/v1/completions' | '/v1/chat/completions'
}
/** Reviewed community-v1/v2 export schemas. The parser enforces each protocol's exact fields. */
export interface ModelBenchmarkReceipt {
  schema_version: 1
  suite: { id: 'community-v1' | 'community-v2'; sha256: string }
  run_id: string
  created_at: string
  runtime: { tensorfold_version: string; backend: 'mlx' | 'cuda' | 'rocm'; python_version: string; platform: 'macos' | 'linux'; dependencies: Record<string, string | null> }
  model: { repo_id: string; revision: string | null; family: string | null; config_sha256: string | null; tokenizer_sha256: string | null; quantization: string | null; drafter_repo_id: string | null; local: boolean }
  hardware: { cpu_model: string | null; system_memory_bytes: number | null; memory_type: 'unified' | 'dedicated' | 'unknown'; gpus: { name: string; memory_bytes: number | null; cores: number | null }[]; gpu_count: number | null; source: 'detected' | 'partial' | 'unavailable' }
  settings: { tokens: number; repetitions: number; temperatures: number[]; serial: boolean; context_tokens: number | null; managed: boolean; rank_count: number | null; kv_cache?: 'bf16' | 'fp8' | null }
  samples: ModelBenchmarkSample[]
}
