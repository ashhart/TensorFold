// Built-in TensorFold engine types shared by the main process and the renderer (type-only).

export type EngineState = 'stopped' | 'starting' | 'ready' | 'failed'

export interface EngineHardware {
  platform: string
  architecture: string
  chip: string | null
}

// Retrieved only when the user requests client connection details; never part of a snapshot/event.
export interface EngineConnection {
  model: string
  baseUrl: string
  apiKey: string
}

// Latest completed request; null metrics mean the engine/stream supplied no usable measurement.
export interface EnginePerformance {
  model: string
  measuredAt: number
  decodeTokensPerSecond: number | null
  prefillTokensPerSecond: number | null
  firstTokenMs: number | null
  completionTokens: number | null
}

export interface EngineStatus {
  hardware?: EngineHardware
  available: boolean
  version: string | null
  binaryPath: string | null
  state: EngineState
  model: string | null
  drafter: string | null
  port: number | null
  startedAt: number | null
  readyAt: number | null
  message: string | null
  logTail: string[]
  totalMemoryBytes: number
  // An optional lowering-only qualification budget; totalMemoryBytes remains real physical RAM.
  memoryBudgetBytes?: number
  memoryLimitBytes: number
  // The served prompt-plus-reply limit (max_model_len) once the engine is ready.
  contextTokens: number | null
  // Cortheon mode: offered by the bundled engine, chosen by the user, and live in the running server.
  cortheon?: { supported: boolean; enabled: boolean; active: boolean }
}

export interface EngineInstalledModel {
  repo: string
  family: string
  modelType: string
  format: string | null
  contextTokens: number | null
  weightsBytes: number | null
  memoryFloorBytes: number | null
  fitsMemory: boolean | null
  drafter: string | null
  drafterInstalled: boolean | null
  // Library checkpoints are borrowed in place; their files never belong to TensorFold.
  source?: 'download' | 'library' | 'cache'
  title?: string
  localPath?: string
}

export interface EngineLibraryFolder {
  id: string
  path: string
  label: string
}

export interface EngineLibraryModel {
  id: string
  repo: string | null
  title: string
  path: string
  source: 'huggingface' | 'folder'
  folderId: string
  revision: string | null
  modelType: string | null
  format: string | null
  complete: boolean
  compatible: boolean
  loadable: boolean
  reason: string | null
  weightsBytes: number | null
  memoryFloorBytes: number | null
  fitsMemory: boolean | null
  drafterOnly?: boolean
}

export interface EngineLibraryScanOptions { includeHuggingFace?: boolean }

export interface EngineModelLibrarySnapshot {
  includeHuggingFace: boolean
  folders: EngineLibraryFolder[]
  models: EngineLibraryModel[]
  scanning: boolean
  scannedAt: number | null
  warnings: string[]
  truncated: boolean
  cancelled: boolean
}

export interface EngineCapabilities {
  version: string
  chip: string | null
  backends: string[]
  families: Record<string, string[]>
  flags: string[]
  env: string[]
}

export type HubGated = false | 'auto' | 'manual'

export interface HubModelSummary {
  repo: string
  downloads: number | null
  likes: number | null
  lastModified: string | null
  gated: HubGated
  modelType: string | null
  pipelineTag: string | null
  libraryName: string | null
}

export type PreflightVerdict = 'ok' | 'installed' | 'unsupported' | 'format' | 'gated' | 'too-big' | 'disk' | 'error'

export interface ModelPreflight {
  repo: string
  revision: string | null
  resolvedRevision?: string | null
  verdict: PreflightVerdict
  reasons: string[]
  modelType: string | null
  family: string | null
  supported: boolean
  format: string | null
  formatOk: boolean | null
  gated: HubGated
  totalBytes: number | null
  fileCount: number | null
  drafter: { repo: string; totalBytes: number | null; resolvedRevision?: string | null } | null
  memoryFloorBytes: number | null
  memoryLimitBytes: number
  fitsMemory: boolean | null
  freeDiskBytes: number | null
  neededDiskBytes: number | null
  fitsDisk: boolean | null
  cachedBytes: number
  installed: boolean
  checkedAt: number
}

export type PullState = 'running' | 'done' | 'failed' | 'cancelled'

export interface PullProgress {
  repo: string
  revision: string | null
  state: PullState
  phase: string
  receivedBytes: number
  totalBytes: number | null
  lastLine: string | null
  message: string | null
  startedAt: number
  finishedAt: number | null
}

export interface SuggestedModel {
  repo: string
  revision: string | null
  title: string
  note: string
}

export interface EngineSnapshot {
  status: EngineStatus
  performance: EnginePerformance | null
  installed: EngineInstalledModel[]
  pulls: PullProgress[]
  suggestions: SuggestedModel[]
  hubCache: string
}

export type EngineEvent =
  | { kind: 'status'; status: EngineStatus }
  | { kind: 'performance'; performance: EnginePerformance | null }
  | { kind: 'pull'; pull: PullProgress }
  | { kind: 'installed'; installed: EngineInstalledModel[] }
  | { kind: 'learning'; snapshot: EngineLearningSnapshot }

export interface EngineLearningFact {
  id: number
  text: string
  source: string
  at: number
  state: 'queued' | 'learning' | 'learned' | 'missed' | 'replaced'
  links: number[]
  round?: number
  message?: string
  questions?: string[]
  neurons?: { layer: number; neuron: number; key: string }[]
  checks?: number
  recalled?: boolean
  loss?: number
  phase?: string
  since?: number
  step?: number
  of?: number
  trace?: number[][]
}

export interface EngineLearningGraph {
  next: number
  facts: EngineLearningFact[]
  model?: {
    layers: number
    kinds: string
    pocket: { neurons: number; layers: number[]; used: number[] }
    shards: { name: string; bytes: number; pocket: boolean; layers: number[] }[]
    save?: { state: 'saving' | 'saved' | 'failed'; at: number; seconds?: number; bytes?: number; files?: string[]; message?: string }
  }
}

export interface EngineLearningSnapshot {
  supported: boolean
  experimental: true
  enabled: boolean
  active: boolean
  model: string | null
  state: 'off' | 'preparing' | 'ready' | 'teaching' | 'cancelling' | 'failed'
  message: string | null
  // The original shared download is never taught; only this app's private copy is used.
  privateCopy: boolean
  progress: { copiedBytes: number; totalBytes: number } | null
  job: {
    id: string
    state: 'running' | 'cancelling' | 'done' | 'cancelled' | 'failed' | 'interrupted'
    startedAt: number
    finishedAt: number | null
    learned: number
    missed: number
    message: string | null
  } | null
  graph: EngineLearningGraph | null
}
