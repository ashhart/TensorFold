import type { EngineEvent } from '../types'
import type { EngineBinary } from './paths'

export interface EngineServiceOptions {
  binaryPath: string
  appHome: string
  env?: NodeJS.ProcessEnv
  // Exact environment for engine processes; tests use it to reach the fake engine's knobs.
  childEnv?: NodeJS.ProcessEnv
  binary?: EngineBinary
  fetch?: typeof fetch
  totalMemoryBytes?: number
  memoryBudgetBytes?: number
  readyTimeoutMs?: number
  pollMs?: number
  emit: (event: EngineEvent) => void
  onInstalled?: (repos: string[]) => void
}
