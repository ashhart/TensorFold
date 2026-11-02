// Run inputs and mutable counters shared by the run lifecycle and its turn handlers.

import { type ToolContext } from './tools'
import type { PluginRuntime } from './plugin-runtime'
import { type RunRecoverySnapshot } from '../../shared/run-recovery'
import {
  type AgentCue,
  type AgentEvent,
  type AgenticGuardPolicySnapshot,
  type ApprovalMode,
  type Provider,
  type Thread,
} from './types'

export interface RunDeps {
  // Host-captured admission cannot be replaced by renderer or model input.
  assertRequestAllowed?: () => void
  onRecovery?: (snapshot: RunRecoverySnapshot) => void
  thread: Thread
  provider: Provider
  model: string
  approvalMode: ApprovalMode
  temperature: number
  reasoningEffort: string
  serviceTier?: string
  emit: (e: AgentEvent) => void
  runId?: string

  requestApproval: (callId: string) => Promise<'allow' | 'allow_always' | 'deny'>
  signal: AbortSignal
  persist: () => void

  autoCompact?: boolean
  contextLimit?: number
  toolContext?: Partial<ToolContext>
  auditToolCall?: (event: { tool: string; args: Record<string, any>; resultOk: boolean; output: string }) => void | Promise<void>

  observe?: (kind: 'api.request' | 'api.response', payload: Record<string, unknown>) => void
  drainCues?: () => AgentCue[]
  waitForCue?: (timeoutMs: number, signal: AbortSignal) => Promise<void>
  maxSteps?: number
  maxWallMs?: number
  guardPolicy?: AgenticGuardPolicySnapshot

  memoryPolicy?: import('./types').ThreadMemoryPolicy
  projectMemory?: string

  personaBlock?: string

  // Channel instructions stay ephemeral and never enter persisted history or compaction.
  channelPrompt?: string
  responseLocale?: string

  // Harvested facts become reviewable memory candidates rather than automatic instructions.
  onMemoryHarvest?: (candidates: string[]) => void

  onProviderRateLimits?: (input: { providerId: string; accountId: string | null; snapshot: unknown }) => void

  // Embedders can isolate plugins; host runs use the trusted process-wide runtime.
  pluginRuntime?: PluginRuntime
}

export interface RunState {
  spillSeq: number
  textLoopBreaks: number
  repeatedActionStop: boolean
  goalIters: number
  verifyNudges: number
  lastVerifyNudgedChangeIndex: number
  budgetStop: 'time budget' | 'step budget' | 'repeated action limit' | null
  completedNaturally: boolean
  uncertainActionStop: boolean
  lastAssistantId: string | null
  textLengthContinuations: number
  todoMutationsSincePlanTouch: number
  todoNudges: number
  todoStopReminders: number
}
