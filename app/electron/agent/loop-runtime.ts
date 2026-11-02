// Adapt host operations into observed phase, verification, and critic results.

import type { LoopPhaseId, LoopShellResult, RunContract, VerificationCheck } from './loop-types'
import type { LoopRuntime } from './loop-engine'
import { roleSystemPrompt, phaseResultFromTimeline } from './loop-runtime-support'
import { buildCriticPrompt, parseVerdict } from './loop-critic'
import { verificationCheck, verificationError } from './loop-verification'

export interface RoleTimelineItem {
  kind: string
  text?: string
  tool?: string
  state?: string
}

export interface LoopModelRef {
  providerId: string
  model: string
}

export interface LoopRuntimeDeps {
  cwd: string
  projectId?: string | null
  workerModel: LoopModelRef
  criticModel: LoopModelRef
  createRoleThread(opts: { cwd: string; projectId?: string | null; providerId: string; model: string; systemPrompt: string; kind: 'task' | 'chat' }): Promise<{ id: string }>
  // Phase identity lets the host enforce role-specific write permissions.
  runRoleThread(threadId: string, instruction: string, phase?: LoopPhaseId): Promise<RoleTimelineItem[]>
  cleanupThread(threadId: string): Promise<void>
  runShell(cwd: string, command: string): Promise<LoopShellResult>
  isCancelled?(): boolean
  // The independent critic returns text; execution results retain separate authority.
  runCritic(system: string, user: string, model: LoopModelRef): Promise<string>
  checkpoint(cycle: number): Promise<string>
  rollback(): Promise<void>
  emit(line: string): void
}

function phaseInstruction(phase: LoopPhaseId, contract: RunContract): string {
  return `Complete the ${phase.toUpperCase()} phase for this goal:\n${contract.goal}\n\nWork in the current directory. When you are done, summarize concretely what you did, with evidence (commands run and their actual results).`
}

export function createLoopRuntime(contract: RunContract, deps: LoopRuntimeDeps): LoopRuntime {
  return {
    async runPhase(phase, c, _cycle) {
      const thread = await deps.createRoleThread({
        cwd: deps.cwd,
        projectId: deps.projectId,
        providerId: deps.workerModel.providerId,
        model: deps.workerModel.model,
        systemPrompt: roleSystemPrompt(phase, c),
        kind: 'task',
      })
      try {
        const items = await deps.runRoleThread(thread.id, phaseInstruction(phase, c), phase)
        return phaseResultFromTimeline(phase, items)
      } finally {
        await deps.cleanupThread(thread.id).catch(() => {})
      }
    },
    async runVerification(c) {
      const checks: VerificationCheck[] = []
      for (const cmd of c.verificationCommands) {
        if (deps.isCancelled?.()) return { checks, cancelled: true }
        try {
          checks.push(verificationCheck(cmd, await deps.runShell(deps.cwd, cmd)))
        } catch (error) {
          checks.push(verificationError(cmd, error))
        }
        if (checks.at(-1)?.outcome === 'cancelled') return { checks, cancelled: true }
      }
      return { checks, cancelled: deps.isCancelled?.() === true }
    },
    async critique(c, evidence) {
      const { system, user } = buildCriticPrompt(c, evidence, '')
      const raw = await deps.runCritic(system, user, deps.criticModel)
      return parseVerdict(raw)
    },
    checkpoint: (cycle) => deps.checkpoint(cycle),
    rollback: () => deps.rollback(),
    isCancelled: () => deps.isCancelled?.() === true,
    log: (line) => deps.emit(line),
  }
}
