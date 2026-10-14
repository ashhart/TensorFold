// Loop Mode — the runtime bridge. Turns injected app primitives (spawn a role
// thread, run shell, call the critic model, git checkpoint) into the LoopRuntime
// interface the pure engine consumes. Decoupled via deps so it stays testable;
// the real deps are provided by the main-process loop host.

import type { LoopPhaseId, RunContract } from './loop-types'
import type { LoopRuntime } from './loop-engine'
import { roleSystemPrompt, phaseResultFromTimeline } from './loop-runtime-support'
import { buildCriticPrompt, parseVerdict } from './loop-critic'

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
  /** Run a role's thread to completion with `instruction`; resolve with the new timeline items.
   *  `phase` lets the host enforce role boundaries (read-only planning roles run without write access). */
  runRoleThread(threadId: string, instruction: string, phase?: LoopPhaseId): Promise<RoleTimelineItem[]>
  cleanupThread(threadId: string): Promise<void>
  runShell(cwd: string, command: string): Promise<{ code: number; stdout: string; stderr: string }>
  /** Call the (independent) critic model and return its raw text. */
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
      if (!c.verificationCommands.length) return '(no verification commands configured)'
      const out: string[] = []
      for (const cmd of c.verificationCommands) {
        const r = await deps.runShell(deps.cwd, cmd)
        out.push(`$ ${cmd}\n[exit ${r.code}]\n${(r.stdout + r.stderr).slice(-3000)}`)
      }
      return out.join('\n\n')
    },
    async critique(c, evidence) {
      const { system, user } = buildCriticPrompt(c, evidence, '')
      const raw = await deps.runCritic(system, user, deps.criticModel)
      return parseVerdict(raw)
    },
    checkpoint: (cycle) => deps.checkpoint(cycle),
    rollback: () => deps.rollback(),
    log: (line) => deps.emit(line),
  }
}
