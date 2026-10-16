import { recordJobApproval, resolveJobApproval } from '../../agent/jobs'
import type { JobPatch } from '../../agent/jobs'
import type { AgentEvent, Job, JobApprovalDecision } from '../../agent/types'
import type { ApprovalPolicyScope } from '../../agent/approval-policy'
import { requiresSavedApprovalPolicy } from '../../agent/approval-policy'
import { READ_ONLY_TOOLS } from '../../agent/tool-schema-builder'
import { store } from '../state'

export interface JobHandle {
  get(): Job | null
  set(job: Job | null): void
}

interface ApprovalRuntimeDeps {
  pendingApprovals: Map<string, (d: 'allow' | 'allow_always' | 'deny') => void>
  pendingApprovalMeta: Map<string, { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }>
  emitJob(job: Job): void
  emitToRenderer(event: AgentEvent): void
  findApprovalRule(scope: 'tool' | 'connector', key: string, context?: { projectId?: string | null; workspace?: string | null }): Promise<{ decision: 'allow' | 'deny'; [key: string]: any } | null>
  patchJob(job: Job | null, patch: JobPatch): Job | null
  persistJobs(): void
  persistThreads(): void
  // Fired when a tool approval becomes genuinely pending the user (not auto-resolved
  // by a saved policy) — the host uses it to raise an OS notification.
  onApprovalPending?: (meta: { threadId: string; itemId: string; tool: string }) => void
}

// Which tool calls get an audit-log entry. Inverting the READ_ONLY allowlist means EVERY
// side-effecting tool (incl. controller_action, browser_automation, message_compose,
// launch_subagent, create_*, and all mcp__ tools) is audited by default — a new
// side-effecting tool is recorded automatically instead of silently dropped when someone
// forgets to extend a hand-maintained list. Pure reads are not audited.
export function isHighRiskTool(name: string): boolean {
  return !READ_ONLY_TOOLS.has(name)
}

export function createApprovalRuntime(deps: ApprovalRuntimeDeps) {
  function noteApprovalPolicyMatch(meta: { threadId: string; itemId: string; policy: ApprovalPolicyScope }, decision: 'allow' | 'deny') {
    const thread = store.threads.find((item) => item.id === meta.threadId)
    if (!thread) return
    const item = thread.timeline.find((entry) => entry.id === meta.itemId)
    if (!item) return
    const verb = decision === 'allow' ? 'Auto-approved' : 'Auto-denied'
    const output = `${verb} by saved policy. ${meta.policy.explanation}`
    const patch = { meta: meta.policy.label, output, approval: output }
    Object.assign(item, patch)
    thread.updatedAt = Date.now()
    deps.persistThreads()
    deps.emitToRenderer({ runId: '', threadId: thread.id, type: 'item_update', id: item.id, patch })
  }

  function jobApprovalFromMeta(callId: string, meta: { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }) {
    return {
      callId,
      threadId: meta.threadId,
      itemId: meta.itemId,
      tool: meta.tool,
      policy: {
        scope: meta.policy.scope,
        key: meta.policy.key,
        label: meta.policy.label,
        explanation: meta.policy.explanation,
      },
    }
  }

  function persistJobApproval(jobHandle: JobHandle | undefined, callId: string, meta: { threadId: string; itemId: string; tool: string; policy: ApprovalPolicyScope }): void {
    const job = jobHandle?.get()
    if (!job) return
    const result = recordJobApproval(store.jobs, job.id, jobApprovalFromMeta(callId, meta))
    if (!result.job) return
    store.jobs = result.jobs
    jobHandle?.set(result.job)
    deps.persistJobs()
    deps.emitJob(result.job)
  }

  function resolvePersistedJobApproval(callId: string, decision: JobApprovalDecision, jobHandle?: JobHandle): Job | null {
    const result = resolveJobApproval(store.jobs, callId, decision)
    if (!result.job) return null
    store.jobs = result.jobs
    if (jobHandle?.get()?.id === result.job.id) jobHandle.set(result.job)
    deps.persistJobs()
    deps.emitJob(result.job)
    return result.job
  }

  function approvalResolver(controller: AbortController, jobHandle?: JobHandle) {
    return async (callId: string): Promise<'allow' | 'allow_always' | 'deny'> => {
      const meta = deps.pendingApprovalMeta.get(callId)
      const thread = meta ? store.threads.find((candidate) => candidate.id === meta.threadId) : null
      let rule: Awaited<ReturnType<ApprovalRuntimeDeps['findApprovalRule']>> = null
      try {
        rule = meta && meta.policy.classification !== 'phone-control' ? await deps.findApprovalRule(meta.policy.scope, meta.policy.key, {
          projectId: thread?.projectId ?? null,
          workspace: thread?.cwd ?? null,
        }) : null
      } catch {
        // A saved-policy lookup is an optimization over the interactive gate. Any lookup or
        // persistence failure fails closed to the manual path; the persistence layer reports
        // the underlying error without taking down the run.
        rule = null
      }
      // The durable lookup can outlive cancellation. AbortSignal does not replay an already-fired
      // abort to a listener registered later, so entering the Promise path here would park forever;
      // applying a rule after cancellation would be worse. Consume may already be durable, but the
      // cancelled call is denied and the bounded use is conservatively spent.
      if (controller.signal.aborted) {
        deps.pendingApprovals.delete(callId)
        deps.pendingApprovalMeta.delete(callId)
        resolvePersistedJobApproval(callId, 'deny', jobHandle)
        return 'deny'
      }
      if (rule?.decision === 'allow' && meta) {
        noteApprovalPolicyMatch(meta, 'allow')
        deps.pendingApprovalMeta.delete(callId)
        // A saved rule for a package-install policy must resolve as 'allow_always' so the
        // loop's interlock treats it as the saved-policy approval path — resolving a bare
        // 'allow' would hit the package-install branch that force-denies it.
        return requiresSavedApprovalPolicy(meta.policy) ? 'allow_always' : 'allow'
      }
      if (rule?.decision === 'deny' && meta) {
        noteApprovalPolicyMatch(meta, 'deny')
        deps.pendingApprovalMeta.delete(callId)
        return 'deny'
      }
      return new Promise<'allow' | 'allow_always' | 'deny'>((resolve) => {
        if (meta) persistJobApproval(jobHandle, callId, meta)
        const onAbort = () => {
          deps.pendingApprovals.delete(callId)
          deps.pendingApprovalMeta.delete(callId)
          resolvePersistedJobApproval(callId, 'deny', jobHandle)
          resolve('deny')
        }
        const done = (d: 'allow' | 'allow_always' | 'deny') => {
          controller.signal.removeEventListener('abort', onAbort)
          deps.pendingApprovals.delete(callId)
          deps.pendingApprovalMeta.delete(callId)
          resolvePersistedJobApproval(callId, d, jobHandle)
          const job = jobHandle?.get()
          if (job && !controller.signal.aborted) jobHandle?.set(deps.patchJob(job, { status: 'running', blockedReason: null }))
          resolve(d)
        }
        deps.pendingApprovals.set(callId, done)
        // It's genuinely the user's turn now — let the host raise an OS notification.
        if (meta) deps.onApprovalPending?.({ threadId: meta.threadId, itemId: meta.itemId, tool: meta.tool })
        controller.signal.addEventListener('abort', onAbort, { once: true })
      })
    }
  }

  return { approvalResolver, persistJobApproval, resolvePersistedJobApproval }
}
