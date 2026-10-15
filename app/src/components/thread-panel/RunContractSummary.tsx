import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import type { AgenticRunContract, AgenticSessionLifecycle, RunVerificationVerdict } from '../../lib/agentic-run-contract'
import { labelize, RunMiniStat } from './primitives'

const VERDICT_TINT: Record<RunVerificationVerdict, string> = {
  verified: 'border-positive/30 bg-positive/10 text-positive',
  unverified: 'border-warn/40 bg-warn/10 text-warn',
  blocked: 'border-warn/40 bg-warn/10 text-warn',
  failed: 'border-danger/30 bg-danger/10 text-danger',
  not_applicable: 'border-line bg-ink-5 text-ink-3',
}

const SESSION_TINT: Record<AgenticSessionLifecycle, string> = {
  idle: 'text-ink-3',
  running: 'text-accent',
  awaiting_approval: 'text-warn',
  queued_followup: 'text-warn',
  suspended: 'text-ink-3',
  blocked: 'text-warn',
  complete: 'text-positive',
  error: 'text-danger',
}

export { SESSION_TINT }

export function RunContractSummary({ contract }: { contract: AgenticRunContract }) {
  const receipt = contract.receipt
  const session = contract.session
  const guard = contract.guardPolicy
  const verdictLabel = labelize(receipt.verdict)
  const sessionLabel = labelize(session.state)
  const branch = contract.branch || 'no branch'
  const model = contract.model || 'model unset'
  const queueText = session.queueDepth > 0 ? ` · ${session.queueDepth} queued` : ''
  const warning = receipt.gaps[0] ?? receipt.warnings[0] ?? null

  return (
    <div className="space-y-2 rounded-xl border border-line bg-card-2 p-2.5">
      <div className="flex min-w-0 items-center gap-2">
        <Icon name={contract.surface === 'chat' ? 'message' : contract.surface === 'assistant' ? 'spark' : 'terminal'} size={14} className="shrink-0 text-ink-3" />
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold capitalize text-ink">{contract.surface} run</span>
        <span className={clsx('shrink-0 rounded-md border px-1.5 py-0.5 text-[12px] font-semibold capitalize', VERDICT_TINT[receipt.verdict])}>{verdictLabel}</span>
      </div>

      <div title={contract.objective} className="line-clamp-2 text-[13px] leading-snug text-ink-2">{contract.objective}</div>

      <div className="grid grid-cols-2 gap-1.5">
        <RunMiniStat label="Target" value={contract.targetKind === 'remote' ? 'Remote' : 'Local'} title={contract.targetLabel} />
        <RunMiniStat label="Branch" value={branch} />
        <RunMiniStat label="Model" value={model} />
        <RunMiniStat label="Approval" value={contract.approvalMode} />
      </div>

      <div className="flex flex-wrap gap-1.5 text-[12px] font-medium">
        <span className={clsx('rounded-md bg-ink-5 px-1.5 py-0.5 capitalize', SESSION_TINT[session.state])}>{sessionLabel}{queueText}</span>
        <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-ink-3">{receipt.changedFiles.length} changes</span>
        <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-ink-3">{receipt.testCount} checks</span>
        <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-ink-3">{receipt.artifactCount} artifacts</span>
        {contract.pendingApprovals > 0 && <span className="rounded-md bg-warn/12 px-1.5 py-0.5 text-warn">{contract.pendingApprovals} approvals</span>}
      </div>

      <div className="flex items-center gap-1.5 border-t border-line pt-2 text-[12px] text-ink-3">
        <Icon name="shield" size={12} className="shrink-0" />
        <span className="min-w-0 flex-1 truncate">Guard {guard.maxSteps} steps · repeat {guard.repeatLimit} · search {guard.searchLimit} · no-progress {guard.noProgressLimit}</span>
      </div>
      {warning && <div className="rounded-lg bg-ink-5 px-2 py-1.5 text-[12px] leading-snug text-ink-3">{warning}</div>}
    </div>
  )
}
