import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { useState } from 'react'
import { api } from '../../lib/ipc'
import { canMarkDiffsReviewed, workbenchDoneVerdict, type WorkbenchPrReviewState } from '../../lib/workbench-verdict'
import { Icon } from '../../lib/icons'
import type { AgenticRunContract } from '../../lib/agentic-run-contract'
import type { Thread } from '../../lib/types'

// Workbench done-gate (VST-017): "done means tests passed + diff reviewed", rendered as an
// explicit verdict with the missing half named. The mark-reviewed action stamps
// thread.diffReviewedAt; a NEWER run receipt invalidates the mark (freshness is derived).
export function WorkbenchDoneGate({ thread, contract, reviewState }: { thread: Thread; contract: AgenticRunContract; reviewState: WorkbenchPrReviewState }) {
  const tx = useAuthoredText()
  const [marking, setMarking] = useState(false)
  const verdict = workbenchDoneVerdict({
    surface: contract.surface,
    receipt: contract.receipt,
    diffReviewedAt: thread.diffReviewedAt,
    receiptFinishedAt: thread.lastRunReceipt?.finishedAt ?? null,
    reviewRequired: reviewState.required,
    reviewApproved: reviewState.approved,
  })
  if (!verdict.applicable) return null
  const markReviewed = async () => {
    setMarking(true)
    try {
      await api.patchThread(thread.id, { diffReviewedAt: Date.now() })
    } finally {
      setMarking(false)
    }
  }
  return (
    <div className={clsx('mt-2 rounded-xl border p-2.5', verdict.done ? 'border-positive/40 bg-positive/5' : 'border-line bg-card-2')}>
      <div className="flex items-center gap-2">
        <Icon name={verdict.done ? 'check' : 'circle'} size={13} className={clsx('shrink-0', verdict.done ? 'text-positive' : 'text-ink-3')} />
        <span className="min-w-0 flex-1 truncate text-[13px] font-semibold text-ink">{verdict.done ? tx("Done") : tx("Not done")}</span>
        <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', verdict.testsPassed ? 'bg-positive/12 text-positive' : 'bg-ink-5 text-ink-3')}>{tx("tests ")}{verdict.testsPassed ? '✓' : '—'}</span>
        <span className={clsx('rounded-md px-1.5 py-0.5 text-[12px] font-medium', verdict.diffReviewed ? 'bg-positive/12 text-positive' : 'bg-ink-5 text-ink-3')}>{tx("review ")}{verdict.diffReviewed ? '✓' : '—'}</span>
      </div>
      <div className="mt-1 text-[12px] leading-snug text-ink-3">{verdict.detail}</div>
      {reviewState.required && !reviewState.approved && (
        <div className="mt-1 text-[12px] text-warn">{tx("Evidence-bound review passes: ")}{reviewState.recordedApprovals}/{reviewState.requiredApprovals}</div>
      )}
      {canMarkDiffsReviewed(verdict) && (
        <button
          className="mt-1.5 rounded-lg border border-line bg-card px-2 py-1 text-[12px] font-medium text-ink-2 hover:bg-ink-5 disabled:opacity-50"
          disabled={marking}
          onClick={() => { void markReviewed() }}
        >
          {marking ? tx("Marking…") : tx("Mark diffs reviewed")}
        </button>
      )}
    </div>
  )
}
