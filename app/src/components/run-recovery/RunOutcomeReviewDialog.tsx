import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import type { Thread } from '../../lib/types'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { useAccessibleDialog } from '../navigation/useAccessibleDialog'
import { useFlowText } from '../../lib/preferences/useFlowText'

export function RunOutcomeReviewDialog({ thread, onClose }: { thread: Thread; onClose(): void }) {
  const tx = useAuthoredText()
  const { t, authored, known } = useFlowText()
  const [confirmed, setConfirmed] = useState(false), [busy, setBusy] = useState(false), [error, setError] = useState('')
  const panel = useRef<HTMLDivElement>(null)
  useAccessibleDialog({ open: true, onClose: () => { if (!busy) onClose() }, dialogRef: panel })
  const acknowledge = async () => {
    if (!confirmed || !thread.runRecovery) return
    setBusy(true); setError('')
    try {
      const result = await api.reviewRunOutcome({ threadId: thread.id, runId: thread.runRecovery.runId })
      useStore.setState(state => ({ threads: state.threads.map(item => item.id === thread.id && item.runRecovery?.runId === result.reviewedRunId ? { ...item, runRecoveryReviewedId: result.reviewedRunId } : item) }))
      onClose()
    } catch { setError('review.error') }
    finally { setBusy(false) }
  }
  return createPortal(<div className="fixed inset-0 z-[130] flex items-center justify-center bg-canvas/85 p-4"><div ref={panel} role="dialog" aria-modal="true" aria-labelledby="run-outcome-title" tabIndex={-1} className="w-full max-w-[560px] space-y-4 rounded-xl border border-line bg-canvas p-5 shadow-pop">
    <h2 id="run-outcome-title" className="text-[16px] font-medium text-ink">{t("review.title")}</h2>
    <p className="text-[13px] leading-relaxed text-ink-2">{thread.runRecovery?.message && <>{!known(thread.runRecovery.message) && <span>{t('common.diagnostic')}: </span>}{authored(thread.runRecovery.message)}</>}</p>
    <p className="text-[13px] leading-relaxed text-ink-2">{t("review.instructions")}</p>
    <label className="flex items-start gap-2 text-[13px] text-ink"><input type="checkbox" checked={confirmed} disabled={busy} onChange={event => setConfirmed(event.target.checked)} /><span>{t("review.confirm")}</span></label>
    {error && <p role="alert" className="text-[13px] text-danger">{t(error)}</p>}
    <div className="flex flex-wrap justify-end gap-2"><button type="button" disabled={busy} className="rounded-lg border border-line px-3 py-2 text-[13px] text-ink" onClick={onClose}>{t("review.cancel")}</button><button type="button" disabled={!confirmed || busy} className="rounded-lg bg-accent px-3 py-2 text-[13px] text-accent-ink disabled:opacity-40" onClick={() => void acknowledge()}>{t(busy ? 'review.saving' : 'review.record')}</button></div>
    <p className="text-[12px] text-ink-3">{t("review.noReplay")}</p>
  </div></div>, document.body)
}
