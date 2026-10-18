import type { RunRecoverySnapshot } from '../../../shared/run-recovery'
import { Button } from '../ui'
import { useFlowText } from '../../lib/preferences/useFlowText'

export function RunRecoveryCard({ recovery, onEditRequest, onChooseModel, onReviewEffects, onRetry }: { recovery?: RunRecoverySnapshot | null; onEditRequest?: () => void; onChooseModel?: () => void; onReviewEffects?: () => void; onRetry?: () => void }) {
  const { t, authored, known } = useFlowText()
  if (!recovery || recovery.state === 'completed' || recovery.state === 'running' && !recovery.reason) return null
  const draining = !recovery.canStartNewRun
  const review = recovery.retry === 'requires_review'
  return <section aria-label={t('recovery.label')} aria-busy={draining} className="my-3 rounded-xl border border-line bg-card p-3.5 text-[13px] shadow-card">
    <div role={review ? 'alert' : 'status'} aria-live="polite">
      <h3 className="font-medium text-ink">{t(`recovery.${recovery.state}`)}</h3>
      <p className="mt-1 break-words text-ink-2">{!known(recovery.message) && <span>{t('common.diagnostic')}: </span>}{authored(recovery.message)}</p>
      <p className="mt-2 text-[12px] text-ink-3">{t(recovery.completedTools === 1 ? 'recovery.oneTool' : 'recovery.tools', { count: recovery.completedTools })} · {t(recovery.completedMutations === 1 ? 'recovery.oneChange' : 'recovery.changes', { count: recovery.completedMutations })}{draining ? ` · ${t(recovery.pendingOperations === 1 ? 'recovery.oneActive' : 'recovery.active', { count: recovery.pendingOperations })}` : ''}</p>
      {recovery.pendingMutations > 0 && <p className="mt-1 text-[12px] text-ink-2">{t(recovery.pendingMutations === 1 ? 'recovery.oneMutation' : 'recovery.mutations', { count: recovery.pendingMutations })}</p>}
      {draining ? <p className="mt-2 text-[12px] text-ink-3">{t('recovery.wait')}</p> : review ? <p className="mt-2 text-[12px] text-ink-2">{!known(recovery.retryReason) && <span>{t('common.diagnostic')}: </span>}{authored(recovery.retryReason)}</p> : null}
    </div>
    <div className="mt-3 flex flex-wrap gap-2">
      {onEditRequest && <Button size="xs" disabled={draining} onClick={onEditRequest}>{t('recovery.adjust')}</Button>}
      {onChooseModel && <Button size="xs" disabled={draining} onClick={onChooseModel}>{t('recovery.model')}</Button>}
      {review && onReviewEffects && <Button size="xs" disabled={draining} onClick={onReviewEffects}>{t('recovery.review')}</Button>}
      {recovery.retry === 'safe' && onRetry && <Button size="xs" disabled={draining} onClick={onRetry}>{t(recovery.delivery === 'not_sent' ? 'recovery.retry' : 'recovery.retryRefused')}</Button>}
    </div>
  </section>
}
