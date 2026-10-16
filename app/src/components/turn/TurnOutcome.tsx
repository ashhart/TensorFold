import { Icon } from '../../lib/icons'
import type { TurnSummary } from '../../lib/turn-blocks'

// The model's answer can sound confident after refused or failed tools. Keep
// the host's recorded outcome visible without treating model text as evidence.
export function TurnOutcome({ turn, onShowActivity }: { turn: TurnSummary; onShowActivity: () => void }) {
  if (turn.running) return null
  const receipt = [...turn.work].reverse().find((item) => item.runReceipt)?.runReceipt
  const warnings = Array.isArray(receipt?.warnings) ? receipt.warnings : []
  const needsReview = !!receipt && ['failed', 'cancelled', 'blocked', 'unverified'].includes(receipt.verdict)
  if (!receipt || (!needsReview && !warnings.length)) return null
  const failed = receipt.verdict === 'failed'
  const title = failed ? 'Actions need review'
    : receipt.verdict === 'cancelled' ? 'Run stopped'
    : receipt.verdict === 'blocked' ? 'Some actions were blocked'
    : receipt.verdict === 'unverified' ? 'Changes need verification'
    : 'Safety checks recorded a warning'
  const detail = failed
    ? Number.isFinite(receipt.errors) && receipt.errors > 0
      ? `${receipt.errors} error${receipt.errors === 1 ? ' was' : 's were'} recorded; verify the result before relying on the answer.`
      : 'The run recorded a failure; verify the result before relying on the answer.'
    : receipt.verdict === 'cancelled' ? 'Earlier actions may have completed; check the activity before continuing.'
    : receipt.verdict === 'unverified' ? 'The recorded changes do not have a successful check after the last edit.'
    : receipt.verdict === 'blocked' ? receipt.summary
    : 'Review the recorded warning and tool results before continuing.'
  return <section aria-label="Recorded run outcome" role={failed ? 'alert' : 'status'} className="flex items-start gap-2.5 rounded-xl border border-line bg-card-2 px-3 py-2.5 text-[13px]">
    <Icon name={failed ? 'warning' : 'shield'} size={15} className={failed ? 'mt-0.5 shrink-0 text-danger' : 'mt-0.5 shrink-0 text-warn'} />
    <div className="min-w-0 flex-1"><p className="font-medium text-ink">{title}</p><p className="mt-0.5 break-words leading-relaxed text-ink-2">{detail}</p><button type="button" onClick={onShowActivity} className="mt-2 rounded-md border border-line bg-card px-2 py-1 font-medium text-ink-2 hover:bg-ink-5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent">Show recorded activity</button></div>
  </section>
}
