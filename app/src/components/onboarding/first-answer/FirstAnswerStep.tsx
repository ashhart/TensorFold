import { useFlowText } from '../../../lib/preferences/useFlowText'
import type { FirstAnswerReadiness, FirstAnswerStarterId } from './model'
import { FIRST_ANSWER_STARTERS, firstAnswerNextAction } from './model'

export function FirstAnswerStep({ starter, onStarter, model, readiness, disabled = false }: { starter: FirstAnswerStarterId; onStarter: (id: FirstAnswerStarterId) => void; model: string | null; readiness: FirstAnswerReadiness; disabled?: boolean }) {
  const { t, authored, known, locale } = useFlowText()
  const selected = FIRST_ANSWER_STARTERS.find((item) => item.id === starter)!
  const next = firstAnswerNextAction(readiness)
  const checkpoint = (complete: boolean, pending: string, done: string) => <><span aria-hidden="true" className={complete ? 'text-positive' : 'text-ink-3'}>{complete ? '✓' : '○'}</span><span>{complete ? done : pending}</span></>
  return <div className="space-y-4" data-testid="first-answer-step">
    <div className="rounded-xl border border-line bg-card p-4">
      <h2 className="text-[14px] font-medium text-ink">{t("first.title")}</h2>
      <ol className="mt-3 space-y-2 text-[13px] leading-relaxed text-ink-2" aria-label={t("first.checklist")}>
        <li className="flex gap-2">{checkpoint(readiness.selected, t("first.choose"), t("first.selected", { model: model || t("first.configured") }))}</li>
        <li className="flex gap-2">{checkpoint(readiness.builtIn ? readiness.cached : readiness.configured, readiness.builtIn ? t("first.download") : t("first.connection"), readiness.builtIn ? t("first.available") : t("first.connected"))}</li>
        <li className="flex gap-2">{checkpoint(readiness.loaded, readiness.loading ? t("first.loading") : readiness.builtIn ? t("first.validate") : t("first.providerSend"), t("first.loaded"))}</li>
        <li className="flex gap-2"><span aria-hidden="true" className="text-ink-3">○</span><span>{t("first.review")}</span></li>
      </ol>
      <p role="status" className="mt-3 border-t border-line pt-3 text-[13px] leading-relaxed text-ink-2">{authored(next.message)}</p>
    </div>
    <fieldset disabled={disabled} className="min-w-0 space-y-2">
      <legend className="mb-2 text-[14px] font-medium text-ink">{t("first.task")}</legend>
      {FIRST_ANSWER_STARTERS.map((item) => <label key={item.id} className="flex cursor-pointer items-start gap-3 rounded-lg border border-line bg-card px-3 py-2.5 text-[13px] text-ink-2">
        <input type="radio" name="first-answer-starter" value={item.id} checked={starter === item.id} onChange={() => onStarter(item.id)} className="mt-1 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent" />
        <span className="min-w-0"><span className="block font-medium text-ink">{authored(item.label)}</span><span className="mt-0.5 block leading-relaxed">{authored(item.description)}</span></span>
      </label>)}
    </fieldset>
    {selected.prompt && <div className="rounded-lg bg-card-2 p-3"><p className="text-[12px] font-medium text-ink-2">{t("first.draft")}</p><p className="mt-1 whitespace-pre-wrap break-words text-[13px] leading-relaxed text-ink">{authored(selected.prompt)}</p></div>}
    <p className="text-[13px] leading-relaxed text-ink-2">{readiness.providerKind === 'cloud' ? t("first.cloudPrivacy") : readiness.builtIn ? t("first.localPrivacy") : t("first.runtimePrivacy")}</p>
    <p className="text-[13px] leading-relaxed text-ink-2">{t("first.stopHelp")}</p>
  </div>
}
