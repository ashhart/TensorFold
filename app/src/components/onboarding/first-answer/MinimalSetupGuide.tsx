import { useFlowText } from '../../../lib/preferences/useFlowText'
import { CORTHEON_RELEASE_POLICY } from '../../../../shared/release-feature-policy'
import { Icon, type IconName } from '../../../lib/icons'

/** Existing local-server guidance stays separate from the Full first-conversation task. */
export function MinimalSetupGuide({ model }: { model: string | null }) {
  const { t, authored, known, locale } = useFlowText()
  const pointers: Array<{ icon: IconName; title: string; body: string }> = [
    { icon: 'cpu', title: t("minimal.load"), body: t("minimal.loadHelp") },
    { icon: 'plug', title: t("minimal.apps"), body: t("minimal.appsHelp") },
    { icon: 'coffee', title: t("minimal.menu"), body: t("minimal.menuHelp") },
  ]
  return <div className="grid gap-4">
    <div className="rounded-xl border border-line bg-card px-4 py-3 text-[13px] text-ink-2"><span className="font-medium text-ink">{t("minimal.mode")}</span><div className="mt-1 break-words">{model ? t("first.selected", { model }) : t("minimal.noModel")}</div></div>
    <div className="rounded-xl border border-line bg-card shadow-card">{pointers.map((pointer, index) => <div key={pointer.title} className={`flex gap-3 p-4 ${index > 0 ? 'border-t border-line' : ''}`}><Icon name={pointer.icon} size={18} className="mt-0.5 shrink-0 text-accent" /><div><h2 className="text-[14px] font-medium text-ink">{pointer.title}</h2><p className="mt-1 text-[13px] leading-relaxed text-ink-2">{pointer.body}</p></div></div>)}</div>
    <p className="text-[13px] leading-relaxed text-ink-2">{t("minimal.network")}</p>
    <details className="rounded-xl border border-line bg-card p-4"><summary className="cursor-pointer text-[13px] font-medium text-ink focus-visible:outline-2 focus-visible:outline-accent">{t("minimal.data")}</summary><div className="mt-3 grid gap-2 text-[13px] leading-relaxed text-ink-2"><p>{authored(CORTHEON_RELEASE_POLICY.reason)} {t("minimal.apiAvailable")}</p><p>{t("minimal.learning")}</p><p>{t("minimal.keys")}</p></div></details>
    <p className="text-[12px] text-ink-2">{t("minimal.reopen")}</p>
  </div>
}
