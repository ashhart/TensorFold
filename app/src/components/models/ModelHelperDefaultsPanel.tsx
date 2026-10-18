import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import type { Settings } from '../../lib/types'
import { useModelQualifications } from './qualification/useModelQualifications'
import { isExternalProvider } from '../../../shared/provider-trust'
import { currentModelQualification, modelQualificationKey, type ModelHelperRole } from '../../../shared/model-fit'
import { Section } from '../settings/shared'

const roles: Array<{ id: ModelHelperRole; title: string; description: string }> = [
  { id: 'naming', title: 'Conversation naming', get description() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Give a conversation a short title.") } },
  { id: 'document_extraction', title: 'Document facts', get description() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Extract facts from text you explicitly provide; native OCR stays available separately.") } },
  { id: 'search', title: 'Source answers', get description() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Answer from matching local document passages. Full-text chat search remains deterministic.") } },
]
export function ModelHelperDefaultsPanel({ draft, patch }: { draft: Settings; patch(input: Partial<Settings>): void }) {
  const tx = useAuthoredText()
  const { reports, error } = useModelQualifications()
  const current = [...new Map(reports.map(report => [report.key, report])).values()].map(report => currentModelQualification(reports, report)).filter((report): report is NonNullable<typeof report> => !!report)
  const choices = current.filter(report => report.checks.text.state === 'observed' && !isExternalProvider(draft.providers.find(provider => provider.id === report.providerId)))
  return <Section title={tx("Optional helper models")} desc={tx("Use a qualified local model for each small job; defaults never substitute a cloud provider. Helpers run only on request, and a built-in helper must already be loaded.")}>
    <div className="space-y-4">{roles.map(role => {
      const preferred = draft.modelHelperDefaults?.[role.id], selected = preferred ? modelQualificationKey(preferred) : ''
      const available = choices.some(report => report.key === selected)
      return <label key={role.id} className="block text-[13px] text-ink"><span className="font-medium">{tx(role.title)}</span><span className="mt-1 block text-[12px] text-ink-2">{role.description}</span><select aria-label={tx("{value0} helper model", { value0: tx(role.title) })} value={selected} onChange={event => {
        const next = { ...draft.modelHelperDefaults }, choice = choices.find(report => report.key === event.target.value)
        if (choice) next[role.id] = { providerId: choice.providerId, accountId: choice.accountId, model: choice.model }; else delete next[role.id]
        patch({ modelHelperDefaults: next })
      }} className="mt-2 w-full rounded-lg border border-line bg-card px-3 py-2"><option value="">{tx("No optional helper")}</option>{preferred && !available && <option value={selected}>{tx("Configured helper unavailable · no fallback")}</option>}{choices.map(report => <option value={report.key} key={report.key}>{report.model} · {draft.providers.find(provider => provider.id === report.providerId)?.name}</option>)}</select></label>
    })}</div><p className="mt-3 text-[12px] text-ink-3">{tx("Run a fixed capability check in a model’s details to qualify it; results belong to that exact provider, account and configuration. Private account and bot chats keep their existing boundaries.")}</p>{error && <p role="status" className="mt-2 text-[12px] text-warn">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
  </Section>
}
