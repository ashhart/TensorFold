import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useRef, useState } from 'react'
import { buildPersonalRoutine, type PersonalRoutineInput } from '../../../shared/personal-routines'
import { CONNECTOR_CATALOG } from '../../../shared/connector-catalog'
import { isExternalProvider } from '../../../shared/provider-trust'
import { useStore } from '../../lib/store'
import type { AgentPreset, AutomationSourceScope } from '../../lib/types'
import { Button, Input } from '../ui'

export function BotRoutineSetup({ bot, onClose }: { bot: AgentPreset; onClose: () => void }) {
  const tx = useAuthoredText()
  const settings = useStore((state) => state.settings)!
  const [kind, setKind] = useState<PersonalRoutineInput['kind']>('morning-brief')
  const [selected, setSelected] = useState<string[]>([])
  const [time, setTime] = useState('09:00')
  const [timeZone, setTimeZone] = useState(() => Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const saved = useRef<{ id: string; definition: string } | null>(null)
  const provider = settings.providers.find((item) => item.id === bot.providerId)
  const sources: Array<AutomationSourceScope & { label: string }> = []
  const add = (connectorId: AutomationSourceScope['connectorId'], values: Array<{ id: string; label?: string; enabled?: boolean }>) => {
    for (const source of values) if (source.enabled) sources.push({ connectorId, sourceId: source.id, label: `${CONNECTOR_CATALOG[connectorId].name}: ${source.label || source.id}` })
  }
  add('email', settings.email.sources); add('slack', settings.slack.sources); add('discord', settings.discord.sources)
  add('mattermost', settings.mattermost.sources); add('matrix', settings.matrix.sources); add('googlechat', settings.googlechat.sources)
  add('whatsapp', settings.whatsapp.sources); add('calendar', settings.calendar.sources); add('docs', settings.documents.sources)
  const save = async (preview: boolean) => {
    if (busy) return
    setBusy(true); setError('')
    try {
      const [atHour, atMinute] = time.split(':').map(Number)
      const config = buildPersonalRoutine({ kind, botId: bot.id, providerId: bot.providerId, providerAccountId: bot.providerAccountId, model: bot.model, timeZone, atHour, atMinute,
        sourceScope: sources.filter((source) => selected.includes(`${source.connectorId}:${source.sourceId}`)).map(({ connectorId, sourceId }) => ({ connectorId, sourceId })),
      })
      const definition = JSON.stringify(config)
      if (!saved.current) {
        const created = await useStore.getState().createAutomation(config)
        saved.current = { id: created.id, definition }
      } else if (saved.current.definition !== definition) {
        await useStore.getState().updateAutomation(saved.current.id, config)
        saved.current.definition = definition
      }
      if (preview) await useStore.getState().runAutomation(saved.current.id)
      onClose()
      useStore.getState().setView({ name: 'automations' })
    } catch (reason) { setError(`${saved.current ? 'Your disabled routine is saved; retry uses the same routine. ' : ''}${reason instanceof Error ? reason.message : 'The routine could not be saved.'}`) }
    finally { setBusy(false) }
  }
  return <section aria-label={tx("Routine for {value0}", { value0: bot.name })} className="my-4 space-y-4 rounded-xl border border-line-strong bg-card-2 p-4">
    <div><h3 className="text-[14px] font-medium">{tx("Make this bot useful every day")}</h3><p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("A routine reads only the selected cached sources and returns a brief to your TensorFold inbox, with no tools or automatic messages.")}</p></div>
    <label className="block text-[13px] text-ink-2">{tx("Routine")}<select className="mt-1 h-9 w-full rounded-lg border border-line bg-card px-2 text-ink" value={kind} onChange={(event) => setKind(event.target.value as PersonalRoutineInput['kind'])}><option value="morning-brief">{tx("Morning brief")}</option><option value="meeting-prep">{tx("Meeting preparation")}</option><option value="follow-up">{tx("Follow-up suggestions")}</option></select></label>
    <fieldset className="space-y-2"><legend className="mb-2 text-[13px] font-medium text-ink-2">{tx("Choose source accounts")}</legend>{sources.length ? sources.map((source) => { const key = `${source.connectorId}:${source.sourceId}`; return <label key={key} className="flex items-start gap-2 text-[13px] text-ink-2"><input type="checkbox" checked={selected.includes(key)} disabled={busy || (!selected.includes(key) && selected.length >= 32)} onChange={(event) => setSelected(event.target.checked ? [...selected, key] : selected.filter((item) => item !== key))} className="mt-0.5 accent-accent" /><span className="break-words">{source.label}</span></label> }) : <p className="text-[13px] text-ink-2">{tx("Connect a source in Settings before creating a brief.")}</p>}</fieldset>
    <div className="grid gap-3 sm:grid-cols-2"><label className="text-[13px] text-ink-2">{tx("Daily time")}<Input type="time" value={time} disabled={busy} onChange={(event) => setTime(event.target.value)} className="mt-1" /></label><label className="text-[13px] text-ink-2">{tx("Timezone")}<Input value={timeZone} disabled={busy} onChange={(event) => setTimeZone(event.target.value)} className="mt-1" /></label></div>
    <p className="text-[12px] leading-relaxed text-ink-2">{isExternalProvider(provider) ? tx("Your selected external model will receive these source excerpts.") : tx("Uses this bot’s selected local model.")} {tx("Saved routines start disabled; Preview runs once and keeps scheduling off, and you can enable the daily schedule in Automations while this Mac is awake.")}</p>
    {error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    <div className="flex flex-wrap justify-end gap-2"><Button size="sm" disabled={busy} onClick={onClose}>{tx("Cancel")}</Button><Button size="sm" disabled={busy || !selected.length} onClick={() => void save(false)}>{tx("Save disabled")}</Button><Button variant="primary" size="sm" disabled={busy || !selected.length} onClick={() => void save(true)}>{busy ? tx("Saving…") : tx("Preview once")}</Button></div>
  </section>
}
