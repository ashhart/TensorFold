import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { Icon, type IconName } from '../lib/icons'
import { relativeTime } from '../lib/format'
import { Button, Field, Input, Segmented, Toggle } from './ui'
import type { Automation, AutomationDeliver } from '../lib/types'
import { accountLabel, modelDisplayName } from '../lib/provider-accounts'
import { modelChoiceValue, parseModelChoice, resolveModelSelect } from '../lib/automation-model-select'
import { parseAutomationIntent } from '../lib/automation-intent'
import { automationTimeZones, dailyAutomationPreview, deviceAutomationTimeZone, normalizeAutomationTimeZone } from '../../shared/automation-time'

const pad = (n: number) => String(n).padStart(2, '0')

// Risks a run may need acknowledged before it fires (VST-038): a run whose required risks aren't all acknowledged is refused by the
// scheduler's risk gate.
const RISK_CATALOG: { id: string; label: string; desc: string }[] = [
  { id: 'local-checkout', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Edits your working copy") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "The agent may modify files in the real project folder, not a scratch copy.") } },
  { id: 'network', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Makes network calls") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "The run can reach the internet (web search, API calls, fetches).") } },
  { id: 'external-delivery', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Sends outbound messages") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Reviewed results can leave the app (email / Telegram / WhatsApp) — potentially to other people.") } },
  { id: 'spend', get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Spends model budget") }, get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Recurring runs consume tokens/credits on the pinned provider.") } },
]

const DELIVER_ICON: Record<AutomationDeliver, IconName> = { inbox: 'message', telegram: 'send', email: 'mail', whatsapp: 'at' }
const DELIVER_LABEL: Record<AutomationDeliver, string> = { get inbox() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "In-app inbox") }, get telegram() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Telegram") }, get email() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Email") }, get whatsapp() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "WhatsApp") } }

const INTERVALS: { value: number; label: string }[] = [
  { value: 15, get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every 15 min") } }, { value: 30, get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every 30 min") } }, { value: 60, get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Hourly") } },
  { value: 180, get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every 3 hours") } }, { value: 360, get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every 6 hours") } }, { value: 720, get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every 12 hours") } }, { value: 1440, get label() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Once a day") } },
]

function scheduleSummary(a: Automation): string {
  if (a.scheduleKind === 'manual') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Manual")
  if (a.scheduleKind === 'daily') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Daily · {value0}:{value1} · {value2}", { value0: String(pad(a.atHour)), value1: String(pad(a.atMinute)), value2: String(a.timeZone ?? authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Legacy host-local timezone")) })
  const m = a.everyMinutes
  if (m % 1440 === 0) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every {value0}d", { value0: String(m / 1440) })
  if (m % 60 === 0) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every {value0}h", { value0: String(m / 60) })
  return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Every {value0} min", { value0: String(m) })
}

function nextLabel(a: Automation): string {
  if (!a.enabled) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Paused")
  if (a.scheduleError) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Fix schedule")
  if (a.scheduleKind === 'manual') return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "On demand")
  if (!a.nextRunAt) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Scheduled")
  const ms = a.nextRunAt - Date.now()
  if (ms <= 0) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Due now")
  const min = Math.round(ms / 60000)
  if (min < 60) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Next in {value0}m", { value0: String(min) })
  const h = Math.round(min / 60)
  if (h < 24) return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Next in {value0}h", { value0: String(h) })
  return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Next in {value0}d", { value0: String(Math.round(h / 24)) })
}

// ------------------------------------------------------------------ list page
export function Automations() {
  const tx = useAuthoredText()
  const automations = useStore((s) => s.automations)
  const settings = useStore((s) => s.settings)
  const createAutomation = useStore((s) => s.createAutomation)
  const [editingId, setEditingId] = useState<string | null>(null)
  const [describe, setDescribe] = useState('')
  const [createError, setCreateError] = useState<string | null>(null)

  const editing = automations.find((a) => a.id === editingId)

  const startNew = async (tpl?: Partial<Automation>) => {
    setCreateError(null)
    const timeZone = deviceAutomationTimeZone()
    const base: Partial<Automation> = {
      name: 'New automation', prompt: '', scheduleKind: 'manual', mode: 'chat', deliverTo: 'inbox', enabled: false,
      providerId: settings?.activeProviderId ?? null, providerAccountId: settings?.activeProviderAccountId ?? null, model: settings?.activeModel ?? null,
      ...(timeZone ? { timeZone } : {}),
      ...tpl,
    }
    try {
      const a = await createAutomation(base)
      setEditingId(a.id)
      return true
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : tx("Could not create this automation."))
      return false
    }
  }

  // NL builder (VST-038): parse a plain-English description into a draft the user reviews.
  const startFromDescription = async () => {
    const text = describe.trim()
    if (!text) return
    try {
      const intent = parseAutomationIntent(text, deviceAutomationTimeZone())
      const created = await startNew({
        name: intent.name, prompt: intent.prompt, scheduleKind: intent.scheduleKind,
        everyMinutes: intent.everyMinutes, atHour: intent.atHour, atMinute: intent.atMinute,
        ...(intent.timeZone ? { timeZone: intent.timeZone } : {}),
        mode: intent.mode, deliverTo: intent.deliverTo,
      })
      if (created) setDescribe('')
    } catch (error) { setCreateError(error instanceof Error ? error.message : tx("Check the schedule and try again.")) }
  }

  if (editing) return <Editor key={editing.id} automation={editing} onClose={() => setEditingId(null)} />

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[760px] px-7 py-9">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-[20px] font-semibold tracking-[-0.02em] text-ink">{tx("Automations")}</h1>
            <p className="mt-1 text-[13px] text-ink-3">{tx("Pin a model to a prompt and a schedule — it runs on its own and delivers the result.")}</p>
          </div>
          <Button variant="primary" icon="plus" onClick={() => startNew()}>{tx("New automation")}</Button>
        </div>

        {/* NL builder (VST-038): describe it in words → a reviewable draft. */}
        {createError && <p role="alert" className="mt-4 text-[13px] text-danger">{createError}</p>}
        <div className="mt-4 rounded-xl border border-line bg-card p-3 shadow-card">
          <div className="mb-1.5 flex items-center gap-1.5 text-[13px] font-medium text-ink-2"><Icon name="spark" size={13} className="text-accent" />{tx("Describe it in plain English")}</div>
          <div className="flex items-center gap-2">
            <Input
              value={describe}
              onChange={(e) => setDescribe(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') void startFromDescription() }}
              placeholder={tx("e.g. daily at 9am in Europe/London, summarize my inbox")}
            />
            <Button variant="default" icon="arrowRight" onClick={() => void startFromDescription()} disabled={!describe.trim()}>{tx("Build")}</Button>
          </div>
          <div className="mt-1.5 text-[12px] leading-snug text-ink-4">{tx("Sets the schedule, mode, and delivery it can infer — you review and adjust before enabling.")}</div>
        </div>

        {automations.length === 0 ? (
          <div className="mt-5 flex flex-col items-center justify-center rounded-2xl border border-dashed border-line py-12 text-center">
            <span className="flex h-12 w-12 items-center justify-center rounded-2xl bg-ink-5 text-ink-3"><Icon name="clock" size={24} /></span>
            <div className="mt-3 text-[14px] font-medium text-ink">{tx("No automations yet")}</div>
            <div className="mt-1 max-w-[400px] text-[13px] leading-relaxed text-ink-3">
              {tx("Create a schedule (interval or daily) that runs a task on any model and sends the output to your inbox, Telegram, or email.")}</div>
          </div>
        ) : (
          <div className="mt-5 space-y-2.5">
            {automations.map((a) => <Row key={a.id} a={a} onEdit={() => setEditingId(a.id)} />)}
          </div>
        )}

        <h2 className="mt-9 text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">{tx("Start from a template")}</h2>
        <div className="mt-2.5 grid grid-cols-1 gap-2 sm:grid-cols-3">
          {TEMPLATES.map((t) => (
            <button key={t.name} onClick={() => startNew(t.make())} className="group rounded-xl border border-line bg-card p-3 text-left shadow-card transition-colors hover:border-line-strong">
              <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name={t.icon} size={16} /></span>
              <div className="mt-2 text-[13px] font-medium text-ink">{t.name}</div>
              <div className="mt-0.5 text-[13px] leading-snug text-ink-3">{t.desc}</div>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}

// ------------------------------------------------------------------ a row
function Row({ a, onEdit }: { a: Automation; onEdit: () => void }) {
  const tx = useAuthoredText()
  const updateAutomation = useStore((s) => s.updateAutomation)
  const deleteAutomation = useStore((s) => s.deleteAutomation)
  const runAutomation = useStore((s) => s.runAutomation)
  const [open, setOpen] = useState(false)
  const lastRun = a.runs[0]
  const running = lastRun?.status === 'running'

  return (
    <div className="rounded-xl border border-line bg-card shadow-card transition-colors hover:border-line-strong">
      <div className="flex items-center gap-3 px-3.5 py-3">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-ink-5 text-ink-2"><Icon name={DELIVER_ICON[a.deliverTo]} size={17} /></span>
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="truncate text-[14px] font-medium text-ink">{a.name}</span>
            <span className="shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">{scheduleSummary(a)}</span>
          </div>
          <div className="mt-0.5 flex items-center gap-2 text-[13px] text-ink-3">
            <span className="flex items-center gap-1"><Icon name="cpu" size={11} />{a.model || tx("no model")}</span>
            <span className="text-ink-4">·</span>
            <span>{a.mode === 'agent' ? tx("Agent") : tx("Chat")}</span>
            <span className="text-ink-4">·</span>
            {running
              ? <span className="flex items-center gap-1 text-warn"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warn" />{tx("Running")}</span>
              : a.lastStatus === 'error'
                ? <span className="flex items-center gap-1 text-danger"><Icon name="warning" size={11} />{tx("Last run failed")}</span>
                : a.lastRunAt
                  ? <span className="flex items-center gap-1 text-positive"><Icon name="check" size={11} />{tx("Ran ")}{relativeTime(a.lastRunAt)}</span>
                  : <span>{tx("Never run")}</span>}
          </div>
        </div>
        <span className="hidden shrink-0 text-[12px] text-ink-3 sm:block">{nextLabel(a)}</span>
        <Toggle checked={a.enabled} onChange={(v) => updateAutomation(a.id, { enabled: v })} label={tx("Enabled")} />
        <button onClick={() => runAutomation(a.id)} disabled={running} title={tx("Run now")} className="flex h-8 w-8 items-center justify-center rounded-lg border border-line text-ink-2 transition-colors hover:border-line-strong hover:text-ink disabled:opacity-40"><Icon name="play" size={14} /></button>
        <button onClick={onEdit} title={tx("Edit")} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"><Icon name="edit" size={15} /></button>
        <button onClick={() => setOpen((v) => !v)} title={tx("History")} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"><Icon name={open ? 'chevronDown' : 'chevronRight'} size={15} /></button>
      </div>

      {open && (
        <div className="border-t border-line px-3.5 py-3">
          <p className="text-[13px] leading-relaxed text-ink-2"><span className="text-ink-3">{tx("Prompt — ")}</span>{a.prompt || <span className="italic text-ink-3">{tx("empty")}</span>}</p>
          {a.runs.length === 0 ? (
            <div className="mt-2 text-[13px] text-ink-3">{tx("No runs yet. Hit ▶ to run it now.")}</div>
          ) : (
            <div className="mt-2.5 space-y-1.5">
              {a.runs.slice(0, 4).map((r) => (
                <div key={r.id} className="rounded-lg border border-line bg-card-2/50 px-2.5 py-2">
                  <div className="flex items-center gap-2 text-[12px]">
                    {r.status === 'running' ? <span className="flex items-center gap-1 text-warn"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-warn" />{tx("Running")}</span>
                      : r.status === 'error' ? <span className="flex items-center gap-1 text-danger"><Icon name="warning" size={10} />{tx("Error")}</span>
                      : <span className="flex items-center gap-1 text-positive"><Icon name="check" size={10} />OK</span>}
                    <span className="text-ink-3">{relativeTime(r.startedAt)}</span>
                    <div className="flex-1" />
                    <span className="flex items-center gap-1 text-ink-3"><Icon name="arrowRight" size={10} />{DELIVER_LABEL[a.deliverTo]}</span>
                  </div>
                  {(r.output || r.error) && <pre className="scroll-thin mt-1.5 max-h-44 overflow-auto whitespace-pre-wrap break-words font-mono text-[12px] leading-relaxed text-ink-2">{r.error ? r.error : r.output}</pre>}
                  {r.deliveryStatus && <div className="mt-2 text-[12px] leading-relaxed text-ink-2">{r.deliveryStatus === 'inbox' ? tx("Result kept in the TensorFold inbox.") : r.deliveryStatus === 'blocked' ? r.deliveryError || tx("External delivery is blocked; the result is kept in the inbox.") : tx("An exact message draft was created for your review.")}{r.deliveryStatus === 'pending_approval' && <button className="ml-2 rounded text-accent underline underline-offset-4" onClick={() => useStore.getState().setView({ name: 'assistant' })}>{tx("Review in Assistant")}</button>}</div>}
                </div>
              ))}
            </div>
          )}
          <div className="mt-2.5 flex justify-end">
            <button onClick={() => { if (confirm(tx("Delete automation \"{value0}\"?", { value0: a.name }))) deleteAutomation(a.id) }} className="flex items-center gap-1.5 text-[13px] text-ink-3 transition-colors hover:text-danger"><Icon name="trash" size={13} />{tx("Delete")}</button>
          </div>
        </div>
      )}
    </div>
  )
}

// ------------------------------------------------------------------ editor
function Editor({ automation, onClose }: { automation: Automation; onClose: () => void }) {
  const tx = useAuthoredText()
  const settings = useStore((s) => s.settings)
  const updateAutomation = useStore((s) => s.updateAutomation)
  const runAutomation = useStore((s) => s.runAutomation)
  const providers = settings?.providers ?? []

  const [draft, setDraft] = useState<Automation>(automation)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [previewNow, setPreviewNow] = useState(Date.now)
  const timeZones = useMemo(automationTimeZones, [tx])
  const deviceZone = deviceAutomationTimeZone()
  const legacyLocal = automation.scheduleKind === 'daily' && automation.timeZone === undefined && draft.timeZone === undefined
  useEffect(() => {
    if (draft.scheduleKind !== 'daily') return
    const timer = window.setInterval(() => setPreviewNow(Date.now()), 30_000)
    return () => window.clearInterval(timer)
  }, [draft.scheduleKind])
  let timeZoneError: string | null = null
  let nextPreview: string | null = null
  if (draft.scheduleKind === 'daily' && !legacyLocal) {
    try {
      const zone = normalizeAutomationTimeZone(draft.timeZone)
      nextPreview = dailyAutomationPreview(draft.atHour, draft.atMinute, previewNow, zone)
    } catch (error) { timeZoneError = error instanceof Error ? error.message : tx("Choose a valid IANA timezone.") }
  }
  const patch = (p: Partial<Automation>) => setDraft((d) => ({ ...d, ...p }))
  // Bind the model <select> to a real rendered option. See automation-model-select.ts for
  // why a naive rebuild can desync from the option set and read as "model reverted to default".
  const { provider, modelGroups, selectedGroup, selectedModelValue, orphanModelValue } = resolveModelSelect(providers, draft)
  const selectedAccount = accountLabel(provider, draft.providerAccountId ?? null)
  const selectedModelLabel = provider && draft.model
    ? `${modelDisplayName(provider, draft.model)} · ${selectedGroup?.group.label ?? selectedAccount ?? provider.name}`
    : tx("Choose a model")
  const pickModel = (value: string) => {
    const picked = parseModelChoice(value)
    if (picked) patch(picked)
  }

  // Only patch user-editable fields. The draft is a snapshot taken when the editor
  // opened, so its bookkeeping fields (runs/lastRunAt/lastStatus/nextRunAt) are stale —
  // including them would clobber run history recorded by the scheduler while editing.
  const editablePatch = (): Partial<Automation> => ({
    name: draft.name,
    prompt: draft.prompt,
    scheduleKind: draft.scheduleKind,
    everyMinutes: draft.everyMinutes,
    atHour: draft.atHour,
    atMinute: draft.atMinute,
    ...(draft.timeZone !== undefined ? { timeZone: draft.timeZone } : {}),
    providerId: draft.providerId,
    providerAccountId: draft.providerAccountId,
    model: draft.model,
    mode: draft.mode,
    cwd: draft.cwd,
    deliverTo: draft.deliverTo,
    deliveryTarget: draft.deliveryTarget,
    previewOnly: draft.previewOnly,
    enabled: draft.enabled,
    misfire: draft.misfire,
    requiredRisks: draft.requiredRisks,
    acknowledgedRisks: draft.acknowledgedRisks,
    completionPolicy: draft.completionPolicy,
  })

  // Toggle a risk requirement; requiring a risk auto-acknowledges it (the user set it
  // knowingly), and un-requiring drops it from both sets.
  const toggleRisk = (id: string) => {
    const required = new Set(draft.requiredRisks ?? [])
    const acked = new Set(draft.acknowledgedRisks ?? [])
    if (required.has(id)) { required.delete(id); acked.delete(id) }
    else { required.add(id); acked.add(id) }
    patch({ requiredRisks: [...required], acknowledgedRisks: [...acked] })
  }
  const toggleAck = (id: string) => {
    const acked = new Set(draft.acknowledgedRisks ?? [])
    acked.has(id) ? acked.delete(id) : acked.add(id)
    patch({ acknowledgedRisks: [...acked] })
  }
  const completionType = draft.completionPolicy?.type ?? 'none'
  const bumpCompletion = (next: Automation['completionPolicy']) =>
    patch({ completionPolicy: next ? { ...next, version: (draft.completionPolicy?.version ?? 0) + 1 } as Automation['completionPolicy'] : undefined })
  const missingAcks = (draft.requiredRisks ?? []).filter((r) => !(draft.acknowledgedRisks ?? []).includes(r))

  const commit = async (run: boolean) => {
    if (saving || timeZoneError) return
    setSaving(true); setSaveError(null)
    try {
      await updateAutomation(automation.id, editablePatch())
      if (run) await runAutomation(automation.id)
      else onClose()
    } catch (error) { setSaveError(error instanceof Error ? error.message : tx("Could not save this automation.")) }
    finally { setSaving(false) }
  }
  const save = () => commit(false)
  const runNow = () => commit(true)

  const time = `${pad(draft.atHour)}:${pad(draft.atMinute)}`

  return (
    <div className="scroll-thin h-full overflow-y-auto">
      <div className="mx-auto w-full max-w-[680px] px-7 py-8">
        <button onClick={onClose} className="mb-3 flex items-center gap-1.5 text-[13px] text-ink-3 transition-colors hover:text-ink"><Icon name="back" size={14} />{tx("Automations")}</button>

        <Field label={tx("Name")}>
          <Input value={draft.name} onChange={(e) => patch({ name: e.target.value })} placeholder={tx("Daily news brief")} />
        </Field>

        {draft.mode === 'digest' ? (
          <div className="mt-4 rounded-lg border border-line bg-card px-3 py-2.5 text-[13px] leading-relaxed text-ink-3">
            {tx("A digest needs no instruction and no model — it composes a briefing from your recent activity (missions needing you, finished jobs, room updates, new messages) and delivers it on schedule.")}</div>
        ) : (
          <div className="mt-4">
            <Field label={tx("Instruction")} hint={draft.mode === 'agent' ? tx("runs as a full agent (tools + files)") : tx("one completion")}>
              <textarea
                value={draft.prompt}
                onChange={(e) => patch({ prompt: e.target.value })}
                rows={4}
                placeholder={tx("What should this automation do each time it runs?")}
                className="scroll-thin w-full resize-none rounded-lg border border-line bg-card px-3 py-2.5 text-[13px] leading-relaxed text-ink outline-none transition-colors placeholder:text-ink-3 focus:border-accent focus:ring-2 focus:ring-accent/15"
              />
            </Field>
          </div>
        )}

        {/* schedule */}
        <div className="mt-5">
          <div className="mb-1.5 text-[13px] font-medium text-ink-2">{tx("Schedule")}</div>
          <Segmented
            value={draft.scheduleKind}
            onChange={(v) => patch({ scheduleKind: v, ...(v === 'daily' && draft.scheduleKind !== 'daily' && draft.timeZone === undefined ? { timeZone: deviceZone ?? '' } : {}) })}
            options={[{ value: 'manual', label: tx("Manual"), icon: 'play' }, { value: 'interval', label: tx("Interval"), icon: 'clock' }, { value: 'daily', label: tx("Daily"), icon: 'sun' }]}
          />
          {draft.scheduleKind === 'interval' && (
            <div className="mt-2.5">
              <Select value={String(draft.everyMinutes)} onChange={(v) => patch({ everyMinutes: Number(v) })} options={INTERVALS.map((i) => ({ value: String(i.value), label: i.label }))} />
            </div>
          )}
          {draft.scheduleKind === 'daily' && (
            <div className="mt-2.5 space-y-3 text-[13px] text-ink-3">
              <div className="flex flex-wrap items-center gap-2">
              <span>{tx("Runs every day at")}</span>
              <input
                type="time"
                value={time}
                onChange={(e) => { const [h, m] = e.target.value.split(':').map(Number); patch({ atHour: h || 0, atMinute: m || 0 }) }}
                className="no-drag h-9 rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none focus:border-accent"
              />
              </div>
              <Field label={tx("Timezone")} hint={legacyLocal ? tx("Existing host-local schedule") : tx("Pinned IANA timezone")}>
                <Input aria-label={tx("Automation timezone")} aria-invalid={!!timeZoneError} list="automation-timezones" value={draft.timeZone ?? ''} onChange={(event) => patch({ timeZone: event.target.value })} placeholder={legacyLocal ? tx("Legacy host-local timezone") : 'Europe/London'} autoCapitalize="off" autoCorrect="off" spellCheck={false} />
                <datalist id="automation-timezones">{timeZones.map((zone) => <option key={zone} value={zone} />)}</datalist>
              </Field>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="default" disabled={!deviceZone} onClick={() => patch({ timeZone: deviceZone })}>{tx("Use this device's timezone")}{deviceZone ? ` · ${deviceZone}` : ''}</Button>
                {automation.scheduleKind === 'daily' && automation.timeZone === undefined && draft.timeZone !== undefined && <Button size="sm" variant="ghost" onClick={() => patch({ timeZone: undefined })}>{tx("Keep legacy host-local time")}</Button>}
              </div>
              {legacyLocal ? <p className="text-[12px] leading-relaxed text-ink-3">{tx("This existing schedule uses the scheduler host's local timezone and follows that host when its timezone changes; choose a timezone to pin future runs.")}</p> : <p className="text-[12px] leading-relaxed text-ink-3">{tx("The selected timezone stays fixed when this device or the scheduler host changes timezone; a clock time skipped by daylight saving does not run that date, and a repeated clock time runs once at its first occurrence.")}</p>}
              {timeZoneError && <p role="alert" className="text-[12px] text-danger">{timeZoneError}</p>}
              {nextPreview && <p role="status" className="text-[12px] text-ink-2">{tx("Next scheduled time: ")}{nextPreview} · {draft.timeZone}</p>}
              <p className="text-[12px] text-ink-4">{tx("Runs on the scheduler host while TensorFold is open; a missed daily time runs once when it is available again.")}</p>
            </div>
          )}
        </div>

        {/* model + mode */}
        {draft.mode !== 'digest' && (
        <div className="mt-5">
          <Field label={tx("Automation model")} hint={selectedModelLabel}>
            <select value={selectedModelValue} onChange={(e) => pickModel(e.target.value)} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none transition-colors focus:border-accent">
              {!selectedModelValue && <option value="">{tx("Choose model")}</option>}
              {orphanModelValue && (
                <option value={orphanModelValue}>{modelDisplayName(provider, draft.model)} {tx("(pinned)")}</option>
              )}
              {modelGroups.map(({ provider: p, group }) => (
                <optgroup key={group.id} label={group.detail ? `${group.label} · ${group.detail}` : group.label}>
                  {group.models.map((model) => (
                    <option key={`${group.id}:${model}`} value={modelChoiceValue(p.id, group.accountId, model)}>
                      {modelDisplayName(p, model)}
                    </option>
                  ))}
                </optgroup>
              ))}
            </select>
          </Field>
        </div>
        )}

        <div className="mt-5 flex flex-wrap items-end justify-between gap-4">
          <div>
            <div className="mb-1.5 text-[13px] font-medium text-ink-2">{tx("Run as")}</div>
            <Segmented value={draft.mode} onChange={(v) => patch({ mode: v })} options={[{ value: 'chat', label: tx("Chat"), icon: 'message' }, { value: 'agent', label: tx("Agent"), icon: 'spark' }, { value: 'digest', label: tx("Digest"), icon: 'sun' }]} />
          </div>
          <div>
            <div className="mb-1.5 text-[13px] font-medium text-ink-2">{tx("Deliver to")}</div>
            <Segmented value={draft.deliverTo} onChange={(v) => patch({ deliverTo: v, deliveryTarget: v === 'inbox' ? undefined : { connectorId: v, accountId: null, recipient: '' } })} options={[{ value: 'inbox', label: tx("Inbox"), icon: 'message' }, { value: 'telegram', label: tx("Telegram"), icon: 'send' }, { value: 'email', label: tx("Email"), icon: 'mail' }, { value: 'whatsapp', label: tx("WhatsApp"), icon: 'at' }]} />
          </div>
        </div>

        {draft.deliverTo !== 'inbox' && <div className="mt-4 space-y-3 rounded-xl border border-line bg-card-2 p-4">
          <p className="text-[13px] text-ink-2">{tx("Each result becomes a draft for your approval in Assistant; scheduling never grants permission to send.")}</p>
          <Field label={tx("Recipient")}><Input value={draft.deliveryTarget?.recipient ?? ''} onChange={(event) => patch({ deliveryTarget: { connectorId: draft.deliverTo as 'whatsapp' | 'telegram' | 'email', accountId: draft.deliveryTarget?.accountId ?? null, recipient: event.target.value, ...(draft.deliveryTarget?.subject ? { subject: draft.deliveryTarget.subject } : {}) } })} placeholder={draft.deliverTo === 'email' ? "person@example.com" : draft.deliverTo === 'telegram' ? tx("Allowed chat ID") : tx("Selected phone or chat JID")} spellCheck={false} /></Field>
          {draft.deliverTo !== 'telegram' && <Field label={tx("Sending account")}><select value={draft.deliveryTarget?.accountId ?? ''} onChange={(event) => patch({ deliveryTarget: { connectorId: draft.deliverTo as 'whatsapp' | 'email', accountId: event.target.value || null, recipient: draft.deliveryTarget?.recipient ?? '', ...(draft.deliveryTarget?.subject ? { subject: draft.deliveryTarget.subject } : {}) } })} className="h-9 w-full rounded-lg border border-line bg-card px-2 text-[13px] text-ink"><option value="">{tx("Choose an account")}</option>{(draft.deliverTo === 'whatsapp' ? settings?.whatsapp.sources : settings?.email.sources)?.filter((source) => source.enabled).map((source) => <option key={source.id} value={source.id}>{source.label || source.id}</option>)}</select></Field>}
          <Toggle label={tx("Keep preview results in the TensorFold inbox")} checked={draft.previewOnly === true} onChange={(previewOnly) => patch({ previewOnly })} />
        </div>}

        {/* Advanced (VST-038): misfire, risk acknowledgement, completion policy. */}
        {draft.scheduleKind === 'interval' && (
          <div className="mt-5">
            <div className="mb-1.5 text-[13px] font-medium text-ink-2">{tx("If a fire time was missed")}</div>
            <Segmented
              value={draft.misfire ?? 'coalesce'}
              onChange={(v) => patch({ misfire: v })}
              options={[{ value: 'coalesce', label: tx("Run once"), icon: 'clock' }, { value: 'run_each', label: tx("Catch up each"), icon: 'refresh' }]}
            />
            <div className="mt-1 text-[12px] leading-snug text-ink-4">{tx("Coalesce: one run after the host wakes. Catch up: replay each missed fire (capped).")}</div>
          </div>
        )}

        <div className="mt-5">
          <div className="mb-1.5 text-[13px] font-medium text-ink-2">{tx("Risks to acknowledge")}</div>
          <div className="space-y-1.5">
            {RISK_CATALOG.map((risk) => {
              const required = (draft.requiredRisks ?? []).includes(risk.id)
              const acked = (draft.acknowledgedRisks ?? []).includes(risk.id)
              return (
                <div key={risk.id} className={clsx('flex items-start gap-2.5 rounded-lg border px-3 py-2', required && !acked ? 'border-warn/40 bg-warn/5' : 'border-line bg-card')}>
                  <button onClick={() => toggleRisk(risk.id)} className={clsx('mt-0.5 flex h-4 w-4 shrink-0 items-center justify-center rounded border', required ? 'border-accent bg-accent text-accent-ink' : 'border-line text-transparent')} title={required ? tx("Required for this automation") : tx("Mark as required")}>
                    <Icon name="check" size={11} />
                  </button>
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-medium text-ink">{risk.label}</div>
                    <div className="text-[12px] leading-snug text-ink-3">{risk.desc}</div>
                  </div>
                  {required && (
                    <button onClick={() => toggleAck(risk.id)} className={clsx('shrink-0 rounded-md px-2 py-1 text-[12px] font-medium transition-colors', acked ? 'bg-positive/10 text-positive' : 'border border-warn/40 text-warn hover:bg-warn/5')}>
                      {acked ? tx("Acknowledged") : tx("Acknowledge")}
                    </button>
                  )}
                </div>
              )
            })}
          </div>
          {missingAcks.length > 0 && (
            <div className="mt-1.5 text-[12px] text-warn">{tx("This automation won’t run until every required risk is acknowledged (")}{missingAcks.length} {tx("pending).")}</div>
          )}
        </div>

        {draft.mode === 'agent' && (
          <div className="mt-5">
            <div className="mb-1.5 text-[13px] font-medium text-ink-2">{tx("When is it done?")}</div>
            <Segmented
              value={completionType}
              onChange={(v) => {
                if (v === 'none') bumpCompletion(undefined)
                else if (v === 'max_iterations') bumpCompletion({ type: 'max_iterations', version: 0, max: draft.completionPolicy?.type === 'max_iterations' ? draft.completionPolicy.max : 3 })
                else if (v === 'ai_evaluated') bumpCompletion({ type: 'ai_evaluated', version: 0, prompt: draft.completionPolicy?.type === 'ai_evaluated' ? draft.completionPolicy.prompt : 'Has the automation fully achieved its objective this run?' })
              }}
              options={[{ value: 'none', label: tx("One pass"), icon: 'play' }, { value: 'max_iterations', label: tx("N iterations"), icon: 'refresh' }, { value: 'ai_evaluated', label: tx("AI judges"), icon: 'spark' }]}
            />
            {completionType === 'max_iterations' && draft.completionPolicy?.type === 'max_iterations' && (
              <div className="mt-2.5">
                <Select value={String(draft.completionPolicy.max)} onChange={(v) => bumpCompletion({ type: 'max_iterations', version: 0, max: Number(v) })} options={[2, 3, 5, 8].map((n) => ({ value: String(n), label: `${n} iterations` }))} />
              </div>
            )}
            {completionType === 'ai_evaluated' && draft.completionPolicy?.type === 'ai_evaluated' && (
              <div className="mt-2.5">
                <textarea
                  value={draft.completionPolicy.prompt}
                  onChange={(e) => bumpCompletion({ type: 'ai_evaluated', version: 0, prompt: e.target.value })}
                  rows={2}
                  placeholder={tx("How should the model judge whether the objective is met?")}
                  className="scroll-thin w-full resize-none rounded-lg border border-line bg-card px-3 py-2 text-[13px] leading-relaxed text-ink outline-none placeholder:text-ink-3 focus:border-accent"
                />
                <div className="mt-1 text-[12px] leading-snug text-ink-4">{tx("After each run the model reads the output and answers complete / continue / needs_review; editing this bumps the policy version, forcing review of any in-flight run.")}</div>
              </div>
            )}
          </div>
        )}

        <div className="mt-6 flex items-center justify-between border-t border-line pt-4">
          <label className="flex items-center gap-2.5 text-[13px] text-ink-2">
            <Toggle label={tx("Automation enabled")} checked={draft.enabled} onChange={(v) => patch({ enabled: v })} />
            {draft.enabled ? tx("Enabled") : tx("Paused")}
          </label>
          <div className="flex items-center gap-2">
            <Button variant="default" icon="play" disabled={saving || !!timeZoneError} onClick={runNow}>{tx("Run now")}</Button>
            <Button variant="primary" icon="check" disabled={saving || !!timeZoneError} onClick={save}>{saving ? tx("Saving…") : tx("Save")}</Button>
          </div>
        </div>
        {saveError && <p role="alert" className="mt-3 text-[13px] text-danger">{saveError}</p>}
      </div>
    </div>
  )
}

// small native select styled like the rest of the app
function Select({ value, onChange, options }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[] }) {
  const tx = useAuthoredText()
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)} className="no-drag h-9 w-full cursor-pointer rounded-lg border border-line bg-card px-2.5 text-[13px] text-ink outline-none transition-colors focus:border-accent">
      {options.length === 0 && <option value="">—</option>}
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  )
}

// ------------------------------------------------------------------ templates
const TEMPLATES: { name: string; icon: IconName; desc: string; make: () => Partial<Automation> }[] = [
  { name: 'Daily AI News Brief', icon: 'globe', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Top 5 AI stories every morning") }, make: () => ({ name: 'Daily AI News Brief', prompt: 'Search the web for the most important AI developments in the last 24 hours and summarize the top 5 with one-line takeaways and links.', scheduleKind: 'daily', atHour: 9, atMinute: 0, mode: 'agent', deliverTo: 'inbox' }) },
  { name: 'Inbox triage', icon: 'mail', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Label + draft replies every 30 min") }, make: () => ({ name: 'Inbox triage', prompt: 'Scan new email, label anything urgent, and draft concise replies for messages that need one. Summarize what you did.', scheduleKind: 'interval', everyMinutes: 30, mode: 'agent', deliverTo: 'inbox' }) },
  { name: 'Standup digest', icon: 'message', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Yesterday’s commits + open PRs") }, make: () => ({ name: 'Standup digest', prompt: "Summarize yesterday's commits and open PRs into a short standup digest.", scheduleKind: 'daily', atHour: 8, atMinute: 30, mode: 'chat', deliverTo: 'inbox' }) },
  { name: 'Daily brief', icon: 'sun', get desc() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "What needs you, every morning (no model)") }, make: () => ({ name: 'Daily brief', prompt: '', scheduleKind: 'daily', atHour: 8, atMinute: 0, mode: 'digest', deliverTo: 'inbox' }) },
]
