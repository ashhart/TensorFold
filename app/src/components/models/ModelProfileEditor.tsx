import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState, type MutableRefObject } from 'react'
import { createPortal } from 'react-dom'
import { activeEngineModelSettings, defaultEngineModelProfile, isGreedyOnlyEngineModel, validateEngineModelProfileSet } from '../../../shared/model-profiles'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { refreshEngine } from '../../lib/engine'
import type { EngineInstalledModel, EngineModelGenerationSettings, EngineModelProfile } from '../../lib/types'
import { Button, Input } from '../ui'

function errorText(error: unknown): string { return String((error as Error)?.message ?? error).replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '') }
function initialProfiles(model: EngineInstalledModel) {
  const settings = useStore.getState().settings
  const saved = settings?.engineModelProfiles?.find((entry) => entry.model === model.repo)
  if (saved) return structuredClone(saved)
  const profile = { ...defaultEngineModelProfile(model.contextTokens, isGreedyOnlyEngineModel(model) ? 0 : settings?.temperature), id: crypto.randomUUID(), name: 'Balanced' }
  return { model: model.repo, revision: 0, profiles: [profile], activeProfileId: profile.id }
}

export interface ModelProfileEditorControls {
  hasChanges: () => boolean
  isBusy: () => boolean
  save: () => Promise<boolean>
  discard: () => void
}
export interface ModelProfileEditorState { dirty: boolean; busy: boolean }

/** Changes remain a local draft until the host durably saves and safely applies this exact model. */
export function ModelProfileEditor({ model, blocked = false, onClose, embedded = false, controllerRef, onStateChange }: {
  model: EngineInstalledModel; blocked?: boolean; onClose: () => void; embedded?: boolean
  controllerRef?: MutableRefObject<ModelProfileEditorControls | null>
  onStateChange?: (state: ModelProfileEditorState) => void
}) {
  const tx = useAuthoredText()
  const [draft, setDraft] = useState(() => initialProfiles(model))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  const [confirmClose, setConfirmClose] = useState(false)
  const baseline = useRef(structuredClone(draft))
  const dialog = useRef<HTMLDivElement>(null)
  const inFlight = useRef(false)
  const previousFocus = useRef<HTMLElement | null>(null)
  const closePrompt = useRef<HTMLDivElement>(null)
  const editFields = useRef<HTMLDivElement>(null)
  const returnToFields = useRef(false)
  const keepEditing = () => { returnToFields.current = true; setConfirmClose(false) }
  useEffect(() => {
    if (confirmClose) {
      closePrompt.current?.scrollIntoView({ block: 'nearest' })
      closePrompt.current?.querySelector<HTMLElement>('button[data-keep-editing="true"]:not(:disabled)')?.focus({ preventScroll: true })
    } else if (returnToFields.current) {
      returnToFields.current = false
      const field = editFields.current?.querySelector<HTMLElement>('#model-profile-contextTokens:not(:disabled)') ?? editFields.current?.querySelector<HTMLElement>('input:not(:disabled),select:not(:disabled),button:not(:disabled)')
      field?.scrollIntoView({ block: 'nearest' })
      field?.focus({ preventScroll: true })
    }
  }, [confirmClose])
  const greedy = isGreedyOnlyEngineModel(model)
  const selected = draft.profiles.find((entry) => entry.id === draft.activeProfileId)
  const disabled = busy
  const liveSettings = useStore((s) => s.settings)
  const original = liveSettings?.engineModelProfiles?.find((entry) => entry.model === model.repo)
  const stale = (original?.revision ?? 0) !== draft.revision
  const dirty = JSON.stringify(draft) !== JSON.stringify(baseline.current)
  const discard = () => { if (!inFlight.current) { setDraft(structuredClone(baseline.current)); setError(null); setSaved(false) } }
  const close = () => { if (inFlight.current) return; if (!embedded && dirty) setConfirmClose(true); else onClose() }
  useEffect(() => { onStateChange?.({ dirty, busy }) }, [dirty, busy, onStateChange])

  useEffect(() => {
    if (embedded) return
    previousFocus.current = document.activeElement as HTMLElement | null
    dialog.current?.querySelector<HTMLElement>('button,select,input')?.focus()
    return () => { if (previousFocus.current?.isConnected) previousFocus.current.focus() }
  }, [embedded])
  useEffect(() => {
    if (embedded) return
    const onKey = (event: KeyboardEvent) => {
      const element = dialog.current
      if (!element) return
      if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); if (confirmClose) keepEditing(); else close(); return }
      if (event.key !== 'Tab') return
      const targets = [...element.querySelectorAll<HTMLElement>('button:not(:disabled),select:not(:disabled),input:not(:disabled),summary,[tabindex="0"]')].filter((target) => target.getClientRects().length > 0)
      if (!targets.length) { event.preventDefault(); element.focus(); return }
      const first = targets[0], last = targets.at(-1)!
      if (event.shiftKey && (document.activeElement === first || !element.contains(document.activeElement))) { event.preventDefault(); last.focus() }
      else if (!event.shiftKey && (document.activeElement === last || !element.contains(document.activeElement))) { event.preventDefault(); first.focus() }
    }
    const focus = (event: FocusEvent) => { if (!dialog.current?.contains(event.target as Node)) dialog.current?.focus() }
    document.addEventListener('keydown', onKey, true)
    document.addEventListener('focusin', focus, true)
    return () => { document.removeEventListener('keydown', onKey, true); document.removeEventListener('focusin', focus, true) }
  }, [onClose, embedded, dirty, confirmClose])

  const patch = (change: Partial<EngineModelProfile>) => {
    if (inFlight.current || !selected) return
    setDraft((current) => ({ ...current, profiles: current.profiles.map((entry) => entry.id === selected.id ? { ...entry, ...change } : entry) })); setSaved(false); setError(null)
  }
  const add = (name: string, values?: EngineModelGenerationSettings) => {
    if (inFlight.current) return
    let nextName = name, n = 2
    while (draft.profiles.some((entry) => entry.name.toLowerCase() === nextName.toLowerCase())) nextName = `${name} ${n++}`
    const profile = { ...(values ?? selected ?? defaultEngineModelProfile(model.contextTokens, greedy ? 0 : liveSettings?.temperature)), id: crypto.randomUUID(), name: nextName }
    setDraft((current) => ({ ...current, profiles: [...current.profiles, profile], activeProfileId: profile.id })); setSaved(false); setError(null)
  }
  const preset = (context: number, name: string) => {
    const values = defaultEngineModelProfile(model.contextTokens, greedy ? 0 : liveSettings?.temperature)
    values.contextTokens = Math.min(context, model.contextTokens ?? context)
    values.maxOutputTokens = Math.min(2048, Math.floor(values.contextTokens / 4))
    add(name, values)
  }
  const save = async () => {
    if (inFlight.current || blocked || stale) return false
    setError(null)
    try { validateEngineModelProfileSet(draft) } catch (problem) { setError(errorText(problem)); return false }
    inFlight.current = true; setBusy(true)
    try {
      const next = await api.engineSaveModelProfiles({ model: model.repo, profiles: draft.profiles, activeProfileId: draft.activeProfileId, expectedRevision: draft.revision })
      useStore.setState({ settings: next })
      const accepted = next.engineModelProfiles?.find((entry) => entry.model === model.repo)
      if (accepted) { baseline.current = structuredClone(accepted); setDraft(structuredClone(accepted)) }
      await refreshEngine()
      setSaved(true)
      return true
    } catch (problem) { setError(errorText(problem)); return false }
    finally { inFlight.current = false; setBusy(false) }
  }
  useEffect(() => {
    if (!controllerRef) return
    const controller = { hasChanges: () => dirty, isBusy: () => inFlight.current, save, discard }
    controllerRef.current = controller
    return () => { if (controllerRef.current === controller) controllerRef.current = null }
  })

  const number = (field: keyof EngineModelGenerationSettings, label: string, help: string, min: number, max: number, step: number, unavailable = false) => <label className="grid min-w-0 gap-1.5 text-[13px] text-ink" htmlFor={`model-profile-${field}`}><span className="font-medium">{label}</span><Input id={`model-profile-${field}`} type="number" min={min} max={max} step={step} value={typeof selected?.[field] === 'number' && Number.isFinite(selected[field]) ? selected[field] as number : ''} disabled={disabled || unavailable} onChange={(event) => patch({ [field]: event.currentTarget.value === '' ? NaN : Number(event.currentTarget.value) })} /><span className="text-[12px] leading-relaxed text-ink-2">{help}</span></label>

  const content = <div className="fixed inset-0 z-[65] flex items-center justify-center bg-black/40 p-4 backdrop-blur-sm" onMouseDown={(event) => { if (event.target === event.currentTarget) close() }}>
    <div ref={dialog} tabIndex={embedded ? undefined : -1} role={embedded ? undefined : "dialog"} aria-modal={embedded ? undefined : true} aria-labelledby={embedded ? undefined : "model-profile-title"} aria-busy={busy} className={embedded ? "min-w-0 text-ink" : "flex max-h-[calc(100dvh-32px)] w-full max-w-[600px] flex-col overflow-hidden rounded-2xl border border-line bg-card text-ink shadow-pop outline-none"}>
      {!embedded && <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-4"><div className="min-w-0"><h2 id="model-profile-title" className="text-[16px] font-semibold">{tx("Model settings")}</h2><p className="mt-1 break-all text-[12px] text-ink-2">{model.title ?? model.repo}</p></div><Button size="sm" variant="ghost" disabled={busy} onClick={close}>{tx("Close")}</Button></div>}
      <div ref={editFields} className={embedded ? "space-y-5" : "scroll-thin min-h-0 space-y-5 overflow-y-auto px-5 py-4"}>
        {confirmClose && <div ref={closePrompt} role="alert" className="space-y-3 rounded-xl border border-warn/30 bg-warn/10 p-3"><p className="text-[13px] text-ink">{tx("Save your model settings before closing?")}</p><div className="flex flex-wrap gap-2"><Button size="sm" disabled={busy || blocked || stale} onClick={() => { void save().then((ok) => { if (ok) onClose() }) }}>{tx("Save & close")}</Button><Button size="sm" variant="danger" disabled={busy} onClick={() => { if (!inFlight.current) onClose() }}>{tx("Discard changes")}</Button><Button size="sm" variant="ghost" disabled={busy} data-keep-editing="true" onClick={keepEditing}>{tx("Keep editing")}</Button></div></div>}
        <p className="text-[13px] leading-relaxed text-ink-2">{tx("Profiles belong to this model and apply to its local chats and API defaults. Saving a loaded model restarts its server when idle and disconnects API clients.")}</p>
        {liveSettings?.engineModelProfilesProblem && <p role="alert" className="text-[13px] text-warn">{liveSettings.engineModelProfilesProblem}</p>}
        <div className="grid gap-2"><label htmlFor="model-profile-select" className="text-[13px] font-medium">{tx("Profile")}</label><select id="model-profile-select" value={draft.activeProfileId ?? ''} disabled={disabled} onChange={(event) => { setDraft((current) => ({ ...current, activeProfileId: event.target.value || null })); setSaved(false); setError(null) }} className="min-h-9 w-full rounded-lg border border-line bg-card px-3 text-[13px] text-ink focus-visible:outline-2 focus-visible:outline-accent"><option value="">{tx("Engine defaults")}</option>{draft.profiles.map((profile) => <option key={profile.id} value={profile.id}>{profile.name}</option>)}</select><div className="flex flex-wrap gap-2"><Button size="xs" disabled={disabled || draft.profiles.length >= 32} onClick={() => preset(8192, 'Small context')}>{tx("Add small context")}</Button><Button size="xs" disabled={disabled || draft.profiles.length >= 32} onClick={() => preset(32768, 'Large context')}>{tx("Add large context")}</Button><Button size="xs" variant="ghost" disabled={disabled || draft.profiles.length >= 32} onClick={() => add('Custom')}>{selected ? tx("Duplicate profile") : tx("Create profile")}</Button></div></div>
        {selected && !original && <p className="text-[12px] text-ink-2">{tx("This new profile takes effect when you choose Save & apply.")}</p>}
        {selected ? <><div className="flex items-end gap-2"><label htmlFor="model-profile-name" className="grid min-w-0 flex-1 gap-1.5 text-[13px] font-medium">{tx("Profile name")}<Input id="model-profile-name" value={selected.name} maxLength={60} disabled={disabled} onChange={(event) => patch({ name: event.currentTarget.value })} /></label><Button size="sm" variant="danger" disabled={disabled} onClick={() => { setDraft((current) => ({ ...current, profiles: current.profiles.filter((entry) => entry.id !== selected.id), activeProfileId: null })); setSaved(false); setError(null) }}>{tx("Delete")}</Button></div>
          <div className="grid gap-4 sm:grid-cols-2">{number('contextTokens', tx("Context window"), model.contextTokens ? tx("Prompt and reply together. Model limit: {count} tokens. Larger windows can use more memory.", { count: model.contextTokens }) : tx("Prompt and reply together. The engine verifies this when loading. Larger windows can use more memory."), 1024, Math.min(model.contextTokens ?? 1_048_576, 1_048_576), 1)}{number('maxOutputTokens', tx("Maximum reply"), tx("Reply limit for each request, within this context window."), 1, Number.isFinite(selected.contextTokens) ? Math.max(1, selected.contextTokens - 1) : 1_048_575, 1)}</div>
          {number('temperature', tx("Temperature"), greedy ? tx("This model supports greedy decoding only in this engine, so temperature stays at 0.") : tx("Lower values make replies more predictable; higher values vary more."), 0, 2, 0.05, greedy)}
          <details className="rounded-xl border border-line px-3 py-3"><summary className="cursor-pointer text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-accent">{tx("Advanced sampling and acceleration")}</summary><div className="mt-4 grid gap-4 sm:grid-cols-2">{number('topP', tx("Top P"), tx("Nucleus sampling. 1 keeps the full distribution."), 0.0001, 1, 0.01, greedy)}{number('topK', tx("Top K"), tx("Consider the most likely tokens. 0 disables this filter."), 0, 1000, 1, greedy)}{number('minP', tx("Min P"), tx("Minimum probability relative to the most likely token. 0 disables it."), 0, 1, 0.01, greedy)}<label htmlFor="model-profile-drafting" className="grid gap-1.5 self-start text-[13px] font-medium">{tx("Speculative drafting")}<select id="model-profile-drafting" value={selected.speculativeDecoding} disabled={disabled} onChange={(event) => patch({ speculativeDecoding: event.target.value as 'auto' | 'off' })} className="min-h-9 rounded-lg border border-line bg-card px-3 text-[13px] text-ink"><option value="auto">{tx("Automatic, when supported")}</option><option value="off">{tx("Off")}</option></select><span className="text-[12px] font-normal leading-relaxed text-ink-2">{tx("Uses compatible model acceleration when available. This engine does not expose custom MTP depth or confidence settings.")}</span></label></div></details>
        </> : <p className="rounded-xl bg-card-2 px-3 py-4 text-[13px] text-ink-2">{tx("Engine defaults will be restored for this model. Your named profiles remain available.")}</p>}
        {blocked && <p role="status" className="text-[13px] text-warn">{tx("Finish active local work before applying a profile. Your edits stay here.")}</p>}
        {stale && <p role="alert" className="text-[13px] text-warn">{tx("Profiles changed in another window. Close and reopen this editor before saving.")}</p>}
        {error && <p role="alert" className="break-words text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
        {saved && <p role="status" className="text-[13px] text-positive">{activeEngineModelSettings(liveSettings ?? {}, model.repo) ? tx("Profile saved and applied.") : tx("Engine defaults saved and applied.")}</p>}
      </div>
      <div className={embedded ? "mt-5 flex flex-wrap items-center justify-end gap-2 border-t border-line pt-4" : "flex items-center justify-end gap-2 border-t border-line px-5 py-3"}><Button size="sm" variant="ghost" disabled={busy} onClick={() => { if (!inFlight.current) { if (embedded) discard(); else onClose() } }}>{embedded ? tx("Cancel changes") : tx("Cancel")}</Button><Button size="sm" variant="default" disabled={disabled || blocked || stale} onClick={() => void save()}>{busy ? tx("Saving and applying…") : tx("Save & apply")}</Button></div>
    </div>
  </div>
  return embedded ? content.props.children : createPortal(content, document.body)
}
