import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { Attachment } from '../../lib/types'
import type { QuickAskApi, QuickAskWindowChoice } from '../../../shared/quick-ask'
import { BrandMark } from '../BrandMark'
import { Icon } from '../../lib/icons'
import { authoredText } from '../../lib/preferences/language-packs'
import { normalizeUiPreferences } from '../../lib/preferences/model'
import { applyUiPreferences } from '../../lib/preferences/apply'
import { applyTheme } from '../../lib/theme'
import { filesToAttachments } from '../../lib/attachments'
import { AskPreferences } from './AskPreferences'
import { AskDialog } from './AskDialog'
import { AskAnswer } from './AskAnswer'
import { askError, useQuickAsk } from './useQuickAsk'
import { useAskVoice } from './useAskVoice'
import { useAskPanelSize } from './useAskPanelSize'
import { AskOptions } from './AskOptions'
import { AskContextChips, AskContextPreview, AskWindowChoices, WindowTargetIcon, askContextGroups, type AskContextGroup } from './AskContext'
import './quick-ask.css'

export default function QuickAskApp({ api = window.tensorfoldQuickAsk! }: { api?: QuickAskApi }) {
  useLayoutEffect(() => { document.body.classList.add('ready') }, [])
  const state = useQuickAsk(api)
  const { snapshot, draft, edit, busy, saving, action, error, setError } = state
  const locale = snapshot?.appearance.uiPreferences?.locale
  const tx = useCallback((source: string, params?: Record<string, string | number | null | undefined>) => authoredText(locale, source, params), [locale])
  const [settings, setSettings] = useState(false), [consent, setConsent] = useState(false)
  const [options, setOptions] = useState(false), [expanded, setExpanded] = useState(false)
  const [preview, setPreview] = useState<AskContextGroup | null>(null)
  const [choices, setChoices] = useState<QuickAskWindowChoice[] | null>(null)
  const [displays, setDisplays] = useState<Array<{ id: string; label: string }> | null>(null)
  const panel = useRef<HTMLElement>(null), question = useRef<HTMLTextAreaElement>(null), file = useRef<HTMLInputElement>(null)
  const bottom = useRef<HTMLDivElement>(null), previousTurn = useRef<string | null>(null)
  const report = useCallback((value: unknown) => setError(askError(value)), [setError])
  const record = useCallback((active: boolean) => api.recordShortcut(active), [api])
  const voice = useAskVoice(api, snapshot, text => edit({ text: `${draft?.text ?? ''}${draft?.text ? ' ' : ''}${text}` }), report)
  const session = snapshot?.session
  const running = session?.turns.some(turn => turn.status === 'running') ?? false
  const blocked = busy || running || voice.listening
  const command = /^\/([^\s]+)/.exec(draft?.text ?? '')?.[1]
  const preset = snapshot?.preferences.presets.find(preset => command ? preset.command.toLowerCase() === command.toLowerCase() : preset.id === draft?.presetId)
  const routeId = session?.route?.id ?? preset?.routeId ?? draft?.routeId
  const route = snapshot?.routes.find(route => route.id === routeId) ?? session?.route
  const modifiedEnter = snapshot?.appearance.uiPreferences?.sendKey === 'modified-enter'
  const groups = askContextGroups(draft?.attachments ?? [])
  useAskPanelSize(api, panel, !!(settings || preview || choices || consent || displays), report)

  useLayoutEffect(() => {
    if (!snapshot) return
    const appearance = snapshot.appearance
    document.documentElement.classList.toggle('dark', appearance.dark)
    applyTheme(appearance.palette, appearance.dark, true)
    applyUiPreferences(normalizeUiPreferences(appearance.uiPreferences))
  }, [snapshot?.appearance])
  useLayoutEffect(() => {
    if (!question.current) return
    question.current.style.height = 'auto'
    question.current.style.height = `${Math.min(96, Math.max(26, question.current.scrollHeight))}px`
  }, [draft?.text])
  useEffect(() => {
    const focus = () => { setExpanded(false); setOptions(false); question.current?.focus() }
    const off = api.onEvent(event => { if (event.kind === 'focus') focus() })
    if (draft) question.current?.focus()
    return off
  }, [api, !!draft])
  useEffect(() => {
    const id = session?.turns.at(-1)?.id ?? null
    if (previousTurn.current && id && previousTurn.current !== id) setExpanded(true)
    previousTurn.current = id
  }, [session?.turns.at(-1)?.id])
  useEffect(() => { if (running && expanded) bottom.current?.scrollIntoView({ block: 'nearest' }) }, [session?.turns.at(-1)?.answer, running, expanded])
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !document.querySelector('dialog[open]')) {
        event.preventDefault(); voice.stopSpeech(); void action(() => api.hide())
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [api, action])

  async function append(items: Attachment[], selectedWindow = false) {
    if (!draft) return
    if (items.some(item => item.status && item.status !== 'ready')) throw new Error(items.find(item => item.error)?.error ?? 'This attachment could not be read.')
    const remaining = selectedWindow ? draft.attachments.filter(item => !/^window-[a-f0-9-]{36}-(?:image|text)$/.test(item.id)) : draft.attachments
    if (remaining.length + items.length > 4) throw new Error('Attach up to four items. Remove one before adding another.')
    if (items.some(item => (item.text?.length ?? 0) > 32_000)) throw new Error('Attach a shorter text selection, up to 32,000 characters.')
    edit({ attachments: [...remaining, ...items] })
  }
  async function ingest(files: FileList | File[]) {
    if (blocked) return
    try { await append(await filesToAttachments(files)) } catch (value) { report(value) }
  }
  async function pickWindow() {
    await action(async () => {
      const selected = await api.pickWindow()
      if (selected.kind === 'attached') await append(selected.attachments, true)
      if (selected.kind === 'choose') setChoices(selected.windows)
    })
    question.current?.focus()
  }
  async function chooseWindow(token: string) {
    setChoices(null)
    await action(async () => append(await api.useWindow(token), true))
    question.current?.focus()
  }
  async function capture(displayId: string) {
    setDisplays(null)
    await action(async () => append([await api.capture(displayId)]))
  }
  async function submit(approved = false) {
    if (blocked || !draft) return
    if (!route) { setError(tx('Choose a model before asking.')); return }
    if (route.destination !== 'device' && !approved) { setConsent(true); return }
    setConsent(false); setOptions(false); setExpanded(true); voice.stopSpeech()
    await action(() => api.send({ revision: state.revision.current, approvedRouteId: approved ? route.id : null }))
  }
  const hide = () => { voice.stopSpeech(); void action(() => api.hide()) }

  return <main ref={panel} className="ask-shell ask-compact bg-canvas text-ink">
    <h1 className="sr-only">{tx('Ask TensorFold')}</h1>
    {expanded && session && <section className="ask-conversation" aria-label={tx('Conversation')}>
      <div className="ask-result-toolbar">
        <button className="ask-text-button" onClick={() => setExpanded(false)}>{tx('Collapse answer')}</button>
        <button className="ask-text-button" disabled={blocked} onClick={() => void action(() => api.handoff())}>{tx('Continue in TensorFold')}<Icon name="arrowRight" size={12} /></button>
      </div>
      {session.turns.slice(-2).map(turn => <AskAnswer key={turn.id} turn={turn} api={api} tx={tx} speak={text => void voice.speak(text)} error={report} />)}
      <div ref={bottom} />
    </section>}
    {!!groups.length && <AskContextChips groups={groups} blocked={blocked} preview={setPreview} tx={tx}
      remove={ids => edit({ attachments: draft!.attachments.filter(item => !ids.includes(item.id)) })} />}
    <div className="ask-prompt-row" onDragOver={event => event.preventDefault()} onDrop={event => { event.preventDefault(); void ingest(event.dataTransfer.files) }}>
      <span className="ask-mark drag" title={tx('Ask TensorFold')}><BrandMark size={25} /></span>
      <textarea ref={question} rows={1} className="ask-question no-drag" aria-label={tx('Question')} placeholder={tx('What would you like help with?')}
        maxLength={32_000} disabled={!draft || blocked || !!session?.handoffThreadId} value={draft?.text ?? ''} onChange={event => edit({ text: event.target.value })}
        onPaste={event => { if (event.clipboardData.files.length) { event.preventDefault(); void ingest(event.clipboardData.files) } }}
        onKeyDown={event => { if (event.key === 'Enter' && !event.nativeEvent.isComposing && !event.shiftKey && (!modifiedEnter || event.metaKey || event.ctrlKey)) { event.preventDefault(); void submit() } }} />
      <button className="ask-icon-button no-drag" title={tx('Choose a window')} aria-label={tx('Choose a window')} disabled={blocked} onClick={() => void pickWindow()}><WindowTargetIcon /></button>
      {running ? <button className="ask-send no-drag" aria-label={tx('Stop')} onClick={() => void action(() => api.cancel())}><Icon name="stop" size={14} /></button>
        : <button className="ask-send no-drag" aria-label={tx('Ask')} title={tx('Ask')} disabled={!draft || blocked || !route || route.state !== 'ready' || !draft.text.trim() && !draft.attachments.length || !!session?.handoffThreadId} onClick={() => void submit()}><Icon name="arrowRight" size={17} /></button>}
      <button className="ask-icon-button no-drag ask-close" title={tx('Hide Ask TensorFold')} aria-label={tx('Hide Ask TensorFold')} onClick={hide}>×</button>
    </div>
    <div className="ask-footer">
      <select className="ask-route" aria-label={tx('Model and destination')} disabled={blocked || !!session?.route || !!preset?.routeId} value={routeId ?? ''} onChange={event => edit({ routeId: event.target.value || null })}>
        <option value="">{tx('Choose a model')}</option>
        {routeId && !snapshot?.routes.some(item => item.id === routeId) && <option value={routeId}>{route?.label ?? tx('Saved model is unavailable')}</option>}
        {snapshot?.routes.map(item => <option key={item.id} value={item.id}>{item.label} · {item.location}{item.state === 'load-required' ? ` · ${tx('Needs loading')}` : ''}</option>)}
      </select>
      {route?.state === 'load-required' && <button className="ask-text-button" disabled={blocked} onClick={() => void action(() => api.loadModel(route.id))}>{tx('Load model')}</button>}
      {!!session?.turns.length && !expanded && <button className="ask-text-button" onClick={() => setExpanded(true)}>{tx(running ? 'Thinking…' : 'View answer')}</button>}
      <span className="flex-1" />
      <span className="sr-only" role="status">{saving ? tx('Saving draft…') : tx('Saved on this device. Context is attached only when you choose it.')}</span>
      <button className={`ask-icon-button ${options ? 'text-accent' : ''}`} aria-label={tx('More options')} aria-expanded={options} onClick={() => setOptions(!options)}><Icon name="ellipsis" size={16} /></button>
    </div>
    {error && <p role="alert" className="ask-error">{error}</p>}
    {options && snapshot && draft && <AskOptions snapshot={snapshot} draft={draft} blocked={blocked} edit={edit} tx={tx}
      attachFiles={() => file.current?.click()} clipboard={() => void action(async () => append(await api.clipboard()))}
      screenshot={() => void action(async () => { const items = await api.displays(); if (items.length === 1) await append([await api.capture(items[0].id)]); else setDisplays(items) })}
      dictate={voice.dictate} listening={voice.listening} settings={() => setSettings(true)}
      reset={() => void action(async () => { await api.reset(); setExpanded(false); setOptions(false); question.current?.focus() })} />}
    <input ref={file} type="file" multiple hidden accept="image/png,image/jpeg,image/webp,text/plain,text/markdown,.md,.txt,.csv,.json,.log" onChange={event => { if (event.target.files) void ingest(event.target.files); event.target.value = '' }} />
    {settings && snapshot && <AskPreferences value={snapshot.preferences} routes={snapshot.routes} tx={tx} error={error} record={record} close={() => setSettings(false)} save={preferences => action(() => api.configure(preferences))} />}
    {preview && <AskContextPreview group={preview} close={() => setPreview(null)} tx={tx} />}
    {choices && <AskWindowChoices windows={choices} choose={token => void chooseWindow(token)} close={() => setChoices(null)} blocked={busy} tx={tx} />}
    {displays && <AskDialog title={tx('Choose a display to capture')} onClose={() => setDisplays(null)}>
      <p className="mb-4 text-sm text-ink-3">{tx('The screenshot will be attached for review before you send it.')}</p>
      <div className="space-y-2">{displays.map(display => <button key={display.id} className="ask-secondary w-full" onClick={() => void capture(display.id)}>{display.label}</button>)}</div>
    </AskDialog>}
    {consent && route && <AskDialog title={tx('Review before sending')} onClose={() => setConsent(false)}>
      <div className="space-y-4 text-sm">
        <p>{tx('This conversation will be sent to {destination}.', { destination: route.location })}</p>
        <p className="rounded-lg border border-line bg-card-2 p-3 font-medium">{route.model}</p>
        <p className="text-ink-3">{tx('Includes the conversation so far, your question and attached context. Review attachments before continuing.')}</p>
        <div className="flex justify-end gap-2"><button className="ask-secondary" onClick={() => setConsent(false)}>{tx('Cancel')}</button><button className="ask-primary" onClick={() => void submit(true)}>{tx('Send to this destination')}</button></div>
      </div>
    </AskDialog>}
  </main>
}
