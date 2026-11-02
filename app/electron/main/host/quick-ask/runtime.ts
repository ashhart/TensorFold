import { app, clipboard, globalShortcut, ipcMain, nativeTheme } from 'electron'
import { randomUUID } from 'node:crypto'
import path from 'node:path'
import type { Thread } from '../../../agent/types'
import type { QuickAskPreferences, QuickAskSnapshot, QuickAskApi } from '../../../../shared/quick-ask'
import { quickAskDefaults } from '../../../../shared/quick-ask'
import { QuickAskSessionHost, type AskSavedState } from '../../../agent/quick-ask/session'
import { createAskShortcut } from '../../../agent/quick-ask/shortcut'
import { boundedText, validateAskPreferences } from '../../../agent/quick-ask/validation'
import { readProtectedJson, writeProtectedJson, vontraHomeDir } from '../../../store'
import { store } from '../../state'
import { handle } from '../ipc'
import { requireSecurityRole } from '../audit'
import { sendToRenderer } from '../renderer-events'
import { openExternalSafely } from '../external'
import { HEADLESS_MODE } from '../../early-electron-paths'
import { createAskWindow } from './window'
import { askClipboard, askDisplays, captureAskDisplay } from './context'
import { allAskRoutes, routedAskGeneration, handoffRoutedAsk, loadRoutedAskModel } from './remote-client'
import { reserveAskOperation, quiesceAskBackend, resumeAskBackend } from './admission'
import { getManagedKokoroService } from '../ipc/provider-ipc'
import { synthesizeLocalTts } from '../voice-runtime'
import { localAskRoutes } from './backend'
import { importAskDraft } from '../../../../shared/quick-ask-handoff'
import { readComposerDraft, writeComposerDraft, clearComposerDraft } from '../../../store/composer-drafts'
import { createAskContextPicker } from './window-context'

let live: ReturnType<typeof createQuickAskRuntime> | null = null
export async function quiesceQuickAsk(): Promise<void> { await Promise.all([live?.quiesce(), quiesceAskBackend()]) }
export function resumeQuickAsk(): void { resumeAskBackend(); live?.resume() }
export async function purgeQuickAsk(): Promise<void> { await live?.purge() }

export function createQuickAskRuntime(options: { baseDir: string; showMain(): void; changed(): void; error(message: string): void }) {
  const root = () => vontraHomeDir()
  const window = createAskWindow(options.baseDir, options.error)
  const contextPicker = createAskContextPicker()
  let preferences = quickAskDefaults(), shortcutError: string | null = null
  let loaded: Promise<void> | null = null, profile = '', pendingThread: Thread | null = null, initialized = false
  let recordingTimer: ReturnType<typeof setTimeout> | undefined
  let pending: Awaited<ReturnType<import('../../../../shared/quick-ask').QuickAskLauncher['consumeHandoff']>> = null
  const flushes = new Map<string, (error?: string) => void>()
  let catalogue: Awaited<ReturnType<typeof allAskRoutes>> | null = null, cataloguePending = false, catalogueAt = 0
  const shortcut = createAskShortcut(globalShortcut, () => { if (contextPicker.busy) { if (contextPicker.cancel()) window.open(); return }; if (window.visible()) window.hide(); else void open().catch(error => options.error(error.message)) })
  const authorize = () => requireSecurityRole('operator', 'use Ask TensorFold')
  const host = new QuickAskSessionHost({
    authorize, reserve: reserveAskOperation, emit: window.send,
    save: state => { if (root() !== profile) throw new Error('The active profile changed.'); return writeProtectedJson(path.join(profile, 'state', 'quick-ask-session.json'), state, { validateCommit: () => { if (root() !== profile) throw new Error('The active profile changed before saving.') } }) },
    routes: async () => (await allAskRoutes()).routes,
    generate: routedAskGeneration,
    handoff: async state => { pendingThread = await handoffRoutedAsk(state); return pendingThread.id },
  })
  async function start(): Promise<void> {
    if (loaded) return loaded
    profile = root()
    loaded = (async () => {
      const stored = await readProtectedJson<QuickAskPreferences>(path.join(profile, 'state', 'quick-ask-preferences.json'), quickAskDefaults(), false, true)
      preferences = validateAskPreferences(stored); host.setPreferences(preferences)
      host.restore(await readProtectedJson<AskSavedState | null>(path.join(profile, 'state', 'quick-ask-session.json'), null, false, true))
      if (!HEADLESS_MODE) { try { shortcut.apply(preferences); shortcutError = null } catch (error) { shortcutError = error instanceof Error ? error.message : 'The shortcut is unavailable.' } }
      initialized = true
    })()
    return loaded
  }
  async function open(): Promise<void> { authorize(); if (HEADLESS_MODE) throw new Error('Open Ask TensorFold on a desktop client.'); await start(); window.open() }
  async function snapshot(options?: { refreshRoutes?: boolean }): Promise<QuickAskSnapshot> {
    await start()
    const local = await localAskRoutes(), settings = store.settings
    if (!cataloguePending && (options?.refreshRoutes || Date.now() - catalogueAt > 10_000)) {
      cataloguePending = true; catalogueAt = Date.now()
      void allAskRoutes().then(result => { catalogue = result; window.send({ kind: 'refresh' }) }, () => {}).finally(() => { cataloguePending = false })
    }
    const result = { routes: [...local, ...(catalogue?.routes.filter(route => route.hostId) ?? [])], errors: catalogue?.errors ?? [] }
    const dark = settings.theme === 'dark' || settings.theme === 'system' && nativeTheme.shouldUseDarkColors
    const voice = settings.voice
    return { session: host.snapshot(), preferences, routes: result.routes, routeErrors: result.errors, shortcutError, platform: process.platform,
      appearance: { palette: dark ? settings.dark : settings.light, dark, uiPreferences: settings.uiPreferences },
      voice: { input: voice.enabled && voice.sttEnabled && voice.sttProvider === 'browser' ? 'browser' : 'unavailable', output: !voice.enabled || !voice.ttsEnabled ? 'unavailable' : ['local', 'kokoro'].includes(voice.ttsProvider) ? 'configured' : 'browser', destination: voice.sttProvider === 'browser' ? 'System speech service' : 'Voice input is unavailable with the current settings' } }
  }
  async function configure(value: QuickAskPreferences): Promise<QuickAskPreferences> {
    await start(); authorize()
    const next = validateAskPreferences(value), previous = preferences
    shortcut.apply(next)
    try { await writeProtectedJson(path.join(profile, 'state', 'quick-ask-preferences.json'), next) }
    catch (error) { shortcut.apply(previous); throw error }
    preferences = next; host.setPreferences(next); shortcutError = null; options.changed(); window.send({ kind: 'refresh' }); return next
  }
  const handlers: Record<string, (...args: any[]) => unknown> = {
    resize: height => window.resize(height),
    'pick-window': async () => {
      const lease = reserveAskOperation(), visible = window.visible()
      if (contextPicker.native) window.hide()
      try { return await contextPicker.pick() }
      finally { lease.release(); if (visible && contextPicker.native) window.open() }
    },
    'use-window': async token => {
      const lease = reserveAskOperation()
      try { return await contextPicker.use(token) } finally { lease.release() }
    },
    'record-shortcut': (active: boolean) => {
      if (typeof active !== 'boolean') throw new Error('Choose a valid recording state.')
      clearTimeout(recordingTimer)
      if (active) { shortcut.dispose(); recordingTimer = setTimeout(() => { try { shortcut.apply(preferences) } catch { shortcutError = 'The shortcut is unavailable. Choose another in settings.' } }, 15_000) }
      else shortcut.apply(preferences)
    },
    snapshot, draft: draft => host.saveDraft(draft), configure,
    send: input => host.send(input), cancel: () => host.cancel(), reset: () => host.reset(),
    clipboard: askClipboard, displays: askDisplays,
    capture: async displayId => {
      const visible = window.visible(); window.hide()
      try { await new Promise(resolve => setTimeout(resolve, 180)); return await captureAskDisplay(displayId) }
      finally { if (visible) window.open() }
    },
    load: async routeId => { const lease = reserveAskOperation(); try { await loadRoutedAskModel(boundedText(routeId, 2000, 'Model route')) } finally { lease.release() } },
    handoff: async () => {
      const draft = host.snapshot().draft
      await host.handoff()
      if (!pendingThread) {
        const saved = host.saved()
        pendingThread = await handoffRoutedAsk(saved)
      }
      const imported = await importAskDraft({ getComposerDraft: readComposerDraft, saveComposerDraft: writeComposerDraft, clearComposerDraft }, { surface: 'chat', conversationId: pendingThread.id, hostId: pendingThread.orchestratorHostId ?? null, incognito: false }, draft)
      pending = { thread: pendingThread, draft: imported ? draft : null }
      options.showMain(); sendToRenderer('quick-ask:navigate'); window.hide(); return { threadId: pendingThread.id }
    },
    copy: text => { clipboard.writeText(boundedText(text, 100_000, 'Copied text')) },
    link: url => { const value = boundedText(url, 2000, 'Link'); if (!['http:', 'https:'].includes(new URL(value).protocol)) throw new Error('Ask TensorFold opens public web links only.'); return openExternalSafely(value) },
    hide: () => window.hide(),
    flushed: input => { flushes.get(input?.requestId)?.(input?.error); flushes.delete(input?.requestId) },
    speak: async text => {
      const value = boundedText(text, 5000, 'Spoken answer'), voice = store.settings.voice
      if (!voice.enabled || !voice.ttsEnabled) return { ok: false, error: 'Enable voice output in TensorFold settings.' }
      if (voice.ttsProvider === 'kokoro') return getManagedKokoroService().synthesize({ requestId: `ask-${Date.now()}`, text: value })
      if (voice.ttsProvider === 'local') return synthesizeLocalTts(value)
      return { ok: false, error: 'Use the system voice selected in Ask TensorFold.' }
    },
  }
  for (const [name, callback] of Object.entries(handlers)) ipcMain.handle(`quick-ask:${name}`, async (event, ...args) => {
    if (!window.authorize(event)) throw new Error('Untrusted Ask TensorFold sender.')
    authorize(); await start(); return callback(...args)
  })
  handle('quick-ask:open', () => open())
  handle('quick-ask:consume-handoff', () => { const value = pending; pending = null; return value })
  app.on('before-quit', () => window.setQuitting(true))
  app.on('will-quit', () => { clearTimeout(recordingTimer); shortcut.dispose(); contextPicker.dispose() })
  const runtime = {
    open, start, enabled: () => preferences.enabled,
    mainClosed: () => { if (!preferences.enabled) window.destroy() },
    quiesce: async () => {
      if (!initialized) return
      if (!contextPicker.cancel()) throw new Error('Cancel the open macOS window picker before quitting or changing profiles.')
      window.setEnabled(false)
      if (window.available()) await new Promise<void>((resolve, reject) => {
        const requestId = randomUUID(), timer = setTimeout(() => { flushes.delete(requestId); reject(new Error('Ask TensorFold could not confirm its draft was saved. Copy the draft before quitting.')) }, 5000)
        flushes.set(requestId, error => { clearTimeout(timer); error ? reject(new Error('The Ask TensorFold draft could not be saved.')) : resolve() })
        window.send({ kind: 'flush', requestId })
      })
      await host.quiesce()
    },
    resume: () => { window.setQuitting(false); window.setEnabled(true); host.resume() }, purge: () => host.purge(),
  }
  live = runtime
  return runtime
}
