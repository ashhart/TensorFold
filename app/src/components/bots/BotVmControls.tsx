import { authoredText } from '../../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useState, useSyncExternalStore } from 'react'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import type { BotVmSetupStatus, BotVmStatus } from '../../lib/types'
import { Button } from '../ui'

type VmAction = 'refresh' | 'setup' | 'runtime' | 'pick' | 'provision' | 'start' | 'stop' | 'rebuild'
const PHASE_LABEL: Record<BotVmSetupStatus['phase'], string> = {
  get idle() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Ready to set up") }, get downloading() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Downloading the signed Apple runtime") }, get verifying() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Checking checksum and Apple signature") },
  get extracting() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Preparing the private runtime") }, get activating() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Activating the verified runtime") }, get starting() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Starting private VM services") },
  get provisioning() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Preparing the bot workspace") }, get ready() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "VM setup complete") }, get cancelled() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Cancellation requested") }, get error() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Setup needs attention") },
}

// One UI subscription shares the host's global installer status across bot cards.
// There is no timer when the view has no subscribers or setup has settled.
let setupCache: { status: BotVmSetupStatus | null; error: string | null; requests: number; requestBotId: string | null } = { status: null, error: null, requests: 0, requestBotId: null }
const setupListeners = new Set<() => void>()
let setupRead: Promise<BotVmSetupStatus> | null = null
let setupTimer: ReturnType<typeof setTimeout> | null = null
function emitSetup() { for (const listener of setupListeners) listener() }
function scheduleSetupRead() {
  if (setupTimer) clearTimeout(setupTimer)
  setupTimer = null
  if (setupListeners.size && (setupCache.status?.active || setupCache.requests)) setupTimer = setTimeout(() => { setupTimer = null; void refreshSharedSetup().catch(() => {}) }, setupCache.error ? 1500 : 750)
}
function publishSetup(status: BotVmSetupStatus) { setupCache = { ...setupCache, status, error: null }; emitSetup(); scheduleSetupRead() }
function refreshSharedSetup(): Promise<BotVmSetupStatus> {
  if (!setupRead) setupRead = Promise.resolve().then(() => api.botVmSetupStatus()).then((status) => { publishSetup(status); return status }).catch((cause) => {
    setupCache = { ...setupCache, error: cause instanceof Error ? cause.message : authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "Could not check VM setup progress.") }; emitSetup(); throw cause
  }).finally(() => { setupRead = null; scheduleSetupRead() })
  return setupRead
}
function subscribeSetup(listener: () => void): () => void {
  setupListeners.add(listener)
  if (setupListeners.size === 1) void refreshSharedSetup().catch(() => {})
  return () => { setupListeners.delete(listener); if (!setupListeners.size && setupTimer) { clearTimeout(setupTimer); setupTimer = null } }
}
function changeSetupRequests(delta: number, botId: string) {
  const requests = Math.max(0, setupCache.requests + delta)
  setupCache = { ...setupCache, requests, requestBotId: delta > 0 && setupCache.requests === 0 ? botId : setupCache.requestBotId }
  emitSetup(); scheduleSetupRead()
}
function downloadSize(bytes: number): string { return `${(Math.max(0, Number.isFinite(bytes) ? bytes : 0) / 1_000_000).toFixed(1)} MB` }

export function BotVmControls({ botId, archived, paused }: { botId: string; archived: boolean; paused: boolean }) {
  const tx = useAuthoredText()
  const sharedSetup = useSyncExternalStore(subscribeSetup, () => setupCache)
  const setup = sharedSetup.status
  const [status, setStatus] = useState<BotVmStatus | null>(null)
  const [operation, setOperation] = useState<VmAction | null>(null)
  const [cancelPending, setCancelPending] = useState(false)
  const [setupStarted, setSetupStarted] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [rebuild, setRebuild] = useState(false)
  const pending = operation !== null
  const setupActive = !!setup?.active || sharedSetup.requests > 0
  const otherSetup = setupActive && (setup?.botId ?? sharedSetup.requestBotId) !== botId
  const busy = pending || setupActive
  useEffect(() => { if (setup?.active && setup.botId === botId) setSetupStarted(true) }, [setup?.active, setup?.botId, botId])
  useEffect(() => {
    let current = true
    setStatus(null); setError(null); setRebuild(false)
    void api.botVmStatus(botId).then((value) => { if (current) setStatus(value) }).catch((cause) => { if (current) setError(String(cause?.message ?? 'Could not check this VM.')) })
    return () => { current = false }
  }, [botId, archived, paused])
  useEffect(() => {
    if (!setupActive) setCancelPending(false)
    if (setup?.phase === 'ready' && !setup.active) {
      let current = true
      void api.botVmStatus(botId).then((value) => { if (current) setStatus(value) }).catch((cause) => { if (current) setError(String(cause?.message ?? 'Could not confirm VM readiness.')) })
      return () => { current = false }
    }
  }, [setupActive, setup?.phase, setup?.active, botId])
  const action = async (kind: VmAction) => {
    if (pending || setupActive && kind !== 'refresh') return
    setOperation(kind); setError(null)
    if (kind === 'setup') { setSetupStarted(false); changeSetupRequests(1, botId) }
    try {
      if (kind === 'setup') {
        const request = api.botVmManagedSetup(botId)
        void refreshSharedSetup().catch(() => {})
        setStatus(await request)
      }
      else if (kind === 'pick') await api.botVmPickRuntime()
      else if (kind === 'runtime') await api.botVmStartRuntime()
      else if (kind === 'provision') await api.botVmProvision(botId)
      else if (kind === 'start') await api.botVmStart(botId)
      else if (kind === 'stop' || kind === 'rebuild') {
        const state = useStore.getState()
        for (const thread of state.threads.filter((item) => item.bot?.presetId === botId && item.status === 'running')) await state.cancel(thread.id)
        if (kind === 'stop') await api.botVmStop(botId)
        else { await api.botVmRebuild(botId); setRebuild(false) }
      }
      const [next] = await Promise.all([api.botVmStatus(botId), refreshSharedSetup()])
      setStatus(next)
    } catch (cause) {
      const latest = await refreshSharedSetup().catch(() => null)
      if (kind !== 'setup' || latest?.phase !== 'cancelled') setError(cause instanceof Error ? cause.message : tx("VM action failed; no work moved to this Mac."))
      setStatus(await api.botVmStatus(botId).catch(() => status))
    } finally {
      if (kind === 'setup') changeSetupRequests(-1, botId)
      setOperation(null)
    }
  }
  const cancelSetup = async () => {
    if (cancelPending || otherSetup || !setup?.active) return
    setCancelPending(true); setError(null)
    try { publishSetup(await api.botVmCancelSetup()) }
    catch (cause) { setError(cause instanceof Error ? cause.message : tx("Could not cancel VM setup.")); setCancelPending(false) }
  }
  const available = !!status && status.state !== 'unsupported' && api.platform !== 'web-remote'
  const needsSetup = !!status && ['setup-required', 'runtime-stopped', 'not-created', 'blocked'].includes(status.state)
  const setupForThisBot = (setup?.botId ?? sharedSetup.requestBotId) === botId
  const failedSetupForThisBot = setupForThisBot && (setup?.phase === 'error' || setup?.phase === 'cancelled')
  const showSetupProgress = !!setup && setupForThisBot && (setupActive || setup.phase === 'ready' || failedSetupForThisBot && needsSetup)
  const awaitingSetupStart = operation === 'setup' && !setupStarted && !setup?.active
  const setupError = error ?? sharedSetup.error
  const downloading = setup?.phase === 'downloading'
  const total = Number.isFinite(setup?.total) && (setup?.total ?? 0) > 0 ? setup!.total : 118_045_087
  const bytes = Math.max(0, Math.min(Number.isFinite(setup?.bytes) ? setup!.bytes : 0, total))

  return <div className="mt-4 rounded-lg border border-line bg-card-2 p-3" aria-label={tx("Bot VM controls")} aria-busy={busy}>
    <div className="flex items-center justify-between gap-2"><span className="text-[13px] font-medium text-ink">{tx("Bot VM")}</span><Button size="xs" variant="ghost" disabled={pending} onClick={() => void action('refresh')}>{tx("Refresh")}</Button></div>
    <p role="status" className="mt-1 break-words text-[12px] leading-relaxed text-ink-2">{operation && operation !== 'setup' ? tx("Updating VM…") : status?.detail ?? tx("Checking this VM…")}</p>
    {setupError && setupError !== setup?.detail && <p role="alert" className="mt-2 break-words text-[12px] text-danger">{setupError}</p>}
    {otherSetup && <p role="status" className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("Another bot VM is being set up; wait for it to finish before starting this setup.")}</p>}

    {showSetupProgress && <section aria-label={tx("VM setup progress")} className="mt-3 space-y-2 rounded-lg border border-line bg-card px-3 py-3">
      <p className="text-[12px] font-medium text-ink" aria-live="polite">{awaitingSetupStart ? tx("Starting setup…") : setup.phase === 'cancelled' ? setup.active ? tx("Stopping setup safely") : tx("Setup cancelled") : PHASE_LABEL[setup.phase]}</p>
      <p className="break-words text-[12px] leading-relaxed text-ink-2" role={setup.phase === 'error' && !awaitingSetupStart ? 'alert' : undefined}>{awaitingSetupStart ? tx("Waiting for the host to begin managed setup.") : setup.detail}</p>
      {downloading ? <><progress className="h-1.5 w-full accent-[var(--color-accent)]" value={bytes} max={total} aria-label={tx("Apple runtime download")} /><p className="text-[12px] tabular-nums text-ink-2">{downloadSize(bytes)} {tx("of ")}{downloadSize(total)}</p></> : setupActive && <div role="progressbar" aria-label={tx("VM setup")} aria-valuetext={PHASE_LABEL[setup.phase]} className="h-1.5 overflow-hidden rounded-full bg-ink-5"><div className="h-full w-1/3 animate-pulse rounded-full bg-accent" /></div>}
      {setup.active && <div className="flex flex-wrap items-center gap-2"><Button size="xs" disabled={cancelPending || setup.phase === 'cancelled'} onClick={() => void cancelSetup()}>{cancelPending || setup.phase === 'cancelled' ? tx("Stopping…") : tx("Cancel setup")}</Button><span className="text-[12px] leading-relaxed text-ink-2">{tx("Retry becomes available after cleanup finishes.")}</span></div>}
    </section>}

    {status && available && !archived && !paused && needsSetup && <section aria-label={tx("Managed VM setup")} className="mt-3 rounded-lg border border-line bg-card px-3 py-3">
      <h3 className="text-[13px] font-medium text-ink">{tx("Set up an isolated workspace")}</h3>
      <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("Downloads the pinned, signed Apple runtime, about 118 MB, and a pinned guest image, about 20 MB, plus Linux support files on first use; verified cached downloads can be reused.")}</p>
      <div className="mt-3 grid gap-2 sm:grid-cols-2 text-[12px] text-ink-2"><p><span className="font-medium text-ink">{tx("Resources")}</span><br />{tx("1 GiB memory, 2 CPU cores, and an 8 GiB private workspace cap per bot; allow 10 GiB free for the first setup.")}</p><p><span className="font-medium text-ink">{tx("Requirements")}</span><br />{tx("Apple silicon and macOS 26 or later, with a model on this Mac and Cortheon off.")}</p></div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("Guest network is off; file and command tools stay in the VM, with no cloud model, CLI or host execution fallback.")}</p>
      <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("Guest files are plaintext on disk; protection at rest depends on macOS disk encryption, such as FileVault.")}</p>
      <p className="mt-2 text-[12px] leading-relaxed text-ink-2">{tx("Setup starts the isolated workspace; it does not invoke a model or begin bot work.")}</p>
      <Button className="mt-3" variant="primary" size="sm" disabled={busy} onClick={() => void action('setup')}>{operation === 'setup' ? tx("Setting up…") : failedSetupForThisBot ? tx("Retry bot VM setup") : tx("Set up bot VM")}</Button>
      <details className="mt-3 text-[12px] text-ink-2"><summary className="cursor-pointer">{tx("Use an installed Apple runtime instead")}</summary><p className="mt-2 leading-relaxed">{tx("Select the signed Apple container executable you already installed; TensorFold verifies it before use.")}</p><div className="mt-2 flex flex-wrap gap-2"><Button size="xs" disabled={busy} onClick={() => void action('pick')}>{tx("Select installed runtime")}</Button>{status.state === 'runtime-stopped' && <Button size="xs" disabled={busy} onClick={() => void action('runtime')}>{tx("Start selected runtime")}</Button>}{status.state === 'not-created' && <Button size="xs" disabled={busy} onClick={() => void action('provision')}>{tx("Create VM with selected runtime")}</Button>}<a className="rounded-md px-2 py-1 text-[12px] text-accent underline" href={status.setupUrl} target="_blank" rel="noreferrer">{tx("Official runtime release")}</a></div></details>
    </section>}

    {status && available && !archived && !paused && <div className="mt-2 flex flex-wrap gap-2">
      {status.state === 'stopped' && <Button size="xs" disabled={busy} onClick={() => void action('start')}>{tx("Start VM")}</Button>}
      {status.state === 'running' && <Button size="xs" disabled={busy} onClick={() => void action('stop')}>{tx("Stop VM and bot run")}</Button>}
      {['stopped', 'running', 'blocked'].includes(status.state) && <Button size="xs" variant="ghost" disabled={busy} onClick={() => setRebuild(!rebuild)}>{tx("Rebuild image")}</Button>}
    </div>}
    {rebuild && !paused && !archived && <div className="mt-3 border-t border-line pt-3"><p className="mb-2 text-[12px] text-ink-2">{tx("Stop this bot and replace its guest system image; its /workspace files stay available.")}</p><div className="flex flex-wrap gap-2"><Button size="xs" disabled={busy} onClick={() => setRebuild(false)}>{tx("Cancel")}</Button><Button size="xs" disabled={busy} onClick={() => void action('rebuild')}>{tx("Rebuild and keep files")}</Button></div></div>}
    <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("Your Mac files, accounts and host tools are not mounted; archiving keeps the bot's VM files.")}</p>
  </div>
}
