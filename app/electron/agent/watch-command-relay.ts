import { AsyncLocalStorage } from 'node:async_hooks'
import { createWatchRelay, parseWatchCommand, WATCH_MAX_AGE_MS, WatchRelayError, type WatchRelayClient, type WatchRelayHost, type WatchResult } from '../../shared/watch-relay'
import type { WatchSpeechBroker } from './watch-speech'

export const WATCH_COMMAND_RESPONSE_BYTES = 60_000
const MAX_RELAY_SESSIONS = 16
export interface HostWatchCommandRelayDeps {
  now?: () => number
  authorize(token: string): {ok: boolean; sessionId?: string; scope?: 'paired' | 'owner'}
  request(method: 'GET' | 'POST', path: string, body: unknown | undefined, token: string, signal: AbortSignal): Promise<{status: number; value: unknown}>
  speech?: WatchSpeechBroker
}
const capNames = ['taskStart', 'taskGuide', 'taskStop', 'approvalResolve', 'transcriptReview', 'changesReview', 'engineControl'] as const
function object(value: unknown): value is Record<string, unknown> { return !!value && typeof value === 'object' && !Array.isArray(value) }
function text(value: unknown): value is string { return typeof value === 'string' }
function count(value: unknown): boolean { return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1_000_000 }
function validateStatus(status: unknown): WatchRelayHost['status'] {
  if (status !== null && (!object(status) || status.scope !== 'paired' || !object(status.capabilities) || !capNames.every(name => typeof (status.capabilities as Record<string, unknown>)[name] === 'boolean')
    || status.engine !== null && (!object(status.engine) || !text(status.engine.state) || status.engine.model !== null && !text(status.engine.model)))) throw new WatchRelayError('Your Mac returned unsupported device permissions.')
  return status as WatchRelayHost['status']
}
function validateHost(snapshot: unknown, status: unknown): WatchRelayHost {
  if (!object(snapshot) || !text(snapshot.serverName) || !Array.isArray(snapshot.activeThreadIds) || snapshot.activeThreadIds.length > 1000 || !snapshot.activeThreadIds.every(text)
    || !Array.isArray(snapshot.threads) || snapshot.threads.length > 1000 || !object(snapshot.work) || !object(snapshot.work.jobs) || !object(snapshot.work.approvals)
    || !count(snapshot.work.jobs.running) || !count(snapshot.work.jobs.queued) || !count(snapshot.work.approvals.pending)
    || !snapshot.threads.every(thread => object(thread) && text(thread.id) && thread.id.length > 0 && thread.id.length <= 256 && text(thread.title) && ['task', 'chat'].includes(String(thread.kind))
      && (thread.providerId === null || text(thread.providerId)) && (thread.model === null || text(thread.model)) && text(thread.status) && typeof thread.updatedAt === 'number' && Number.isFinite(thread.updatedAt)
      && Array.isArray(thread.timeline) && thread.timeline.length <= 5000 && thread.timeline.every(item => object(item) && text(item.id) && text(item.kind) && (item.text === undefined || text(item.text))))) throw new WatchRelayError('Your Mac returned an unsupported job snapshot.')
  // Missing/legacy status stays read-only; no capability is inferred from the phone or watch.
  return {snapshot, status: validateStatus(status)} as unknown as WatchRelayHost
}
/** Strict route envelope; fields are inert data and never become URLs, credentials, shell or file paths. */
export function parseHostWatchCommand(raw: unknown) {
  if (!object(raw) || Object.keys(raw).some(key => !['version', 'id', 'action', 'session', 'threadId', 'text', 'itemId', 'speechId', 'sequence', 'generatedAt'].includes(key))) throw new WatchRelayError('Unsupported watch request.')
  return parseWatchCommand(raw)
}

/** The always-alive Mac owns the ledger; neither React nor an unlocked/foreground iPhone participates. */
export function createHostWatchCommandRelay(deps: HostWatchCommandRelayDeps) {
  const now = deps.now ?? Date.now
  const context = new AsyncLocalStorage<AbortSignal>()
  const relays = new Map<string, {token: string; relay: ReturnType<typeof createWatchRelay>}>()
  let active = true
  function identity(token: string): string {
    if (!active || context.getStore()?.aborted) return ''
    const value = deps.authorize(token)
    return value.ok && value.sessionId && (value.scope === undefined || value.scope === 'paired') ? value.sessionId : ''
  }
  async function request(token: string, method: 'GET' | 'POST', path: string, body?: unknown) {
    if (!identity(token)) throw new WatchRelayError('This phone pairing changed; connect again on your iPhone.')
    const result = await deps.request(method, path, body, token, context.getStore() ?? new AbortController().signal)
    if (!identity(token)) throw new WatchRelayError('This phone pairing changed; connect again on your iPhone.')
    if (result.status < 200 || result.status >= 300) throw new WatchRelayError(result.status === 403 ? 'Your Mac does not permit this action for this paired device.' : result.status === 409 ? 'The job changed; refresh before replying.' : 'Your Mac could not complete this watch request.')
    return result.value
  }
  async function permissions(token: string): Promise<WatchRelayHost['status']> {
    return validateStatus(await request(token, 'GET', '/v1/device-status'))
  }
  function create(token: string, session: string) {
    const client: WatchRelayClient = {
      async send(threadId, userText, running, caps) {
        if (!(running ? caps.taskGuide : caps.taskStart)) throw new WatchRelayError('Sending is disabled by your Mac for this paired device.')
        await request(token, 'POST', `/v1/threads/${encodeURIComponent(threadId)}/${running ? 'guide' : 'start'}`, {userText: userText.trim()})
      },
      async watchSpeechStatus() {
        if (!identity(token) || !deps.speech) return {available: false, provider: 'kokoro', reason: 'Install and enable Kokoro watch replies on your Mac.'}
        return deps.speech.status(session)
      },
      async startWatchSpeech(threadId, itemId) {
        if (!identity(token) || !deps.speech) throw new WatchRelayError('Kokoro watch replies are unavailable.')
        return deps.speech.start(session, {threadId, itemId})
      },
      async readWatchSpeech(speechId, sequence) {
        if (!identity(token) || !deps.speech) throw new WatchRelayError('Kokoro watch replies are unavailable.')
        return deps.speech.chunk(session, speechId, sequence)
      },
      async cancelWatchSpeech(speechId) {
        if (!identity(token) || !deps.speech) return {cancelled: false}
        return deps.speech.cancel(session, speechId)
      },
    }
    return createWatchRelay({now, session: () => identity(token) === session ? session : '', foreground: () => active && !context.getStore()?.aborted,
      client: () => client, host: async () => validateHost(await request(token, 'GET', '/v1/snapshot'), await permissions(token)), publish() {}, ledgerMaxAgeMs: WATCH_MAX_AGE_MS})
  }
  return {
    async handle(token: string, raw: unknown, signal: AbortSignal): Promise<WatchResult> {
      return context.run(signal, async () => {
        let id = ''
        try {
          const command = parseHostWatchCommand(raw); id = command.id
          const session = identity(token)
          if (!session || command.session !== session) throw new WatchRelayError('The phone pairing changed; connect again on your iPhone.')
          if (Math.abs(now() - command.generatedAt) > WATCH_MAX_AGE_MS) throw new WatchRelayError('This request is stale; refresh before replying.')
          // Check current capabilities even for a duplicate receipt; it cannot bypass a later role downgrade.
          const currentStatus = await permissions(token)
          if (command.action === 'reply' && (!object(currentStatus) || !object(currentStatus.capabilities) || currentStatus.capabilities.taskStart !== true && currentStatus.capabilities.taskGuide !== true)) throw new WatchRelayError('Sending is disabled by your Mac for this paired device.')
          if (command.action.startsWith('audio-') && (!deps.speech || !deps.speech.isAllowed(session))) throw new WatchRelayError('Enable Kokoro watch replies for this paired device on your Mac.')
          let entry = relays.get(session)
          if (entry && entry.token !== token) throw new WatchRelayError('The phone pairing changed; connect again on your iPhone.')
          if (!entry) {
            // Revoked sessions can release their bounded ledger; valid session ledgers are never evicted to admit replay.
            for (const [key, value] of relays) if (!identity(value.token)) { await value.relay.cancelAll(); relays.delete(key) }
            if (relays.size >= MAX_RELAY_SESSIONS) throw new WatchRelayError('Too many connected watches; disconnect an unused paired device on your Mac.')
            entry = {token, relay: create(token, session)}; relays.set(session, entry)
          }
          const handled = await entry.relay.handle(command)
          // A reply can await dispatch while visibility/roles change. Only return
          // its receipt; the watch obtains a fresh projection through refresh.
          const {snapshot: _snapshot, ...receipt} = handled
          const result = command.action === 'reply' ? receipt : handled
          if (!identity(token)) return {id, ok: false, message: result.uncertain || command.action === 'reply' ? 'Delivery could not be confirmed; check the job before sending again.' : 'The phone pairing changed; refresh before listening.', ...(command.action === 'reply' ? {uncertain: true} : {})}
          if (Buffer.byteLength(JSON.stringify(result)) > WATCH_COMMAND_RESPONSE_BYTES) throw new WatchRelayError('This watch response is too large; refresh the job on your Mac.')
          return result
        } catch (error) { return {id, ok: false, message: error instanceof WatchRelayError ? error.message : 'Could not reach your Mac; check the private host connection.'} }
      })
    },
    async stop() { active = false; await Promise.allSettled([...relays.values()].map(entry => entry.relay.cancelAll())); relays.clear() },
  }
}
