/** Neutral protocol/business rules shared by the native-phone host route and mobile UI. */
export interface WatchRelayCapabilities { taskStart: boolean; taskGuide: boolean; taskStop: boolean; approvalResolve: boolean; transcriptReview: boolean; changesReview: boolean; engineControl: boolean }
export interface WatchRelayHost {
  snapshot: {
    serverName: string; activeThreadIds: string[];
    threads: {id: string; title: string; kind: 'task' | 'chat'; incognito?: boolean; phoneLocalOnly?: boolean; googleLocalOnly?: boolean; providerId: string | null; model: string | null; status: string; updatedAt: number; timeline: {id: string; kind: string; state?: string; text?: string}[]}[];
    work: {jobs: {running: number; queued: number}; approvals: {pending: number}};
  };
  status: {capabilities: WatchRelayCapabilities; engine: {state: string; model: string | null} | null} | null;
}
export interface WatchRelayClient {
  send(threadId: string, text: string, active: boolean, capabilities: WatchRelayCapabilities): Promise<void>;
  watchSpeechStatus(): Promise<{available: boolean; provider: 'kokoro'; reason: string | null}>;
  startWatchSpeech(threadId: string, itemId: string): Promise<{speechId: string; provider: 'kokoro'; sampleRate: number}>;
  readWatchSpeech(speechId: string, sequence: number): Promise<{speechId: string; sequence: number; state: 'pending' | 'audio' | 'done' | 'error'; audioBase64?: string; sampleRate?: number; error?: string}>;
  cancelWatchSpeech(speechId: string): Promise<{cancelled: boolean}>;
}
const RESTRICTED_CAPABILITIES: WatchRelayCapabilities = {taskStart: false, taskGuide: false, taskStop: false, approvalResolve: false, transcriptReview: false, changesReview: false, engineControl: false}
export class WatchRelayError extends Error { constructor(message: string) { super(message); this.name = 'WatchRelayError' } }
function mobileThreadState(thread: WatchRelayHost['snapshot']['threads'][number], active: readonly string[]): 'running' | 'waiting' | 'failed' | 'finished' | 'idle' {
  if (active.includes(thread.id) && thread.timeline.some(item => item.state === 'awaiting' || item.state === 'pending' && item.kind === 'approval')) return 'waiting'
  if (active.includes(thread.id)) return 'running'
  if (thread.status === 'error') return 'failed'
  return thread.timeline.some(item => item.kind === 'assistant') ? 'finished' : 'idle'
}

export const WATCH_PROTOCOL = 1
export const WATCH_MAX_AGE_MS = 45_000
export const WATCH_REPLY_LIMIT = 2_000
export interface WatchJob {
  id: string; title: string; state: ReturnType<typeof mobileThreadState>; updatedAt: number;
  canReply: boolean; latestReply: string | null; latestReplyId: string | null;
}
export interface WatchSnapshot {
  version: 1; session: string; generatedAt: number; emittedAt: number; connected: boolean;
  serverName: string; jobs: WatchJob[]; running: number; queued: number; approvals: number;
  transcriptReview: boolean; error: string | null;
}
export interface WatchCommand {
  version: 1; id: string; action: 'refresh' | 'reply' | 'audio-start' | 'audio-next' | 'audio-stop'; session: string;
  threadId?: string; text?: string; itemId?: string; speechId?: string; sequence?: number; generatedAt: number;
}
export interface WatchResult { id: string; ok: boolean; message: string; uncertain?: boolean; snapshot?: WatchSnapshot; speech?: Awaited<ReturnType<WatchRelayClient['startWatchSpeech']>>; audio?: Awaited<ReturnType<WatchRelayClient['readWatchSpeech']>> }
function bounded(value: unknown, max: number): string { return typeof value === 'string' ? value.slice(0, max) : '' }

/** Only the phone's already-redacted host response enters the watch; no token, URL, tool output or path. */
export function watchSnapshot(host: WatchRelayHost | null, session: string, now: number, error: string | null = null, emittedAt = now): WatchSnapshot {
  const caps = host?.status?.capabilities ?? RESTRICTED_CAPABILITIES
  const active = host?.snapshot.activeThreadIds ?? []
  const jobs = (host?.snapshot.threads ?? []).filter((thread) => thread.kind === 'task' && thread.incognito !== true && thread.phoneLocalOnly !== true && thread.googleLocalOnly !== true)
    .sort((a, b) => Number(active.includes(b.id)) - Number(active.includes(a.id)) || b.updatedAt - a.updatedAt)
    .slice(0, 20).map((thread): WatchJob => {
      const item = [...thread.timeline].reverse().find((entry) => entry.kind === 'assistant')
      const localReady = thread.providerId !== 'tensorfold' || host?.status?.engine?.state === 'ready' && host.status.engine.model === thread.model
      return { id: bounded(thread.id, 256), title: bounded(thread.title, 100), state: mobileThreadState(thread, active), updatedAt: thread.updatedAt,
        canReply: !error && (active.includes(thread.id) ? caps.taskGuide : caps.taskStart && localReady),
        latestReply: item && caps.transcriptReview ? bounded(item.text, 1_500) : null, latestReplyId: item ? bounded(item.id, 256) : null }
    })
  return { version: WATCH_PROTOCOL, session, generatedAt: now, emittedAt, connected: !!host && !error, serverName: bounded(host?.snapshot.serverName, 80), jobs,
    running: host?.snapshot.work.jobs.running ?? 0, queued: host?.snapshot.work.jobs.queued ?? 0, approvals: host?.snapshot.work.approvals.pending ?? 0,
    transcriptReview: caps.transcriptReview, error: error ? bounded(error, 240) : host ? null : 'Open TensorFold on your paired iPhone and connect to your Mac.' }
}

export function parseWatchCommand(raw: unknown): WatchCommand {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new WatchRelayError('Unsupported watch request.')
  const input = raw as Record<string, unknown>
  if (input.version !== WATCH_PROTOCOL || typeof input.id !== 'string' || !/^[a-zA-Z0-9-]{8,80}$/.test(input.id)
    || typeof input.session !== 'string' || input.session.length > 80 || !Number.isFinite(input.generatedAt)
    || !['refresh', 'reply', 'audio-start', 'audio-next', 'audio-stop'].includes(String(input.action))) throw new WatchRelayError('Unsupported watch request.')
  if (input.action === 'reply' && (typeof input.threadId !== 'string' || !input.threadId || input.threadId.length > 256
    || typeof input.text !== 'string' || !input.text.trim() || input.text.length > WATCH_REPLY_LIMIT)) throw new WatchRelayError('Dictate a reply of up to 2,000 characters.')
  if (input.action === 'audio-start' && (typeof input.threadId !== 'string' || !input.threadId || input.threadId.length > 256 || typeof input.itemId !== 'string' || !input.itemId || input.itemId.length > 256)) throw new WatchRelayError('Select a job reply to hear.')
  if (['audio-next', 'audio-stop'].includes(String(input.action)) && (typeof input.speechId !== 'string' || !/^[a-zA-Z0-9-]{8,80}$/.test(input.speechId))) throw new WatchRelayError('Invalid speech session.')
  if (input.action === 'audio-next' && (!Number.isInteger(input.sequence) || Number(input.sequence) < 0 || Number(input.sequence) > 10_000)) throw new WatchRelayError('Invalid audio sequence.')
  return input as unknown as WatchCommand
}

/** Keep a bounded, at-most-once ledger; uncertain mutations never retry automatically. */
export function createWatchRelay<H extends WatchRelayHost = WatchRelayHost>(deps: {
  now(): number; session(): string; foreground(): boolean; client(): WatchRelayClient | null;
  host(): Promise<H>; publish(host: H): void;
  safeError?(error: unknown): string | null;
  /** Host relay can age out commands only after their original freshness fence expires. */
  ledgerMaxAgeMs?: number;
}) {
  const results = new Map<string, {promise: Promise<WatchResult>; generatedAt: number; fingerprint: string; settled: boolean}>()
  let inflight = 0, lifecycle = 0
  const audioSessions = new Map<string, WatchRelayClient>()
  async function perform(command: WatchCommand): Promise<WatchResult> {
    if (!deps.foreground()) return { id: command.id, ok: false, message: 'Open TensorFold on your iPhone to reply.' }
    const client = deps.client(), session = deps.session(), generation = lifecycle
    if (!client || !session || command.session !== session) return { id: command.id, ok: false, message: 'The phone connection changed; refresh this watch before replying.' }
    if (Math.abs(deps.now() - command.generatedAt) > WATCH_MAX_AGE_MS) return { id: command.id, ok: false, message: 'This request is stale; refresh before replying.' }
    let mutation = false
    try {
      if (command.action === 'audio-next') {
        const audio = await client.readWatchSpeech(command.speechId!, command.sequence!)
        if (generation !== lifecycle || deps.session() !== session || deps.client() !== client || !deps.foreground()) throw new WatchRelayError('The phone connection changed; refresh before listening.')
        if (audio.state === 'done' || audio.state === 'error') audioSessions.delete(command.speechId!)
        return {id: command.id, ok: true, message: 'Kokoro audio from your Mac.', audio}
      }
      if (command.action === 'audio-stop') {
        await client.cancelWatchSpeech(command.speechId!)
        audioSessions.delete(command.speechId!)
        if (generation !== lifecycle || deps.session() !== session || deps.client() !== client || !deps.foreground()) throw new WatchRelayError('The phone connection changed; refresh before listening.')
        return {id: command.id, ok: true, message: 'Kokoro playback stopped.'}
      }
      const host = await deps.host()
      if (generation !== lifecycle || deps.session() !== session || deps.client() !== client || !deps.foreground()) throw new WatchRelayError('The phone connection changed; refresh before replying.')
      const snapshot = watchSnapshot(host, session, deps.now())
      deps.publish(host)
      if (command.action === 'audio-start') {
        const capability = await client.watchSpeechStatus()
        if (!capability.available || capability.provider !== 'kokoro') throw new WatchRelayError(capability.reason || 'Install Kokoro in Voice settings on your Mac.')
        if (generation !== lifecycle || deps.session() !== session || deps.client() !== client || !deps.foreground()) throw new WatchRelayError('The phone connection changed; refresh before listening.')
        const thread = host.snapshot.threads.find((entry) => entry.id === command.threadId && entry.kind === 'task' && entry.incognito !== true && entry.phoneLocalOnly !== true && entry.googleLocalOnly !== true)
        if (!thread?.timeline.some((item) => item.id === command.itemId && item.kind === 'assistant')) throw new WatchRelayError('This reply is unavailable; refresh the job.')
        const speech = await client.startWatchSpeech(thread.id, command.itemId!)
        if (generation !== lifecycle || deps.session() !== session || deps.client() !== client || !deps.foreground()) {
          await client.cancelWatchSpeech(speech.speechId).catch(() => {})
          throw new WatchRelayError('The phone connection changed; refresh before listening.')
        }
        audioSessions.set(speech.speechId, client)
        return {id: command.id, ok: true, message: 'Preparing Kokoro on your Mac.', speech}
      }
      if (command.action === 'refresh') return { id: command.id, ok: true, message: 'Updated from your Mac.', snapshot }
      const thread = host.snapshot.threads.find((entry) => entry.id === command.threadId && entry.kind === 'task' && entry.incognito !== true && entry.phoneLocalOnly !== true && entry.googleLocalOnly !== true)
      const job = snapshot.jobs.find((entry) => entry.id === command.threadId)
      if (!thread || !job?.canReply) throw new WatchRelayError('This job cannot receive replies with the current host permissions or model state.')
      mutation = true
      await client.send(thread.id, command.text!, host.snapshot.activeThreadIds.includes(thread.id), host.status?.capabilities ?? RESTRICTED_CAPABILITIES)
      if (generation !== lifecycle || deps.session() !== session || deps.client() !== client || !deps.foreground()) throw new WatchRelayError('The phone connection changed after dispatch; check the original job before sending again.')
      return { id: command.id, ok: true, message: 'Reply accepted by your Mac.', snapshot }
    } catch (error) {
      return { id: command.id, ok: false, message: mutation ? 'Delivery could not be confirmed; check the job before sending again.' : error instanceof WatchRelayError ? error.message : deps.safeError?.(error) || 'Could not reach your Mac; refresh on your iPhone.', ...(mutation ? { uncertain: true } : {}) }
    }
  }
  return {
    async cancelAll() {
      lifecycle++
      const active = [...audioSessions.entries()]; audioSessions.clear()
      await Promise.allSettled(active.map(([id, client]) => client.cancelWatchSpeech(id)))
    },
    async handle(raw: unknown): Promise<WatchResult> {
      let command: WatchCommand
      try { command = parseWatchCommand(raw) } catch (error) { return { id: '', ok: false, message: (error as Error).message } }
      const key = `${command.session}:${command.id}`
      // Freshness/identity precede cache lookup: old/revoked requests cannot replay cached audio or transcripts.
      if (command.session !== deps.session() || !deps.foreground() || Math.abs(deps.now() - command.generatedAt) > WATCH_MAX_AGE_MS) return perform(command)
      if (deps.ledgerMaxAgeMs !== undefined) for (const [id, entry] of results) {
        if (entry.settled && deps.now() - entry.generatedAt > Math.max(WATCH_MAX_AGE_MS, deps.ledgerMaxAgeMs)) results.delete(id)
      }
      const fingerprint = JSON.stringify([command.action, command.threadId, command.text, command.itemId, command.speechId, command.sequence, command.generatedAt])
      if (command.action === 'audio-next' || command.action === 'refresh') {
        if (inflight >= 4) return {id: command.id, ok: false, message: 'The phone is busy; try again.'}
        inflight++; try { return await perform(command) } finally { inflight-- }
      }
      const previous = results.get(key)
      if (previous) {
        if (previous.fingerprint !== fingerprint) return {id: command.id, ok: false, message: 'This request identifier was already used; create a new watch request.'}
        const result = await previous.promise
        // The mutation is never rerun, and an old transcript projection is never replayed.
        const {snapshot: _snapshot, ...receipt} = result
        return receipt
      }
      if (inflight >= 4 || results.size >= 256) return { id: command.id, ok: false, message: 'Too many watch requests; reopen TensorFold on your iPhone.' }
      inflight++
      const entry = {promise: Promise.resolve(null as unknown as WatchResult), generatedAt: command.generatedAt, fingerprint, settled: false}
      entry.promise = perform(command).finally(() => { inflight--; entry.settled = true })
      results.set(key, entry)
      const result = entry.promise
      return result
    },
  }
}
