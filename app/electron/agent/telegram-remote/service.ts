import type { Room, Thread } from '../types'
import type { TelegramRemoteConfigInput, TelegramRemoteIngestResult, TelegramRemotePublicStatus, TelegramRemoteSession, TelegramRemoteTurnRequest, TelegramRemoteTurnResult } from '../../../shared/telegram-remote'
import { listTelegramTargets, parseTelegramRemoteCommand, TELEGRAM_REMOTE_HELP, telegramTurnReply } from './commands'
import { authenticateTelegramOwner, eligibleTelegramRoom, eligibleTelegramThread, remoteSettings, targetInTelegramScope, telegramAuthorizationRevision, telegramRemoteScope, telegramScopeRevision, type RemoteTelegramSettings } from './policy'

export interface TelegramRemoteServiceDeps {
  settings(): RemoteTelegramSettings
  threads(): readonly Thread[]
  rooms(): readonly Room[]
  /** Must complete durable protected persistence; a failed checkpoint forbids dispatch. */
  persist(): Promise<void>
  dispatchTurn(request: TelegramRemoteTurnRequest): Promise<TelegramRemoteTurnResult>
  now?(): number
  timeoutMs?: number
}

export function createTelegramRemoteService(deps: TelegramRemoteServiceDeps) {
  const pending = new Map<string, AbortController>()
  const ingressOwners = new Set<string>()
  const now = deps.now ?? Date.now
  function getStatus(): TelegramRemotePublicStatus {
    const remote = remoteSettings(deps.settings())
    const threads = deps.threads()
    return {
      enabled: remote.enabled, scope: remote.scope, threadIds: [...remote.threadIds], roomIds: [...remote.roomIds], revision: remote.revision,
      eligibleThreads: threads.filter((thread) => eligibleTelegramThread(thread, threads)).map(({ id, title }) => ({ id, title })),
      eligibleRooms: deps.rooms().filter(eligibleTelegramRoom).map(({ id, name, state }) => ({ id, name, state })),
    }
  }
  function cancelPending(): void {
    for (const controller of pending.values()) controller.abort(new Error('Telegram remote access changed.'))
  }
  async function configureRemote(input: TelegramRemoteConfigInput): Promise<TelegramRemotePublicStatus> {
    if (!input || typeof input.enabled !== 'boolean' || !['selected', 'all-eligible'].includes(input.scope)) throw new Error('Choose explicit Telegram remote access and scope.')
    for (const ids of [input.threadIds, input.roomIds]) if (!Array.isArray(ids) || ids.length > 500 || ids.some((id) => typeof id !== 'string' || !id || id.length > 160)) throw new Error('Telegram remote scope must contain bounded existing IDs.')
    const status = getStatus()
    if (input.threadIds.some((id) => !status.eligibleThreads.some((thread) => thread.id === id)) || input.roomIds.some((id) => !status.eligibleRooms.some((room) => room.id === id))) throw new Error('A selected chat or room is private, deleted, or unavailable.')
    cancelPending()
    const settings = deps.settings()
    const previous = settings.remote
    const priorState = remoteSettings(settings)
    if (!Number.isSafeInteger(priorState.revision + 1)) throw new Error('Telegram access revision cannot advance; review the local configuration.')
    const previousCursor = settings.lastUpdateId
    const consumedCursor = Math.max(settings.lastUpdateId ?? -1, ...priorState.sessions.map((session) => session.lastUpdateId))
    if (consumedCursor >= 0) settings.lastUpdateId = consumedCursor
    settings.remote = { enabled: input.enabled, scope: input.scope, threadIds: [...new Set(input.threadIds)], roomIds: [...new Set(input.roomIds)], revision: priorState.revision + 1, sessions: [] }
    try { await deps.persist() } catch (error) { settings.remote = previous; settings.lastUpdateId = previousCursor; throw error }
    return getStatus()
  }
  async function ingest(update: unknown): Promise<TelegramRemoteIngestResult> {
    const settings = deps.settings()
    if (!remoteSettings(settings).enabled) return { consumed: false, reply: '' }
    const owner = authenticateTelegramOwner(update, settings)
    if (!owner) return { consumed: true, reply: '', skipped: true }
    if (ingressOwners.has(owner.ownerKey)) return { consumed: true, reply: '', skipped: true }
    ingressOwners.add(owner.ownerKey)
    try { return await ingestOwned(update, owner) } finally { ingressOwners.delete(owner.ownerKey) }
  }
  async function ingestOwned(update: unknown, owner: NonNullable<ReturnType<typeof authenticateTelegramOwner>>): Promise<TelegramRemoteIngestResult> {
    const settings = deps.settings()
    const revision = telegramAuthorizationRevision(settings)
    const remote = remoteSettings(settings)
    const previous = remote.sessions.find((item) => item.ownerKey === owner.ownerKey)
    if (owner.updateId <= (previous?.lastUpdateId ?? -1) || owner.updateId <= (settings.lastUpdateId ?? -1)) return { consumed: true, reply: '', skipped: true }
    const priorSessions = remote.sessions
    const session: TelegramRemoteSession = previous?.authorizationRevision === revision ? { ...previous, target: { ...previous.target }, recentMessageTimes: [...previous.recentMessageTimes] } : {
      ownerKey: owner.ownerKey, authorizationRevision: revision, target: { kind: 'chief' }, lastUpdateId: previous?.lastUpdateId ?? -1, updatedAt: now(), recentMessageTimes: [],
    }
    const scope = telegramRemoteScope(settings, deps.threads(), deps.rooms())
    const scopeRevision = telegramScopeRevision(scope)
    const command = parseTelegramRemoteCommand(owner.text)
    const limited = session.recentMessageTimes.filter((time) => time > now() - 60_000)
    const tooMany = limited.length >= 12
    const active = pending.has(owner.ownerKey)
    session.recentMessageTimes = [...limited, now()].slice(-12)
    session.lastUpdateId = owner.updateId
    session.updatedAt = now()
    if (!remote.sessions.includes(session)) remote.sessions = [...remote.sessions.filter((item) => item.ownerKey !== owner.ownerKey), session].slice(-100)
    const proposedSessions = remote.sessions
    let reply: string | null = previous && previous.authorizationRevision !== revision && command.kind === 'message'
      ? 'Telegram access changed, so the previous destination was cleared. Choose /chief, /chats or /rooms; this message was not sent to another destination.' : null
    if (tooMany) reply = 'Too many remote requests. Wait a minute and try again.'
    else if (active) reply = 'A remote request for this owner is still running. No new message was sent.'
    else if (command.kind === 'help') reply = TELEGRAM_REMOTE_HELP
    else if (command.kind === 'unsupported') reply = 'Use /help for supported commands. Approvals and destructive controls remain in TensorFold on your Mac.'
    else if (command.kind === 'rooms' || command.kind === 'chats') reply = listTelegramTargets(command.kind, scope)
    else if (command.kind === 'chief' || command.kind === 'back') { session.target = { kind: 'chief' }; reply = 'Chief of staff selected. Only your enabled chats and rooms are available.' }
    else if (command.kind === 'select') {
      if (!targetInTelegramScope(command.target, scope)) reply = 'That chat or room is not allowed, is private, or no longer exists. The current destination was kept.'
      else { session.target = command.target; reply = `${command.target.kind === 'chat' ? 'Chat' : 'Room'} ${command.target.id} selected. Send a message to continue there.` }
    }
    try { await deps.persist() } catch (error) {
      // Nothing has been dispatched. Failed durability must not consume this delivery
      // or replace the previous owner's destination/replay/rate-limit checkpoint.
      if (deps.settings().remote === remote && telegramAuthorizationRevision(deps.settings()) === revision) {
        if (remote.sessions === proposedSessions) remote.sessions = priorSessions
        else {
          remote.sessions = remote.sessions.filter((item) => item !== session)
          if (previous && !remote.sessions.some((item) => item.ownerKey === owner.ownerKey)) remote.sessions = [...remote.sessions, previous].slice(-100)
        }
      }
      throw error
    }
    const target = { ...session.target }
    const targetKey = JSON.stringify(target)
    const isAuthorized = (requireTarget = true) => {
      const current = deps.settings()
      const currentSession = remoteSettings(current).sessions.find((item) => item.ownerKey === owner.ownerKey)
      const currentScope = telegramRemoteScope(current, deps.threads(), deps.rooms())
      return !!authenticateTelegramOwner(update, current) && telegramAuthorizationRevision(current) === revision
        && currentSession?.authorizationRevision === revision && JSON.stringify(currentSession.target) === targetKey
        && telegramScopeRevision(currentScope) === scopeRevision && (!requireTarget || targetInTelegramScope(target, currentScope))
    }
    const result = (text: string): TelegramRemoteIngestResult => ({ consumed: true, reply: text.length > 3900 ? `${text.slice(0, 3800)}\n\nReply shortened. Open TensorFold for the full result.` : text, chatId: owner.chatId, authorizationRevision: revision, scopeRevision, target })
    const interrupted = (): TelegramRemoteIngestResult => {
      const current = deps.settings()
      const currentSession = remoteSettings(current).sessions.find((item) => item.ownerKey === owner.ownerKey)
      if (authenticateTelegramOwner(update, current) && telegramAuthorizationRevision(current) === revision && currentSession?.authorizationRevision === revision && JSON.stringify(currentSession.target) === targetKey) {
        return { ...result('The selected context changed or became private, so this request was stopped. Open TensorFold to continue; no captured answer was shared.'), scopeRevision: telegramScopeRevision(telegramRemoteScope(current, deps.threads(), deps.rooms())) }
      }
      return { consumed: true, reply: '', skipped: true }
    }
    if (!isAuthorized(false)) return { consumed: true, reply: '', skipped: true }
    if (reply !== null) return result(reply)
    if (!targetInTelegramScope(target, scope)) return result('The selected destination is no longer available. Choose /chats or /rooms; no fallback was used.')
    if (target.kind === 'room') {
      const room = scope.rooms.find((item) => item.id === target.id)
      if (!room || !['running', 'waiting_user', 'waiting_external'].includes(room.state)) return result('This room is not active. Start it in TensorFold first; Telegram will not start a room automatically.')
    }
    const controller = new AbortController()
    pending.set(owner.ownerKey, controller)
    const timeout = setTimeout(() => controller.abort(new Error('Telegram remote request timed out.')), deps.timeoutMs ?? 75_000)
    const watchdog = setInterval(() => { if (!isAuthorized()) controller.abort(new Error('Telegram remote access changed.')) }, 250)
    try {
      const run = deps.dispatchTurn({ ownerKey: owner.ownerKey, target, text: owner.text, scope, signal: controller.signal, isAuthorized: () => !controller.signal.aborted && isAuthorized() })
      const aborted = new Promise<never>((_, reject) => controller.signal.addEventListener('abort', () => reject(controller.signal.reason), { once: true }))
      const response = await Promise.race([run, aborted])
      if (!isAuthorized()) return interrupted()
      return result(telegramTurnReply(response))
    } catch {
      if (!isAuthorized()) return interrupted()
      return result(controller.signal.aborted ? 'The remote wait ended before a final answer. Check TensorFold before retrying; an admitted run may still need attention.' : 'The selected request failed. Check TensorFold for details before retrying; no alternate destination was used.')
    } finally {
      clearTimeout(timeout); clearInterval(watchdog)
      if (pending.get(owner.ownerKey) === controller) pending.delete(owner.ownerKey)
    }
  }
  function canDeliver(result: TelegramRemoteIngestResult): boolean {
    const settings = deps.settings()
    const scope = telegramRemoteScope(settings, deps.threads(), deps.rooms())
    const session = remoteSettings(settings).sessions.find((item) => item.ownerKey === `telegram:${result.chatId}:${result.chatId}`)
    return !!result.chatId && !!result.authorizationRevision && remoteSettings(settings).enabled && telegramAuthorizationRevision(settings) === result.authorizationRevision
      && telegramScopeRevision(scope) === result.scopeRevision && JSON.stringify(session?.target) === JSON.stringify(result.target) && parseAllowedOwner(result.chatId)
  }
  function parseAllowedOwner(chatId: string): boolean {
    return !!authenticateTelegramOwner({ update_id: 0, message: { chat: { id: chatId, type: 'private' }, from: { id: chatId, is_bot: false }, text: 'delivery' } }, deps.settings())
  }
  return { ingest, configureRemote, getStatus, cancelPending, canDeliver }
}
