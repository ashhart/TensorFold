import type { Room, Settings, StartRunPayload, Thread } from '../../agent/types'
import type { ThreadRunAdmission } from './thread-runtime'
import type { ChatParams, StreamChunk } from '../../agent/openai'
import type { TelegramRemoteTurnRequest, TelegramRemoteTurnResult } from '../../../shared/telegram-remote'
import { eligibleTelegramRoom, eligibleTelegramThread } from '../../agent/telegram-remote/policy'
import { runTelegramChief } from './telegram-remote-chief'

export interface TelegramRemoteDispatchDeps {
  settings(): Settings
  threads(): readonly Thread[]
  rooms(): readonly Room[]
  aborters: Map<string, AbortController>
  requireOperator(): void
  startThreadRun(payload: StartRunPayload, admission: ThreadRunAdmission): Promise<{ runId: string }>
  steerRoom(roomId: string, text: string): void
  streamChat(params: ChatParams): AsyncIterable<StreamChunk>
  waitMs?: number
  pollMs?: number
  chiefTimeoutMs?: number
}

export function createTelegramRemoteDispatch(deps: TelegramRemoteDispatchDeps) {
  function assertAllowed(request: TelegramRemoteTurnRequest): void {
    deps.requireOperator()
    if (request.signal.aborted || !request.isAuthorized()) throw new Error('Telegram remote authorization changed.')
  }
  async function chat(request: TelegramRemoteTurnRequest, threadId: string, text: string): Promise<TelegramRemoteTurnResult> {
    assertAllowed(request)
    const allowed = new Set(request.scope.threads.map((thread) => thread.id))
    const thread = deps.threads().find((item) => item.id === threadId)
    if (!thread || !allowed.has(threadId) || !eligibleTelegramThread(thread, deps.threads())) return { status: 'error', text: 'This chat is private, deleted, or outside the allowed Telegram scope.' }
    if (thread.status === 'running' || deps.aborters.has(threadId)) return { status: 'busy', text: 'This chat is already running. No message was appended or queued.' }
    const before = thread.timeline.length
    let owned: AbortController | null = null
    let admitted = false
    let admissionOpen = true
    const cancelOwned = () => { if (owned && deps.aborters.get(threadId) === owned) owned.abort(new Error('Telegram remote admission revoked.')) }
    const guard = () => {
      if (!admissionOpen) throw new Error('Telegram admission deadline expired.')
      assertAllowed(request)
      const live = deps.threads().find((item) => item.id === threadId)
      if (!live || live !== thread || !allowed.has(live.id) || !eligibleTelegramThread(live, deps.threads())) throw new Error('The selected chat changed or became private.')
    }
    request.signal.addEventListener('abort', cancelOwned)
    const watcher = setInterval(() => {
      if (admitted && owned && deps.aborters.get(threadId) !== owned) { clearInterval(watcher); request.signal.removeEventListener('abort', cancelOwned); return }
      try { guard() } catch { cancelOwned(); if (!owned) clearInterval(watcher) }
    }, 100)
    watcher.unref?.()
    try {
      guard()
      const startup = deps.startThreadRun({ threadId, userText: text }, { assertAllowed: guard, onController: (controller) => { owned = controller; admitted = true; guard(); if (request.signal.aborted) cancelOwned() } })
      const deadline = Date.now() + (deps.waitMs ?? 25_000)
      let rejectSetup!: () => void
      let setupTimer: ReturnType<typeof setTimeout> | undefined
      const setupAbort = new Promise<never>((_, reject) => {
        rejectSetup = () => { admissionOpen = false; cancelOwned(); reject(new Error('Telegram remote admission aborted.')) }
        request.signal.addEventListener('abort', rejectSetup, { once: true })
        setupTimer = setTimeout(rejectSetup, deps.waitMs ?? 25_000)
      })
      let runId: string
      try { ({ runId } = await Promise.race([startup, setupAbort])) }
      finally { clearTimeout(setupTimer); request.signal.removeEventListener('abort', rejectSetup) }
      while (Date.now() < deadline) {
        guard()
        const current = deps.aborters.get(threadId)
        const state = deps.threads().find((item) => item.id === threadId)?.status
        if (owned && current && current !== owned) return { status: 'incomplete', runId, text: 'Another run now owns this chat. Open TensorFold for the original run result.' }
        const items = thread.timeline.slice(before).filter((item) => item.trace?.runId === runId || item.runReceipt?.runId === runId)
        if (items.some((item) => item.kind === 'tool' && item.state === 'awaiting' || item.question?.state === 'pending')) return { status: 'incomplete', runId, text: 'This run needs an approval or answer in TensorFold on your Mac. Telegram cannot approve it.' }
        if (!current && state !== 'running') {
          const answer = items.filter((item) => item.kind === 'assistant' && item.text).at(-1)?.text
          const receipt = thread.lastRunReceipt?.runId === runId ? thread.lastRunReceipt : items.find((item) => item.runReceipt)?.runReceipt
          if (state === 'error' || items.some((item) => item.kind === 'error') || receipt?.verdict === 'failed') return { status: 'error', runId, text: 'The selected chat run failed. Open TensorFold for its error and retained history.' }
          if (receipt?.verdict === 'cancelled' || receipt?.verdict === 'blocked') return { status: 'incomplete', runId, text: 'The selected chat run stopped or needs review in TensorFold.' }
          if (receipt?.needsReview || receipt?.errors) return { status: 'incomplete', runId, text: `${answer ?? 'The selected run ended.'}\nReview the run details in TensorFold before treating it as complete.` }
          return answer ? { status: 'completed', runId, text: answer } : { status: 'incomplete', runId, text: 'The run ended without a final text answer. Open TensorFold for its results.' }
        }
        await new Promise((resolve) => setTimeout(resolve, deps.pollMs ?? 100))
      }
      return { status: 'queued', runId, text: 'The selected chat is still running. Its message is retained in TensorFold.' }
    } catch {
      cancelOwned()
      return { status: 'error', text: 'The selected chat could not start or remote access changed. No alternate chat or provider was used.' }
    } finally {
      if (!owned || deps.aborters.get(threadId) !== owned) { clearInterval(watcher); request.signal.removeEventListener('abort', cancelOwned) }
    }
  }
  return async function dispatchTurn(request: TelegramRemoteTurnRequest): Promise<TelegramRemoteTurnResult> {
    assertAllowed(request)
    if (request.target.kind === 'chat') return chat(request, request.target.id, request.text)
    if (request.target.kind === 'room') {
      const id = request.target.id
      const room = deps.rooms().find((item) => item.id === id)
      if (!room || !request.scope.rooms.some((item) => item.id === id) || !eligibleTelegramRoom(room) || !['running', 'waiting_user', 'waiting_external'].includes(room.state)) return { status: 'error', text: 'That allowed room is no longer active. Telegram will not start another room.' }
      assertAllowed(request)
      deps.steerRoom(id, request.text)
      return { status: 'queued', text: 'Guidance was added to the selected active room. Open TensorFold for its next response.' }
    }
    const dispatch = request.text.match(/^dispatch\s+(?:chat\s+)?([A-Za-z0-9_.:-]{1,160})\s+([\s\S]+)$/i)
    if (dispatch) {
      const result = await chat(request, dispatch[1], dispatch[2])
      return { ...result, text: `Chat ${dispatch[1]}: ${result.text ?? result.status}` }
    }
    return runTelegramChief(request, { settings: deps.settings, threads: deps.threads, rooms: deps.rooms, streamChat: deps.streamChat, timeoutMs: deps.chiefTimeoutMs })
  }
}
