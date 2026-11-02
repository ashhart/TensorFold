// A bounded final summary uses no tools and never replays an uncertain model request.

import { randomUUID } from 'node:crypto'
import { streamChat } from './openai'
import { classifyApiError, jitteredBackoff } from './api-error-classifier'
import { failedRequestDelivery } from './run-recovery/policy'
import { runRetryVerdict } from '../../shared/run-recovery'
import { recordModelCall } from './turn-stats'
import { type TimelineItem } from './types'
import { LOOP_CHECK_STRIDE, LOOP_CHECK_MIN, detectTextLoop } from './run-loop-guards'
import {
  STREAM_MAX_RETRIES,
  STREAM_RETRY_BASE_MS,
  STREAM_RETRY_MAX_MS,
  isDelegatedTransport,
  abortableDelay,
} from './run-loop-retry'
import { type RunSession } from './run-loop-session'

const FINAL_SUMMARY_MAX_MS = 30_000

export async function runGraceTurn(session: RunSession, reason: string): Promise<void> {
  const {
    deps, state, thread, provider, model, signal, persist, assertRequestAllowed, cwd, send, recovery,
    textLoopBreakerEnabled, messagesForModel, turnStats,
  } = session
  if (signal.aborted) return
  send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: `Run ${reason} reached — wrapping up with a final summary (no more actions).`, createdAt: Date.now() } })
  thread.apiMessages.push({ role: 'user', content: `[${reason} reached. You may NOT call any more tools. In one final message, briefly summarize what you accomplished, what still remains, and the recommended next step.]` })
  send({ type: 'status', status: 'thinking' })
  let item: TimelineItem | null = null
  let text = ''

  // Keep one deadline across summary attempts and backoff waits.
  const summaryDeadline = AbortSignal.timeout(Math.max(1, Math.min(FINAL_SUMMARY_MAX_MS, Math.ceil(deps.maxWallMs && deps.maxWallMs > 0 ? deps.maxWallMs : FINAL_SUMMARY_MAX_MS))))
  const summarySignal = AbortSignal.any([signal, summaryDeadline])

  for (let attempt = 1; ; attempt++) {
    if (signal.aborted) return
    if (summaryDeadline.aborted) {
      send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: 'The run stopped, but the model did not finish its final summary in time. You can send a new message or choose another model.', createdAt: Date.now() } })
      return
    }
    if (item) send({ type: 'item_update', id: item.id, patch: { text: '' } })
    item = null
    text = ''
    let summaryReasoning = ''
    let summaryLoopBroken = false
    let lastLoopCheckLen = 0
    const summaryAbort = new AbortController()
    const summaryRecoveryId = `summary:${attempt}`
    recovery.begin(summaryRecoveryId)
    recovery.requestStarted()
    try {
      for await (const chunk of streamChat({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, requireProviderFinish: true, messages: messagesForModel(), tools: [], temperature: deps.temperature, reasoningEffort: deps.reasoningEffort, approvalMode: 'readonly', cwd, signal: AbortSignal.any([summarySignal, summaryAbort.signal]), assertRequestAllowed })) {
        if (signal.aborted) break
        if (chunk.type === 'content' || chunk.type === 'reasoning') recovery.requestOutput()
        if (chunk.type === 'usage') recordModelCall(turnStats, chunk)
        if (chunk.type === 'content') {
          if (!item) { item = { id: randomUUID(), kind: 'assistant', text: '', createdAt: Date.now() }; state.lastAssistantId = item.id; send({ type: 'item_add', item }); send({ type: 'status', status: 'streaming' }) }
          text += chunk.text
          send({ type: 'item_delta', id: item.id, field: 'text', chunk: chunk.text })
        }
        if (chunk.type === 'reasoning') summaryReasoning += chunk.text
        const grown = text.length + summaryReasoning.length
        if (textLoopBreakerEnabled && grown >= LOOP_CHECK_MIN && grown - lastLoopCheckLen >= LOOP_CHECK_STRIDE) {
          lastLoopCheckLen = grown
          if (detectTextLoop(text) || detectTextLoop(summaryReasoning)) {
            summaryLoopBroken = true
            summaryAbort.abort()
            break
          }
        }
      }
      if (!signal.aborted && !summaryLoopBroken) recovery.requestComplete()
      if (summaryLoopBroken) {
        text = text.slice(0, 1200)
        send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: 'The model repeated itself in its final summary, so the response was stopped. Try another model or rephrase the request.', createdAt: Date.now() } })
      }
      if (item) send({ type: 'item_update', id: item.id, patch: { text, ...(summaryLoopBroken ? { meta: 'stopped — repeating itself' } : {}) } })
      if (text) thread.apiMessages.push({ role: 'assistant', content: text })
      persist()
      return
    } catch (e: any) {
      if (signal.aborted) return
      if (summaryDeadline.aborted) {
        if (item) send({ type: 'item_update', id: item.id, patch: { text, meta: 'stopped — summary time limit' } })
        if (text) thread.apiMessages.push({ role: 'assistant', content: text })
        send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: 'The run stopped, but the model did not finish its final summary in time. You can send a new message or choose another model.', createdAt: Date.now() } })
        persist()
        return
      }
      const delivery = failedRequestDelivery(e, !!(text || summaryReasoning))
      recovery.requestFailed(delivery)
      if (classifyApiError(e).retryable && runRetryVerdict({ delivery, visibleOutput: !!(text || summaryReasoning), delegated: isDelegatedTransport(provider.transport) }).allowed && attempt <= STREAM_MAX_RETRIES - 1) {
        await abortableDelay(jitteredBackoff(attempt, STREAM_RETRY_BASE_MS, STREAM_RETRY_MAX_MS), summarySignal)
        continue
      }
      if (text) {
        thread.apiMessages.push({ role: 'assistant', content: text })
        if (item) send({ type: 'item_update', id: item.id, patch: { text, meta: 'partial response preserved; not replayed' } })
        persist()
      }
      return
    } finally { recovery.end(summaryRecoveryId) }
  }
}
