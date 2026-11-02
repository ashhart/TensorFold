// Own model attempts, retry evidence, and steering cleanup within one turn.

import { randomUUID } from 'node:crypto'
import { streamChat } from './openai'
import {
  createSteeringChannel,
  providerSupportsLiveSteering,
  runSteerCuePump,
  createHomegrownSteerWatcher,
  type HomegrownSteerWatcher,
} from './steering'
import { failedRequestDelivery } from './run-recovery/policy'
import { estimateTokens, kfmt } from './compaction'
import { recordModelCall } from './turn-stats'
import { type AgentCue, type CompactionLedgerEntry, type TimelineItem, type ToolCall } from './types'
import { LOOP_CHECK_STRIDE, LOOP_CHECK_MIN, detectTextLoop, compactionMadeProgress } from './run-loop-guards'
import {
  STREAM_MAX_RETRIES,
  MAX_STEER_RESTARTS,
  serverEfforts,
  isDelegatedTransport,
  streamRetryDecision,
  abortableDelay,
} from './run-loop-retry'
import { compactThread } from './run-loop-compaction'
import { type RunSession } from './run-loop-session'

export async function runModelTurn(session: RunSession, step: number) {
  const {
    deps, state, thread, provider, model, signal, persist, memoryPolicy, assertRequestAllowed, cwd,
    callbacksAllowed, send, recovery, autoCompact, contextLimit, textLoopBreakerEnabled, deadlineAt,
    modelToolSchemas, messagesForModel, turnStats,
  } = session
  let assistantItem: TimelineItem | null = null
  let reasoningItem: TimelineItem | null = null
  let content = ''
  let reasoning = ''
  let toolCalls: ToolCall[] = []
  let finishReason: string | null = null

  let loopBroken = false

  let compactRetries = 0
  const effortKey = `${provider.id}\n${model}\n${deps.reasoningEffort}`
  let serverEffort = serverEfforts.get(effortKey)

  const steerChannel = providerSupportsLiveSteering(provider) && deps.drainCues && deps.waitForCue
    ? createSteeringChannel()
    : undefined
  const steerPumpAbort = new AbortController()
  if (steerChannel) {
    void runSteerCuePump({
      channel: steerChannel,
      signal: AbortSignal.any([signal, steerPumpAbort.signal]),
      waitForCue: deps.waitForCue!,
      drainCues: () => (deps.drainCues!() ?? []).map((cue) => ({ itemId: cue.itemId, text: cue.text })),
      onSteered: (cue) => {
        // Persist delivered cues so forks and exports retain what the delegated engine received.
        thread.apiMessages.push({
          role: 'user',
          content: `[Live steer from the user, delivered into the running engine turn. It has already been applied.]\n${cue.text}`,
        })
        send({ type: 'item_update', id: cue.itemId, patch: { state: 'ok', meta: 'steered' } })
        persist()
      },
      onRejected: (cue) => {
        send({ type: 'item_update', id: cue.itemId, patch: { state: 'denied', meta: 'not applied — send again' } })
      },
    })
  }

  const homegrownSteerEligible = !steerChannel && !!deps.drainCues && !!deps.waitForCue
  let steerRestartsThisTurn = 0
  let homegrownSteer: HomegrownSteerWatcher | undefined

  for (let attempt = 1; ; attempt++) {
    if (signal.aborted) break
    if (deadlineAt && Date.now() >= deadlineAt) { state.budgetStop = 'time budget'; break }

    if (assistantItem) send({ type: 'item_update', id: assistantItem.id, patch: { text: '' } })
    if (reasoningItem) send({ type: 'item_update', id: reasoningItem.id, patch: { text: '' } })
    assistantItem = null
    reasoningItem = null
    content = ''
    reasoning = ''
    toolCalls = []
    finishReason = null
    loopBroken = false
    let lastLoopCheckLen = 0

    const streamAbort = new AbortController()

    const streamDeadline = deadlineAt
      ? AbortSignal.timeout(Math.max(1, Math.min(2_147_483_647, Math.ceil(deadlineAt - Date.now()))))
      : undefined
    const requestSignal = AbortSignal.any([signal, streamAbort.signal, ...(streamDeadline ? [streamDeadline] : [])])

    homegrownSteer = homegrownSteerEligible && steerRestartsThisTurn < MAX_STEER_RESTARTS
      ? createHomegrownSteerWatcher({
          signal: AbortSignal.any([signal, streamAbort.signal]),
          waitForCue: deps.waitForCue!,
          drainCues: () => (deps.drainCues!() ?? []).map((cue) => ({ itemId: cue.itemId, text: cue.text })),
          onSteer: () => { streamAbort.abort() },
        })
      : undefined

    const recoveryOperationId = `model:${step}:${attempt}`
    let recoveryOperationStarted = false
    const apiStartedAt = Date.now()
    try {
      // Begin delivery accounting only after preparation and live request guards succeed.
      const preparedMessages = messagesForModel()
      const preparedTools = modelToolSchemas()
      recovery.begin(recoveryOperationId)
      recoveryOperationStarted = true
      recovery.requestStarted()
    send({ type: 'lifecycle', phase: 'model_start', ts: Date.now(), step })
    callbacksAllowed() && deps.observe?.('api.request', { provider: provider.id, model, attempt })
    const stream = streamChat({
      assertRequestAllowed,
      provider,
      baseUrl: provider.baseUrl,
      apiKey: provider.apiKey,
      model,
      requireProviderFinish: true,
      messages: preparedMessages,
      tools: preparedTools,
      temperature: deps.temperature,
      reasoningEffort: deps.reasoningEffort,
      serverReasoningEffort: serverEffort,
      serviceTier: deps.serviceTier,
      approvalMode: deps.approvalMode,

      threadId: thread.id,
      codexResumeCursor: thread.codexResumeCursor,
      claudeResumeCursor: thread.claudeResumeCursor,
      openCodeResumeCursor: thread.openCodeResumeCursor,
      piResumeCursor: thread.piResumeCursor,
      cursorResumeCursor: thread.cursorResumeCursor,
      steering: steerChannel,
      cwd,
      signal: requestSignal,
    })

      for await (const chunk of stream) {
        if (signal.aborted) break
        if (chunk.type === 'content' || chunk.type === 'reasoning' || chunk.type === 'tool_calls') recovery.requestOutput()
        if (streamDeadline?.aborted) { state.budgetStop = 'time budget'; break }
        if (chunk.type === 'codex_resume_cursor') {
          thread.codexResumeCursor = { threadId: chunk.threadId }
          continue
        }
        if (chunk.type === 'claude_resume_cursor') {
          thread.claudeResumeCursor = { sessionId: chunk.sessionId }
          continue
        }
        if (chunk.type === 'opencode_resume_cursor') {
          thread.openCodeResumeCursor = { sessionId: chunk.sessionId }
          continue
        }
        if (chunk.type === 'pi_resume_cursor') {
          thread.piResumeCursor = { sessionId: chunk.sessionId }
          continue
        }
        if (chunk.type === 'cursor_resume_cursor') {
          thread.cursorResumeCursor = { sessionId: chunk.sessionId }
          continue
        }
        if (chunk.type === 'notice') {
          send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: chunk.text, createdAt: Date.now() } })
          continue
        }
        if (chunk.type === 'provider_rate_limits') {
          deps.onProviderRateLimits?.({ providerId: provider.id, accountId: thread.providerAccountId ?? null, snapshot: chunk.snapshot })
          continue
        }
        if (chunk.type === 'usage') {
          recordModelCall(turnStats, chunk)
          continue
        }
        if (chunk.type === 'reasoning') {
          if (!reasoningItem) {
            reasoningItem = { id: randomUUID(), kind: 'reasoning', text: '', createdAt: Date.now() }
            send({ type: 'item_add', item: reasoningItem })
          }
          reasoning += chunk.text
          send({ type: 'item_delta', id: reasoningItem.id, field: 'text', chunk: chunk.text })
        } else if (chunk.type === 'content') {
          if (!assistantItem) {
            assistantItem = { id: randomUUID(), kind: 'assistant', text: '', createdAt: Date.now() }
            state.lastAssistantId = assistantItem.id
            send({ type: 'item_add', item: assistantItem })
            send({ type: 'status', status: 'streaming' })
          }
          content += chunk.text
          send({ type: 'item_delta', id: assistantItem.id, field: 'text', chunk: chunk.text })
        } else if (chunk.type === 'tool_calls') {
          toolCalls = chunk.toolCalls
        } else if (chunk.type === 'finish') {
          finishReason = chunk.reason
        }

        const grown = content.length + reasoning.length
        if (grown >= LOOP_CHECK_MIN && grown - lastLoopCheckLen >= LOOP_CHECK_STRIDE) {
          lastLoopCheckLen = grown
          if (textLoopBreakerEnabled && (detectTextLoop(content) || detectTextLoop(reasoning))) {
            loopBroken = true
            streamAbort.abort()
            break
          }
        }
      }
      if (!signal.aborted && !loopBroken && !streamDeadline?.aborted) recovery.requestComplete()

      callbacksAllowed() && deps.observe?.('api.response', { provider: provider.id, model, attempt, ok: !signal.aborted, ms: Date.now() - apiStartedAt, chars: content.length + reasoning.length })
      break
    } catch (e: any) {
      if (streamDeadline?.aborted && !signal.aborted) { state.budgetStop = 'time budget'; break }

      if (homegrownSteer?.requested && !signal.aborted) {
        const cues = homegrownSteer.cues
        homegrownSteer = undefined
        if (cues.length) {
          steerRestartsThisTurn++
          send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: `Restarting turn to apply your guide${steerRestartsThisTurn > 1 ? ` (${steerRestartsThisTurn}×)` : ''}…`, createdAt: Date.now() } })

          thread.apiMessages.push({
            role: 'user',
            content: `[Live guide from the user, queued while this run was active. Apply it now and continue the same run.]\n${cues.map((c) => c.text).join('\n')}`,
          })
          for (const cue of cues) send({ type: 'item_update', id: cue.itemId, patch: { state: 'ok', meta: 'steered · turn restarted' } })
          persist()

          attempt = 0
          continue
        }
      }

      if (loopBroken && !signal.aborted) break

      if (signal.aborted) throw e

      const delivery = recoveryOperationStarted ? failedRequestDelivery(e, !!(content || reasoning || toolCalls.length)) : 'not_sent'
      recovery.requestFailed(delivery)
      if (!recoveryOperationStarted) throw e
      const action = streamRetryDecision(e, attempt, { compactRetries, canCompact: autoCompact && contextLimit > 0, delegated: isDelegatedTransport(provider.transport), acceptance: { delivery, visibleOutput: !!(content || reasoning || toolCalls.length) }, effortRetried: serverEffort != null })

      if (action.kind === 'effort') {
        serverEffort = action.effort
        serverEfforts.set(effortKey, action.effort)
        send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: `The server refused that reasoning effort; sending ${action.effort}, the nearest level it takes.`, createdAt: Date.now() } })
        attempt = 0
        continue
      }

      if (action.kind === 'compact') {
        compactRetries++
        const noticeId = randomUUID()
        send({ type: 'item_add', item: { id: noticeId, kind: 'notice', text: 'Context too large for the model — compacting and retrying…', createdAt: Date.now() } })
        send({ type: 'status', status: 'thinking' })
        const beforeTokens = estimateTokens(thread.apiMessages)
        let entry: CompactionLedgerEntry | null = null
        try { entry = await compactThread(thread, provider, model, requestSignal, 'auto', deps.onMemoryHarvest, memoryPolicy, assertRequestAllowed) } catch {}
        if (streamDeadline?.aborted && !signal.aborted) { state.budgetStop = 'time budget'; break }
        const afterTokens = estimateTokens(thread.apiMessages)
        // Compaction must shrink the actual request before an overflow retry can proceed.
        if (!compactionMadeProgress(entry, beforeTokens, afterTokens)) {
          send({ type: 'item_update', id: noticeId, patch: { text: 'Context overflow could not be compacted safely. The turn was not retried with the same oversized payload.' } })
          persist()
          throw e
        }
        send({ type: 'item_update', id: noticeId, patch: { text: `Context compacted after an overflow (~${kfmt(beforeTokens)} -> ~${kfmt(afterTokens)} tokens) — retrying the turn.` } })
        persist()
        if (signal.aborted) break
        continue
      }

      if (action.kind === 'retry') {
        send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: `Provider hiccup (${action.reason}) — retrying in ${(action.delayMs / 1000).toFixed(1)}s (attempt ${attempt} of ${STREAM_MAX_RETRIES}).`, createdAt: Date.now() } })
        send({ type: 'status', status: 'thinking' })
        await abortableDelay(action.delayMs, requestSignal)
        if (signal.aborted) break

        if (streamDeadline?.aborted) { state.budgetStop = 'time budget'; break }
        continue
      }

      if (content) thread.apiMessages.push({ role: 'assistant', content })
      callbacksAllowed() && deps.observe?.('api.response', { provider: provider.id, model, attempt, ok: false, ms: Date.now() - apiStartedAt, message: String((e as Error)?.message ?? e) })
      steerPumpAbort.abort()
      throw e
    } finally {
      if (recoveryOperationStarted) recovery.end(recoveryOperationId)

      // Retire watchers on every exit path and return unclaimed cues to the conversation.
      if (homegrownSteer) {
        homegrownSteer.stop()

        const unclaimedSteerCues = (homegrownSteer as any)?.unclaimedCues as AgentCue[] | undefined
        if (unclaimedSteerCues?.length) {
          for (const cue of unclaimedSteerCues) {
            thread.apiMessages.push({
              role: 'user',
              content: `[Live guide from the user, queued while this run was active. Apply it now and continue the same run.]\n${cue.text}`,
            })
            send({ type: 'item_update', id: cue.itemId, patch: { state: 'ok', meta: 'applied · turn ended first' } })
          }
          persist()
        }
      }
    }
  }

  steerPumpAbort.abort()
  send({ type: 'lifecycle', phase: 'model_end', ts: Date.now(), step })
  return { assistantItem, reasoningItem, content, reasoning, toolCalls, finishReason, loopBroken }
}

export type ModelTurn = Awaited<ReturnType<typeof runModelTurn>>
