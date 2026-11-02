// Coordinate the run lifecycle; turn handlers share one captured session and mutable state.

import { randomUUID } from 'node:crypto'
import { budgetPauseReason } from './run-recovery/policy'
import { midRunTodoNudge } from './todo-tracker'
import { estimateTokens, kfmt } from './compaction'
import { type ApiMessage, type TimelineItem } from './types'
import { type RunDeps } from './run-loop-contracts'
import { reconcileOrphanedToolCalls } from './run-loop-history'
import { MAX_TEXT_LOOP_BREAKS } from './run-loop-guards'
import { compactThread } from './run-loop-compaction'
import { createRunSession } from './run-loop-session'
import { runGraceTurn } from './run-loop-summary'
import { runModelTurn } from './run-loop-stream'
import { completeTextTurn } from './run-loop-completion'
import { runToolTurn } from './run-loop-tool-turn'

export async function runAgent(input: RunDeps): Promise<void> {
  const session = await createRunSession(input)
  const {
    deps, state, thread, provider, model, signal, persist, memoryPolicy, assertPhonePrivacy,
    assertRequestAllowed, pluginRuntime, send, recovery, onStopRequested, autoCompact, contextLimit, maxSteps,
    deadlineAt, drainLiveCues, attachTurnStats,
  } = session
  let runOk = false
  signal.addEventListener('abort', onStopRequested, { once: true })
  recovery.start()
  if (signal.aborted) recovery.stopRequested()
  try {
    pluginRuntime.emitSessionStart(thread.id)
    send({ type: 'lifecycle', phase: 'run_start', ts: Date.now() })
    for (let step = 0; step < maxSteps; step++) {
      if (signal.aborted) break
      assertPhonePrivacy()
      drainLiveCues()
      const todoNudge = midRunTodoNudge(state.todoMutationsSincePlanTouch, thread.plan ?? [], state.todoNudges)
      if (todoNudge) {
        state.todoNudges++
        state.todoMutationsSincePlanTouch = 0
        send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', title: 'Plan check', text: todoNudge, createdAt: Date.now() } })
        thread.apiMessages.push({ role: 'user', content: `[plan check] ${todoNudge}` })
        persist()
      }
      if (deadlineAt && Date.now() > deadlineAt) { state.budgetStop = 'time budget'; break }
      send({ type: 'lifecycle', phase: 'turn_start', ts: Date.now(), step })

      if (autoCompact && contextLimit > 0) {
        const est = estimateTokens(thread.apiMessages)
        if (est > contextLimit) {
          const noticeId = randomUUID()
          send({ type: 'item_add', item: { id: noticeId, kind: 'notice', text: 'Compacting context to continue…', createdAt: Date.now() } })
          send({ type: 'status', status: 'thinking' })
          try {
            const entry = await compactThread(thread, provider, model, signal, 'auto', deps.onMemoryHarvest, memoryPolicy, assertRequestAllowed)
            const after = estimateTokens(thread.apiMessages)
            send({ type: 'item_update', id: noticeId, patch: { text: entry ? `Context compacted. Folded ${entry.messagesFolded} earlier messages into a summary (~${kfmt(est)} -> ~${kfmt(after)} tokens).` : `Context at ~${kfmt(est)} tokens (limit ${kfmt(contextLimit)}).` } })
          } catch {
            send({ type: 'item_update', id: noticeId, patch: { text: `Context at ~${kfmt(est)} tokens. Compaction skipped.` } })
          }
          persist()
          if (signal.aborted) break
        }
      }

      send({ type: 'status', status: 'thinking' })

      const turn = await runModelTurn(session, step)
      const { assistantItem, reasoningItem, content, reasoning, toolCalls, finishReason, loopBroken } = turn

      if (state.budgetStop === 'time budget') {
        if (assistantItem) send({ type: 'item_update', id: assistantItem.id, patch: { text: content, meta: 'stopped — time limit' } })
        if (reasoningItem) send({ type: 'item_update', id: reasoningItem.id, patch: { text: reasoning, meta: 'stopped — time limit' } })
        if (content) thread.apiMessages.push({ role: 'assistant', content })
        persist()
        break
      }

      if (reasoningItem) send({ type: 'item_update', id: reasoningItem.id, patch: { text: reasoning } })

      if (loopBroken) {
        state.textLoopBreaks++
        const head = content.slice(0, 1200)
        if (assistantItem) send({ type: 'item_update', id: assistantItem.id, patch: { text: head, meta: 'stopped — repeating itself' } })
        if (reasoningItem) send({ type: 'item_update', id: reasoningItem.id, patch: { text: reasoning.slice(0, 1200), meta: 'stopped — repeating itself' } })
        thread.apiMessages.push({ role: 'assistant', content: head || '[stopped: the response began repeating itself]' })
        if (state.textLoopBreaks >= MAX_TEXT_LOOP_BREAKS) {
          recovery.pause('repeated_output', 'The model repeatedly emitted the same text; this run is paused and its existing budgets have not been reset.')
          send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: 'The model kept repeating itself, so the run was stopped. Try a different (or larger) model, or rephrase the request.', createdAt: Date.now() } })
          state.completedNaturally = true; persist(); break
        }
        send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: 'Caught the model repeating itself and stopped that turn. Nudging it to answer concisely…', createdAt: Date.now() } })
        thread.apiMessages.push({ role: 'user', content: '[guardrail] Your previous response began repeating the same lines over and over and was cut off. Do NOT repeat yourself. Give the user a single, concise final answer now. If you genuinely need more information, call exactly one tool — do not restate a checklist, plan, or your reasoning.' })
        send({ type: 'status', status: 'thinking' })
        persist(); continue
      }

      if (assistantItem) send({ type: 'item_update', id: assistantItem.id, patch: { text: content } })

      const assistantMsg: ApiMessage = { role: 'assistant', content: content || (toolCalls.length ? null : '') }
      if (toolCalls.length) assistantMsg.tool_calls = toolCalls
      thread.apiMessages.push(assistantMsg)

      if (signal.aborted) break

      if (!toolCalls.length) {
        if (await completeTextTurn(session, step, turn) === 'continue') continue
        break
      }

      await runToolTurn(session, step, toolCalls, finishReason)
      if (state.repeatedActionStop || state.uncertainActionStop || state.budgetStop) break
    }

    if (!signal.aborted && !state.uncertainActionStop) {
      const reason = state.budgetStop ?? (!state.completedNaturally ? 'step budget' : null)
      if (reason) {
        recovery.pause(budgetPauseReason(reason), `Run paused after reaching its ${reason}; completed work is preserved and continuing requires an explicit new request.`)
        await runGraceTurn(session, reason)
      }
    }

    attachTurnStats()
    send({ type: 'status', status: 'done' })
    send({ type: 'done' })
    runOk = true
  } catch (e: any) {
    attachTurnStats()
    if (signal.aborted) { send({ type: 'status', status: 'done' }); send({ type: 'done' }); runOk = true; return }
    const message = e?.message ? String(e.message) : String(e)
    recovery.fail(message)
    const errItem: TimelineItem = { id: randomUUID(), kind: 'error', text: message, createdAt: Date.now() }
    send({ type: 'item_add', item: errItem })
    send({ type: 'status', status: 'error' })
    send({ type: 'error', message })
  } finally {
    signal.removeEventListener('abort', onStopRequested)
    recovery.settle(runOk)

    // Repair every interrupted batch before persistence so the next provider request remains valid.
    thread.apiMessages = reconcileOrphanedToolCalls(thread.apiMessages)
    persist()
    send({ type: 'lifecycle', phase: 'run_end', ts: Date.now(), ok: runOk })
    void pluginRuntime.queueSessionEnd(thread.id)
  }
}
