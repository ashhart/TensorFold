// Decide whether a text-only turn completes, continues, or needs fresh verification.

import { randomUUID } from 'node:crypto'
import { stopTodoReminder } from './todo-tracker'
import { textTruncationAction, verifyOnStopSnapshot } from './run-loop-guards'
import { MAX_GOAL_ITERS, verifyGoal } from './run-loop-auxiliary'
import { type RunSession } from './run-loop-session'
import { type ModelTurn } from './run-loop-stream'

export async function completeTextTurn(session: RunSession, step: number, turn: ModelTurn): Promise<'continue' | 'break'> {
  const {
    deps, state, thread, provider, model, signal, persist, assertRequestAllowed, send, isChat, planningOnly,
    guardPolicy, verifyTimelineStart, drainLiveCues,
  } = session
  const { content, toolCalls, finishReason } = turn
  send({ type: 'lifecycle', phase: 'turn_end', ts: Date.now(), step })
  const truncation = textTruncationAction(finishReason, toolCalls.length, state.textLengthContinuations)
  if (truncation === 'continue') {
    state.textLengthContinuations++
    send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: `The model hit its output limit before finishing. Continuing the same answer (${state.textLengthContinuations}/2)…`, createdAt: Date.now() } })
    thread.apiMessages.push({
      role: 'user',
      content: '[continuation required] Your previous text response ended because the maximum output length was reached. Continue exactly where it stopped. Do not repeat the completed portion, do not claim the task is done prematurely, and keep this continuation concise.',
    })
    persist()
    return 'continue'
  }
  if (truncation === 'fail') {
    throw new Error('The model response hit its output limit repeatedly and could not finish. Use a larger output limit or ask it to split the work into smaller parts.')
  }
  if (drainLiveCues() > 0) return 'continue'
  if (deps.waitForCue) {
    await deps.waitForCue(250, signal)
    if (drainLiveCues() > 0) return 'continue'
  }
  if (guardPolicy?.verifyOnStop) {
    const verification = verifyOnStopSnapshot(thread.timeline, verifyTimelineStart)
    if (verification.needsVerification && state.verifyNudges < 2 && verification.lastChangeIndex > state.lastVerifyNudgedChangeIndex) {
      state.verifyNudges++
      state.lastVerifyNudgedChangeIndex = verification.lastChangeIndex
      const fileList = verification.changedFiles.slice(0, 8).join(', ')
      send({
        type: 'item_add',
        item: {
          id: randomUUID(),
          kind: 'notice',
          title: 'Verify-on-stop',
          text: `Code changes need fresh verification before this run can conclude${fileList ? `: ${fileList}` : '.'}`,
          state: 'awaiting',
          createdAt: Date.now(),
        },
      })
      thread.apiMessages.push({
        role: 'user',
        content: `[verify-on-stop] You changed code in this run (${fileList || 'files changed'}). Before giving a final answer, run an appropriate fresh verification command such as tests, typecheck, lint, build, or verify. If verification cannot run, take one concrete repair/blocker action and state the blocker explicitly instead of claiming success.`,
      })
      persist()
      return 'continue'
    }
  }
  if (!isChat && !planningOnly && thread.goal && thread.goalStatus === 'active' && state.goalIters < MAX_GOAL_ITERS) {
    state.goalIters++
    send({ type: 'status', status: 'thinking' })
    const v = await verifyGoal(provider, model, thread, signal, assertRequestAllowed)

    if (signal.aborted) return 'break'
    if (v.done) {
      thread.goalStatus = 'complete'
      for (const p of thread.plan ?? []) p.done = true
      send({ type: 'plan', plan: thread.plan ?? [], goalStatus: 'complete' })
      send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: `Goal complete. ${v.note || 'Verified.'}`, createdAt: Date.now() } })
      state.completedNaturally = true; persist(); return 'break'
    }
    send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', text: `Goal not met yet. Continuing: ${v.note}`, createdAt: Date.now() } })
    thread.apiMessages.push({ role: 'user', content: `[goal check] Not done yet: ${v.note}\nKeep working toward the goal, update the plan, and don't stop until it is fully achieved.` })
    persist(); return 'continue'
  }

  if (!thread.goal && !planningOnly) {
    const reminder = stopTodoReminder(thread.plan ?? [], state.todoStopReminders, content)
    if (reminder) {
      state.todoStopReminders++
      state.todoMutationsSincePlanTouch = 0
      send({ type: 'item_add', item: { id: randomUUID(), kind: 'notice', title: 'Open plan items', text: `Stopped with ${reminder.open.length} open plan item${reminder.open.length === 1 ? '' : 's'}; asking the agent to finish them or update the plan.`, createdAt: Date.now() } })
      thread.apiMessages.push({ role: 'user', content: `[plan check] ${reminder.message}` })
      persist()
      return 'continue'
    }
  }
  state.completedNaturally = true; persist(); return 'break'
}
