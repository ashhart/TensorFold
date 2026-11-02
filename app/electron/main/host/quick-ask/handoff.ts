import { randomUUID } from 'node:crypto'
import type { Thread, TimelineItem } from '../../../agent/types'
import type { AskSavedState } from '../../../agent/quick-ask/session'
import { askUserMessage } from '../../../agent/quick-ask/session'
import { validateAskSavedState } from '../../../agent/quick-ask/state-validation'
import { reserveAskOperation } from './admission'
import { store } from '../../state'
import { saveThreads } from '../../../store'
import { sendToRenderer } from '../renderer-events'
import { requireSecurityRole } from '../audit'

const pending = new Map<string, Promise<Thread>>()
export async function handoffAsk(state: AskSavedState): Promise<Thread> {
  requireSecurityRole('operator', 'continue an ask in TensorFold')
  validateAskSavedState(state)
  const key = state.session.id, existing = pending.get(key)
  if (existing) return existing
  const operation = createAskThread(state)
  pending.set(key, operation)
  try { return await operation } finally { pending.delete(key) }
}
async function createAskThread(state: AskSavedState): Promise<Thread> {
  requireSecurityRole('operator', 'continue an ask in TensorFold')
  validateAskSavedState(state)
  const session = state.session, route = session.route
  if (!route || !session.turns.length) throw new Error('Ask a question before continuing in TensorFold.')
  const existing = store.threads.find(thread => thread.id === `ask-${session.id}`)
  if (existing) return existing
  const timeline: TimelineItem[] = []
  for (const turn of session.turns) {
    timeline.push({ id: `${turn.id}-user`, kind: 'user', text: turn.text, attachments: turn.attachments, createdAt: turn.at })
    if (turn.reasoning) timeline.push({ id: `${turn.id}-reasoning`, kind: 'reasoning', text: turn.reasoning, createdAt: turn.at })
    if (turn.answer) timeline.push({ id: `${turn.id}-answer`, kind: 'assistant', text: turn.answer, createdAt: turn.at })
    for (const tool of turn.tools) timeline.push({ id: randomUUID(), kind: 'notice', text: `${tool.name}: ${tool.detail} · ${tool.status}`, createdAt: turn.at })
    if (turn.error) timeline.push({ id: `${turn.id}-notice`, kind: 'notice', text: turn.error, createdAt: turn.at })
  }
  const last = session.turns.at(-1)!
  const messages = state.messages.length ? state.messages : [askUserMessage(last.text, last.attachments), ...(last.answer ? [{ role: 'assistant' as const, content: last.answer }] : [])]
  const now = Date.now()
  const thread: Thread = { id: `ask-${session.id}`, kind: 'chat', title: session.turns[0].text.trim().slice(0, 80) || 'Ask TensorFold', cwd: '', workspace: null, projectId: null, documentSourceIds: [], providerId: route.providerId, providerAccountId: route.accountId, model: route.model, createdAt: session.turns[0].at, updatedAt: now, status: 'idle', timeline,
    apiMessages: [{ role: 'system', content: session.prompt ?? 'Answer clearly and directly.' }, ...structuredClone(messages)] }
  const lease = reserveAskOperation()
  store.threads.unshift(thread)
  try { await saveThreads(store.threads) }
  catch (error) { store.threads = store.threads.filter(item => item !== thread); throw error }
  finally { lease.release() }
  sendToRenderer('thread:update', thread)
  return thread
}
