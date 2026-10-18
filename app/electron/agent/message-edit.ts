import { createHash, randomUUID } from 'node:crypto'
import type { ApiMessage, Provider, Thread } from './types'
import type { ThreadMessageEditInput, ThreadMessageEditPreparation } from '../../shared/thread-message-edit'
import { buildUserContent } from './content'

export function apiUserMessageRef(messages: ApiMessage[], content: ApiMessage['content'], compactionCount: number) {
  return { ordinal: messages.filter((message) => message.role === 'user').length, fingerprint: fingerprint(content), compactionCount }
}
function fingerprint(value: unknown): string { return createHash('sha256').update(JSON.stringify(value)).digest('hex') }
function revision(thread: Thread): number {
  const value = thread.conversationRevision ?? 0
  if (!Number.isSafeInteger(value) || value < 0 || value >= Number.MAX_SAFE_INTEGER) throw new Error('This chat has invalid history metadata. Start a new chat.')
  return value
}
export function threadMessageEditToken(thread: Thread): string {
  return fingerprint({ timeline: thread.timeline, apiMessages: thread.apiMessages, revision: revision(thread), compactions: thread.compactions ?? [], providerId: thread.providerId, providerAccountId: thread.providerAccountId, model: thread.model, engine: thread.engine, incognito: thread.incognito })
}
export function assertThreadMessageEditable(thread: Thread, provider?: Provider): void {
  if (thread.archivedAt) throw new Error('Restore this chat before editing a message.')
  if (!['chat', 'task'].includes(thread.kind) || !['idle', 'error', 'running'].includes(thread.status)) throw new Error('This conversation has incomplete history metadata. Reload it or start a new chat before editing.')
  if (thread.status === 'running') throw new Error('Stop the active run before editing a sent message.')
  if (thread.orchestratorHostId || (thread.workspace?.kind === 'ssh')) throw new Error('Edit this conversation on its owning host.')
  if ((thread.engine && thread.engine !== 'native') || (provider?.transport && provider.transport !== 'chat-completions') || thread.codexResumeCursor || thread.claudeResumeCursor || thread.openCodeResumeCursor || thread.piResumeCursor || thread.cursorResumeCursor || thread.ompResumeCursor) {
    throw new Error('This provider owns a separate session history. Start a new chat with an API model to edit messages safely.')
  }
}
function resolveMessage(thread: Thread, itemId: string): { timelineIndex: number; apiIndex: number } {
  const timelineIndex = thread.timeline.findIndex((item) => item.id === itemId)
  const item = thread.timeline[timelineIndex]
  if (!item || item.kind !== 'user' || item.memorySource === 'external') throw new Error('Choose one of your own sent messages to edit.')
  const content = buildUserContent(item.text ?? '', item.attachments, item.quotedMessages)
  const userMessages = thread.apiMessages.map((message, index) => ({ message, index })).filter(({ message }) => message.role === 'user')
  if (item.apiMessageRef) {
    const ref = item.apiMessageRef
    if (ref.compactionCount !== (thread.compactions?.length ?? 0)) throw new Error('This message predates a history compaction. Start a new chat or fork a recorded turn to retain the correct context.')
    const found = userMessages[ref.ordinal]
    if (!found || fingerprint(found.message.content) !== ref.fingerprint || fingerprint(content) !== ref.fingerprint) throw new Error('This message no longer matches the model history. Reload the chat and try again, or start a new chat.')
    return { timelineIndex, apiIndex: found.index }
  }
  // Conservative compatibility: every user turn must align, including attachments.
  // Summaries, injected user handoffs and incomplete legacy logs cannot be guessed.
  const timelineUsers = thread.timeline.filter((candidate) => candidate.kind === 'user')
  if ((thread.compactions?.length ?? 0) > 0 || timelineUsers.length !== userMessages.length || timelineUsers.some((candidate, index) => fingerprint(buildUserContent(candidate.text ?? '', candidate.attachments, candidate.quotedMessages)) !== fingerprint(userMessages[index].message.content))) {
    throw new Error('This older message cannot be matched safely to the model history. Start a new chat or fork a recorded turn.')
  }
  return { timelineIndex, apiIndex: userMessages[timelineUsers.indexOf(item)].index }
}
export function prepareThreadMessageEdit(thread: Thread, itemId: string): ThreadMessageEditPreparation {
  assertThreadMessageEditable(thread)
  const { timelineIndex } = resolveMessage(thread, itemId)
  return { itemId, expectedConversationRevision: revision(thread), expectedHistoryToken: threadMessageEditToken(thread), expectedMessageText: thread.timeline[timelineIndex].text ?? '', affectedItems: thread.timeline.length - timelineIndex - 1, mode: thread.kind === 'task' ? 'branch-task' : 'replace-chat' }
}

export function stageThreadMessageEdit(thread: Thread, input: ThreadMessageEditInput, now = Date.now()): Partial<Thread> {
  const preparation = prepareThreadMessageEdit(thread, input.itemId)
  if (input.threadId !== thread.id || input.expectedConversationRevision !== preparation.expectedConversationRevision || input.expectedHistoryToken !== preparation.expectedHistoryToken || input.expectedMessageText !== preparation.expectedMessageText) throw new Error('The conversation changed while you were editing. Cancel, reopen Edit, and review the current messages.')
  if (typeof input.text !== 'string' || input.text.length > 200_000) throw new Error('The edited message must be text of at most 200,000 characters.')
  const { timelineIndex, apiIndex } = resolveMessage(thread, input.itemId)
  const original = thread.timeline[timelineIndex]
  if (!input.text.trim() && !original.attachments?.length) throw new Error('Enter a message or keep an attachment before saving.')
  if (input.text === preparation.expectedMessageText) return {}
  const revisions = thread.messageEditRevisions ?? []
  if (revisions.length >= 100) throw new Error('This chat has reached its saved edit-history limit. Start a new chat to retain all earlier revisions.')
  const edited = { ...original, text: input.text, apiMessageRef: apiUserMessageRef(thread.apiMessages.slice(0, apiIndex), buildUserContent(input.text, original.attachments, original.quotedMessages), thread.compactions?.length ?? 0) }
  const newRevision = revision(thread) + 1
  const archive = { id: randomUUID(), itemId: input.itemId, editedAt: now, conversationRevision: revision(thread), timeline: structuredClone(thread.timeline.slice(timelineIndex)), apiMessages: structuredClone(thread.apiMessages.slice(apiIndex)) }
  const nextRevisions = [...revisions, archive]
  if (JSON.stringify(nextRevisions).length > 64 * 1024 * 1024) throw new Error('Saving this edit would exceed the protected edit-history limit. Start a new chat to preserve the original conversation.')
  return {
    timeline: [...thread.timeline.slice(0, timelineIndex), edited, { id: randomUUID(), kind: 'notice', text: `Message edited. ${preparation.affectedItems} later conversation ${preparation.affectedItems === 1 ? 'item is' : 'items are'} preserved in earlier revisions and excluded from future model context. File changes, saved memory, sent messages and other external actions were not undone.`, createdAt: now }],
    apiMessages: [...thread.apiMessages.slice(0, apiIndex), { ...thread.apiMessages[apiIndex], content: buildUserContent(input.text, original.attachments, original.quotedMessages) }],
    messageEditRevisions: nextRevisions,
    conversationRevision: newRevision,
    // Receipt/context facts describe the superseded run, not the edited conversation.
    activeRunContract: null, lastRunReceipt: null, contextFragments: [], diffReviewedAt: null,
    status: 'idle', updatedAt: now,
  }
}

/** New identity means old file checkpoints can never be mistaken for this edited turn. */
export function createEditedTaskBranch(source: Thread, patch: Partial<Thread>, itemId: string, now = Date.now()): Thread {
  const branch: Thread = {
    id: randomUUID(), title: `Edited: ${source.title}`, kind: 'task',
    memoryPolicy: source.memoryPolicy ? { ...source.memoryPolicy } : undefined,
    memoryPolicyRevision: 0,
    ...(source.bot ? { bot: frozenBotSnapshot(source.bot) } : {}),
    ...(source.phoneLocalOnly ? { phoneLocalOnly: true } : {}),
    ...(source.googleLocalOnly ? { googleLocalOnly: true } : {}),
    projectId: source.projectId ?? null, cwd: source.cwd, workspace: source.workspace ? structuredClone(source.workspace) : null,
    ...(source.workspaceBoundary ? { workspaceBoundary: source.workspaceBoundary } : {}),
    parentId: source.id, forkedAt: now,
    providerId: source.providerId, providerAccountId: source.providerAccountId ?? null, model: source.model,
    ...(source.modelRoute ? { modelRoute: structuredClone(source.modelRoute) } : {}),
    createdAt: now, updatedAt: now, status: 'idle',
    apiMessages: structuredClone(patch.apiMessages ?? source.apiMessages),
    timeline: structuredClone(patch.timeline ?? source.timeline),
    conversationRevision: 0,
    messageEditRevisions: structuredClone(patch.messageEditRevisions ?? []),
    messageEditSource: { threadId: source.id, itemId, editedAt: now },
    ...(source.goal ? { goal: source.goal } : {}),
    ...(source.goalStatus ? { goalStatus: source.goalStatus } : {}),
    ...(source.plan ? { plan: structuredClone(source.plan) } : {}),
    ...(source.notepad ? { notepad: source.notepad } : {}),
    ...(source.incognito ? { incognito: true } : {}),
    ...(source.compactions ? { compactions: structuredClone(source.compactions) } : {}),
    activeRunContract: null, lastRunReceipt: null, systemPromptSnapshot: null,
  }
  branch.timeline.push({ id: randomUUID(), kind: 'notice', text: 'Edited branch created. The original conversation and its checkpoints are unchanged. This branch uses the current workspace files; no files or external actions were rewound.', createdAt: now })
  return branch
}

function frozenBotSnapshot(bot: NonNullable<Thread['bot']>): NonNullable<Thread['bot']> {
  const snapshot = structuredClone(bot)
  const seen = new WeakSet<object>()
  const freeze = (value: unknown): void => {
    if (!value || typeof value !== 'object' || seen.has(value)) return
    seen.add(value)
    for (const child of Object.values(value)) freeze(child)
    Object.freeze(value)
  }
  freeze(snapshot)
  return snapshot
}
