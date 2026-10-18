import { createHash } from 'node:crypto'
import type { InputHistoryEntry, InputHistoryScope, InputHistorySnapshot } from '../../../shared/input-history'
import { validateComposerDraftScope } from '../../../shared/composer-drafts'
import type { Thread } from '../../agent/types'
import { vontraPath } from '../../store/paths'
import { inputHistoryClearState, clearInputHistoryState } from '../../store/input-history'
import { store } from '../state'

const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex')
function ordinary(thread: Thread): boolean {
  return !thread.incognito && !thread.bot && !thread.phoneLocalOnly && !thread.googleLocalOnly && !thread.workspaceBoundary && !thread.orchestratorHostId && thread.workspace?.kind !== 'ssh' && !thread.archivedAt
}
function workspace(thread: Pick<Thread, 'projectId' | 'cwd' | 'workspace'>): string {
  const project = thread.projectId ? store.projects.find(row => row.id === thread.projectId) : undefined
  if (thread.projectId && !project) throw new Error('This workspace is unavailable.')
  if (project?.workspace?.kind === 'ssh' || project?.orchestratorId) throw new Error('Remote input history is unavailable on this desktop.')
  return JSON.stringify([thread.projectId ?? null, thread.workspace?.path || thread.cwd || project?.workspace?.path || project?.cwd || ''])
}
function account(providerId: string | null, accountId: string | null): void {
  const provider = store.settings.providers.find(row => row.id === providerId)
  if (!provider || accountId !== null && !provider.accounts?.some(row => row.id === accountId)) throw new Error('This model account is unavailable. Reopen input history after selecting an account.')
}
export function inputHistoryOwner(raw: InputHistoryScope): { key: string; eligible: (thread: Thread) => boolean } {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw) || Object.keys(raw).some(key => !['draft', 'projectId', 'cwd', 'providerId', 'accountId'].includes(key))) throw new Error('Invalid input history scope.')
  const draft = validateComposerDraftScope(raw.draft)
  for (const key of ['projectId', 'providerId', 'accountId'] as const) if (raw[key] !== null && (typeof raw[key] !== 'string' || !raw[key].trim() || raw[key]!.length > 320)) throw new Error('Invalid input history identity.')
  if (typeof raw.cwd !== 'string' || raw.cwd.length > 16_384) throw new Error('Invalid input history workspace.')
  if (draft.incognito || draft.surface === 'assistant' || draft.hostId !== null) throw new Error('Input history is unavailable for private, connector or remote conversations.')
  account(raw.providerId, raw.accountId)
  let selected: string
  if (draft.surface === 'chat') {
    const thread = store.threads.find(row => row.id === draft.conversationId)
    if (!thread || !ordinary(thread)) throw new Error('This conversation cannot use shared input history.')
    if (thread.providerId !== raw.providerId || (thread.providerAccountId ?? null) !== raw.accountId) throw new Error('The conversation account changed. Reopen input history.')
    selected = workspace(thread)
  } else {
    if (draft.conversationId !== null) throw new Error('Invalid input history home scope.')
    const project = raw.projectId ? store.projects.find(row => row.id === raw.projectId) : null
    if (!project && raw.cwd !== (store.settings.defaultCwd ?? '')) throw new Error('Start a conversation in this folder before using its input history.')
    if (project && raw.cwd && ![project.cwd, project.workspace?.path].includes(raw.cwd)) throw new Error('The selected workspace changed. Reopen input history.')
    selected = workspace({ projectId: raw.projectId, cwd: raw.cwd })
  }
  const key = hash([selected, raw.providerId, raw.accountId])
  return { key, eligible: thread => ordinary(thread) && thread.providerId === raw.providerId && (thread.providerAccountId ?? null) === raw.accountId && workspace(thread) === selected }
}
function entries(owner: ReturnType<typeof inputHistoryOwner>, before: number): { entries: InputHistoryEntry[]; partial: boolean } {
  const rows: InputHistoryEntry[] = [], seen = new Set<string>()
  let visited = 0, partial = false
  const candidates = store.threads.filter(thread => { try { return owner.eligible(thread) } catch { return false } }).sort((a, b) => b.updatedAt - a.updatedAt)
  outer: for (const thread of candidates) for (let index = thread.timeline.length - 1; index >= 0; index--) {
    if (++visited > 10_000) { partial = true; break outer }
    const item = thread.timeline[index]
    if (item.kind !== 'user' || item.memorySource !== 'user' || !item.text?.trim() || !Number.isSafeInteger(item.createdAt) || item.createdAt! <= before || item.text.length > 32_000 || 'quotedMessages' in item) continue
    if (seen.has(item.text)) continue
    seen.add(item.text)
    rows.push({ id: hash([thread.id, item.id, item.text, item.createdAt]), text: item.text, createdAt: item.createdAt! })
  }
  rows.sort((a, b) => b.createdAt - a.createdAt)
  return { entries: rows.slice(0, 30), partial: partial || rows.length > 30 }
}
function capture(scope: InputHistoryScope) {
  const profile = vontraPath(), owner = inputHistoryOwner(scope)
  const current = () => { if (vontraPath() !== profile || inputHistoryOwner(scope).key !== owner.key) throw new Error('The active input history scope changed. Reopen it.') }
  return { profile, owner, current }
}
export async function getInputHistory(scope: InputHistoryScope): Promise<InputHistorySnapshot> {
  const { profile, owner, current } = capture(scope)
  const state = await inputHistoryClearState(profile, owner.key, current)
  current(); return { revision: state.revision, ...entries(owner, state.before) }
}
export async function recallInputHistory(input: { scope: InputHistoryScope; id: string; expectedRevision: number }): Promise<string> {
  if (!input || typeof input.id !== 'string' || !/^[a-f0-9]{64}$/.test(input.id) || !Number.isSafeInteger(input.expectedRevision)) throw new Error('Invalid input history selection.')
  const result = await getInputHistory(input.scope)
  if (result.revision !== input.expectedRevision) throw new Error('Input history changed. Refresh it before inserting a prompt.')
  const entry = result.entries.find(row => row.id === input.id)
  if (!entry) throw new Error('This input is no longer available in the selected workspace and account.')
  return entry.text
}
export async function clearInputHistory(input: { scope: InputHistoryScope; expectedRevision: number }): Promise<InputHistorySnapshot> {
  if (!input || !Number.isSafeInteger(input.expectedRevision) || input.expectedRevision < 0) throw new Error('Invalid input history revision.')
  const { profile, owner, current } = capture(input.scope)
  const state = await clearInputHistoryState(profile, owner.key, input.expectedRevision, current)
  current(); return { revision: state.revision, entries: [], partial: false }
}
