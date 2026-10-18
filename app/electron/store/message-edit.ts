import type { Thread } from '../agent/types'
import type { ThreadMessageEditInput } from '../../shared/thread-message-edit'
import { createEditedTaskBranch, stageThreadMessageEdit } from '../agent/message-edit'
import { stripIncognitoThreads } from '../agent/incognito'
import { threadsPath } from './paths'
import { writeProtectedJson } from './write-infra'
import { syncSessionSearchIndex } from './session-search-index'
import { blindSearchToken } from './app-crypto'
import { vontraPath } from './paths'

/** Detached encrypted commit; live API/timeline change together only after rename. */
export async function saveThreadMessageEdit(getThreads: () => Thread[], input: ThreadMessageEditInput, validate: (thread: Thread) => void): Promise<Thread> {
  const resolve = () => {
    const thread = getThreads().find((candidate) => candidate.id === input.threadId)
    if (!thread) throw new Error('Chat not found.')
    validate(thread)
    return thread
  }
  const initial = resolve()
  let patch = stageThreadMessageEdit(initial, input)
  if (!Object.keys(patch).length) return initial
  let branch = initial.kind === 'task' ? createEditedTaskBranch(initial, patch, input.itemId) : null
  if (initial.incognito) {
    if (branch) { getThreads().unshift(branch); return branch }
    Object.assign(initial, patch); return initial
  }
  let target: Thread | undefined
  await writeProtectedJson(threadsPath(), stripIncognitoThreads(getThreads()), {
    prepareData: () => {
      target = resolve()
      patch = stageThreadMessageEdit(target, input)
      if (target.kind === 'task') branch = createEditedTaskBranch(target, patch, input.itemId)
      return stripIncognitoThreads(branch ? [branch, ...getThreads()] : getThreads().map((thread) => thread === target ? { ...thread, ...patch } : thread))
    },
    onCommitted: () => { if (branch) getThreads().unshift(branch); else if (target) Object.assign(target, patch) },
  })
  try { syncSessionSearchIndex(vontraPath('state', 'session-search.sqlite'), stripIncognitoThreads(getThreads()), blindSearchToken) }
  catch { /* disposable index failure cannot invalidate the protected transcript */ }
  return branch ?? target!
}
