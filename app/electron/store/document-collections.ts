import type { Thread } from '../agent/types'
import { validateDocumentSourceIds, documentPolicySignature } from '../../shared/document-collections'
import { stripIncognitoThreads } from '../agent/incognito'
import { threadsPath } from './paths'
import { writeProtectedJson } from './write-infra'

/** Writes an immutable candidate in the serialized file slot before publishing the accepted policy. */
export async function saveThreadDocumentCollections(getThreads: () => Thread[], id: string, value: unknown, validate: (thread: Thread) => void): Promise<Thread> {
  const requested = validateDocumentSourceIds(value)
  if (!requested) throw new Error('Choose document collections explicitly.')
  const initial = getThreads().find(thread => thread.id === id)
  if (!initial || initial.archivedAt) throw new Error('Choose an active chat.')
  const previous = documentPolicySignature(initial.documentSourceIds)
  const resolve = () => {
    const current = getThreads().find(thread => thread.id === id)
    if (!current || current !== initial || current.archivedAt || documentPolicySignature(current.documentSourceIds) !== previous) throw new Error('Document choices changed in another window; reopen this menu and try again.')
    validate(current)
    return current
  }
  const patch = { documentSourceIds: [...requested], updatedAt: Date.now() }
  resolve()
  if (initial.incognito) { Object.assign(initial, patch); return initial }
  await writeProtectedJson(threadsPath(), stripIncognitoThreads(getThreads()), {
    prepareData: () => { const target = resolve(); return stripIncognitoThreads(getThreads().map(thread => thread === target ? { ...thread, ...patch } : thread)) },
    onCommitted: () => Object.assign(initial, patch),
  })
  return initial
}
