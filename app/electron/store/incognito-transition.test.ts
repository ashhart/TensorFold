import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { DatabaseSync } from 'node:sqlite'
import type { Thread } from '../agent/types'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-private-transition-'))
process.env.VONTRA_HOME = profile
const { saveThreads, loadThreads, writeProtectedJson, threadsPath, blindSearchToken, closeSessionSearchIndexes } = await import('../store')
const { syncSessionSearchIndex } = await import('./session-search-index')
const originalRename = fs.rename
try {
  await saveThreads([])
  let entered!: () => void, release!: () => void
  const enteredWrite = new Promise<void>((resolve) => { entered = resolve })
  const gate = new Promise<void>((resolve) => { release = resolve })
  let block = true
  fs.rename = async (from, to) => {
    if (String(to) === threadsPath() && block) { block = false; entered(); await gate }
    return originalRename(from, to)
  }
  const blocker = writeProtectedJson(threadsPath(), [])
  await enteredWrite
  const thread = { id: 'transition-example', title: 'Empty task', timeline: [], apiMessages: [], createdAt: 1, updatedAt: 1 } as unknown as Thread
  const queuedBeforePrivateFlag = saveThreads([thread])
  thread.incognito = true
  thread.title = 'EXAMPLE-PRIVATE-TRANSITION-427'
  thread.apiMessages.push({ role: 'user', content: thread.title })
  thread.timeline.push({ id: 'private-example', kind: 'user', text: thread.title, createdAt: 2 })
  release()
  await blocker
  await queuedBeforePrivateFlag
  // No later clearing save is issued: the first queued write itself must be safe.
  assert.deepEqual(await loadThreads(), [], 'a pre-flag save cannot serialize the later private title or messages')
  const indexFile = path.join(profile, 'state/session-search.sqlite')
  syncSessionSearchIndex(indexFile, [thread], blindSearchToken)
  closeSessionSearchIndexes()
  const db = new DatabaseSync(indexFile, { readOnly: true })
  try { assert.deepEqual(db.prepare('SELECT thread_id FROM thread_search').all(), [], 'direct indexing never retains an Incognito thread ID or tokens') }
  finally { db.close() }
  console.log('Queued Incognito transition and direct blind-index privacy passed without a clearing write')
} finally {
  fs.rename = originalRename
  closeSessionSearchIndexes()
  await fs.rm(profile, { recursive: true, force: true })
}
