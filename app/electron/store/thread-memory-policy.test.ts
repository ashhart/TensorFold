import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import type { Thread } from '../agent/types'
const home = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-memory-policy-store-'))
process.env.VONTRA_HOME = home
const { saveThreads, loadThreads, saveThreadMemoryPolicy, writeProtectedJson, threadsPath, closeSessionSearchIndexes } = await import('../store')
const { createThreadMemoryPolicyHost } = await import('../main/host/thread-memory-policy-host')
const { WorkspaceMutationGuard } = await import('../main/host/workspace-mutation-guard')
const thread: Thread = { id: 'memory-owner', title: 'Synthetic saved history', kind: 'chat', cwd: '', status: 'idle', createdAt: 1, updatedAt: 1, providerId: 'local', model: 'synthetic', apiMessages: [{ role: 'user', content: 'PRESERVED-HISTORY-AMBER' }], timeline: [], memoryPolicy: { useExisting: true, contribute: true } }
let threads = [thread]
const provider: any = { id: 'local', transport: 'chat-completions' }
const aborters = new Map<string, AbortController>()
const guard = new WorkspaceMutationGuard()
let published = 0
const update = createThreadMemoryPolicyHost({ threads: () => threads, provider: () => provider, aborters, guard, publish: () => { published++ } })
const rename = fs.rename
try {
  await saveThreads(threads)
  let entered!: () => void, release!: () => void
  const blocked = new Promise<void>((resolve) => { entered = resolve })
  const gate = new Promise<void>((resolve) => { release = resolve })
  let block = true
  fs.rename = async (from, to) => { if (String(to) === threadsPath() && block) { block = false; entered(); await gate }; return rename(from, to) }
  const changing = update(thread.id, { useExisting: false, contribute: true }, 0)
  await blocked
  assert.deepEqual(thread.memoryPolicy, { useExisting: true, contribute: true }, 'threads:get live object must not claim acceptance during blocked durable rename')
  assert.equal(published, 0)
  assert.throws(() => guard.reserveStart({ threadId: thread.id }), /starting|restoration/, 'a model run cannot start while policy save is pending')
  const queuedGeneric = saveThreads(threads)
  release(); await changing; await queuedGeneric
  assert.equal(published, 1)
  assert.deepEqual((await loadThreads())[0].memoryPolicy, { useExisting: false, contribute: true }, 'later queued generic save cannot restore an old choice')
  assert.equal((await loadThreads())[0].apiMessages[0].content, 'PRESERVED-HISTORY-AMBER')
  await assert.rejects(update(thread.id, { useExisting: true, contribute: false }, 0), /another window/)
  const beforeFailure = await fs.readFile(threadsPath())
  fs.rename = async (from, to) => { if (String(to) === threadsPath()) throw new Error('synthetic durable failure'); return rename(from, to) }
  await assert.rejects(update(thread.id, { useExisting: false, contribute: false }, 1), /synthetic durable failure/)
  assert.equal(thread.memoryPolicyRevision, 1)
  assert.equal(published, 1)
  assert.deepEqual(await fs.readFile(threadsPath()), beforeFailure)
  fs.rename = rename
  aborters.set(thread.id, new AbortController())
  await assert.rejects(update(thread.id, { useExisting: false, contribute: false }, 1), /Stop this run/)
  aborters.clear()
  for (const useExisting of [true, false]) for (const contribute of [true, false]) {
    await update(thread.id, { useExisting, contribute }, thread.memoryPolicyRevision!)
    const restarted = (await loadThreads())[0]
    assert.deepEqual(restarted.memoryPolicy, { useExisting, contribute })
    assert.equal(restarted.apiMessages[0].content, 'PRESERVED-HISTORY-AMBER')
  }
  assert.doesNotMatch(await fs.readFile(threadsPath(), 'utf8'), /PRESERVED-HISTORY-AMBER/)
  for (const bad of [undefined, null, { useExisting: false, contribute: 'false' }, { useExisting: false, contribute: false, injected: true }]) await assert.rejects(update(thread.id, bad, thread.memoryPolicyRevision!), /boolean|explicitly/)
  thread.memoryPolicy = { useExisting: 'damaged', contribute: true } as any
  await update(thread.id, { useExisting: false, contribute: false }, thread.memoryPolicyRevision!)
  assert.deepEqual((await loadThreads())[0].memoryPolicy, { useExisting: false, contribute: false }, 'explicit saved choices repair damaged policy without discarding ordinary history')
  provider.transport = 'codex-cli'
  await assert.rejects(update(thread.id, { useExisting: false, contribute: true }, thread.memoryPolicyRevision!), /API/)
  provider.transport = 'chat-completions'
  thread.archivedAt = 4
  await assert.rejects(update(thread.id, { useExisting: false, contribute: true }, thread.memoryPolicyRevision!), /active/)
  delete thread.archivedAt
  const privateThread = { ...thread, id: 'private-memory-owner', incognito: true, memoryPolicyRevision: 0 }
  threads.push(privateThread)
  const baseline = await fs.readFile(threadsPath())
  await update(privateThread.id, { useExisting: false, contribute: true }, 0)
  assert.deepEqual(await fs.readFile(threadsPath()), baseline, 'Incognito policy changes never touch unrelated protected history')
  // A queued choice revalidates ownership/revision in the actual serialized disk slot.
  let entered2!: () => void, release2!: () => void
  const blocked2 = new Promise<void>((r) => { entered2 = r })
  const gate2 = new Promise<void>((r) => { release2 = r })
  let block2 = true
  fs.rename = async (from, to) => { if (String(to) === threadsPath() && block2) { block2 = false; entered2(); await gate2 }; return rename(from, to) }
  const blocker = writeProtectedJson(threadsPath(), threads)
  await blocked2
  const stale = update(thread.id, { useExisting: false, contribute: false }, thread.memoryPolicyRevision!)
  thread.archivedAt = 5
  release2(); await blocker
  await assert.rejects(stale, /active/)
  fs.rename = rename
  delete thread.archivedAt
  const beforeQuarantineRevision = thread.memoryPolicyRevision
  await fs.writeFile(threadsPath(), '{synthetic malformed state')
  await loadThreads()
  await assert.rejects(update(thread.id, { useExisting: false, contribute: true }, thread.memoryPolicyRevision!), /quarantined/)
  assert.equal(thread.memoryPolicyRevision, beforeQuarantineRevision, 'quarantine must never appear as a successful empty save')
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), '{synthetic malformed state')
  await update(privateThread.id, { useExisting: true, contribute: false }, privateThread.memoryPolicyRevision!)
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), '{synthetic malformed state', 'Incognito choice changes still avoid the unrelated quarantined store')
  console.log('Memory choices: actual encrypted durable publication, queued-save composition, failure rollback, start/active/stale/archive/CLI guards, restart, ordinary history and private memory-only changes passed')
} finally { fs.rename = rename; closeSessionSearchIndexes(); await fs.rm(home, { recursive: true, force: true }) }
