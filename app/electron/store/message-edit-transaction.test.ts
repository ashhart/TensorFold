import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { Thread } from '../agent/types'
import { prepareThreadMessageEdit } from '../agent/message-edit'
import { WorkspaceMutationGuard } from '../main/host/workspace-mutation-guard'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-message-edit-'))
process.env.VONTRA_HOME = profile
const { saveThreads, loadThreads, threadsPath, closeSessionSearchIndexes } = await import('../store')
const { createThreadMessageEditHost } = await import('../main/host/thread-message-edit-host')
const { saveThreadMessageEdit } = await import('./message-edit')
function fixture(id = 'synthetic-chat'): Thread { return { id, title: 'Synthetic saved chat', kind: 'chat', cwd: '', providerId: 'local', model: 'synthetic', createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'SYNTHETIC_ORIGINAL' }, { role: 'assistant', content: 'SYNTHETIC_LATER' }], timeline: [{ id: 'u', kind: 'user', text: 'SYNTHETIC_ORIGINAL' }, { id: 'a', kind: 'assistant', text: 'SYNTHETIC_LATER' }] } }
const threads = [fixture()]
const guard = new WorkspaceMutationGuard()
const aborters = new Map<string, AbortController>()
let publishes = 0
const host = createThreadMessageEditHost({ threads: () => threads, provider: () => undefined, aborters, guard, publish: () => { publishes++ } })
const originalRename = fs.rename
try {
  await saveThreads(threads)
  const input = { ...host.prepare(threads[0].id, 'u'), threadId: threads[0].id, text: 'SYNTHETIC_REVISED' }
  const before = JSON.stringify(threads)
  let entered!: () => void, release!: () => void
  const ready = new Promise<void>((resolve) => { entered = resolve })
  const gate = new Promise<void>((resolve) => { release = resolve })
  let block = true
  fs.rename = async (from, to) => { if (String(to) === threadsPath() && block) { block = false; entered(); await gate }; return originalRename(from, to) }
  const editing = host.edit(input)
  await ready
  assert.equal(JSON.stringify(threads), before, 'threads:get and visible model history remain unchanged until encrypted rename')
  assert.equal(publishes, 0)
  assert.throws(() => guard.reserveStart({ threadId: threads[0].id }), /already starting/, 'a concurrent run cannot enter the history transaction')
  const generic = saveThreads(threads)
  release(); await editing; await generic
  assert.equal(publishes, 1)
  assert.equal(threads[0].apiMessages.length, 1)
  assert.equal(threads[0].apiMessages[0].content, input.text)
  assert.equal(threads[0].timeline[0].text, input.text)
  assert.equal(threads[0].messageEditRevisions?.[0].timeline[1].text, 'SYNTHETIC_LATER')
  const durable = await loadThreads()
  assert.equal(durable[0].apiMessages[0].content, input.text, 'a later queued generic save cannot overwrite the edited model history')
  assert.equal(durable[0].messageEditRevisions?.[0].timeline[1].text, 'SYNTHETIC_LATER', 'superseded conversation survives restart')
  const raw = await fs.readFile(threadsPath(), 'utf8')
  for (const sentinel of ['SYNTHETIC_ORIGINAL', 'SYNTHETIC_REVISED', 'SYNTHETIC_LATER']) assert.ok(!raw.includes(sentinel), 'active and superseded messages remain encrypted')
  const lease = guard.reserveStart({ threadId: threads[0].id }); lease.release()
  // Preparing then cancelling never schedules persistence or host acceptance.
  const cancelBytes = await fs.readFile(threadsPath(), 'utf8')
  host.prepare(threads[0].id, 'u')
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), cancelBytes)
  assert.equal(publishes, 1)
  // Actual atomic rename failure must retain both live and durable conversations.
  fs.rename = originalRename
  const failedInput = { ...host.prepare(threads[0].id, 'u'), threadId: threads[0].id, text: 'SYNTHETIC_REJECTED' }
  const failedBefore = JSON.stringify(threads)
  let fail = true
  fs.rename = async (from, to) => { if (String(to) === threadsPath() && fail) { fail = false; throw new Error('Synthetic disk write refused') }; return originalRename(from, to) }
  await assert.rejects(host.edit(failedInput), /disk write refused/)
  assert.equal(JSON.stringify(threads), failedBefore)
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), cancelBytes)
  assert.equal(publishes, 1)
  // The lease releases after rejection, and stale/active work remains blocked.
  const afterFailureLease = guard.reserveStart({ threadId: threads[0].id }); afterFailureLease.release()
  aborters.set(threads[0].id, new AbortController())
  assert.throws(() => host.prepare(threads[0].id, 'u'), /active run/)
  await assert.rejects(host.edit(failedInput), /active run/)
  aborters.clear()
  threads[0].timeline.push({ id: 'pending-guide', kind: 'cue', text: 'Synthetic guidance', state: 'pending' })
  assert.throws(() => host.prepare(threads[0].id, 'u'), /pending guidance/)
  threads[0].timeline.pop()
  threads[0].timeline.push({ id: 'late', kind: 'notice', text: 'Late state' })
  await assert.rejects(host.edit(failedInput), /conversation changed/)
  assert.equal(publishes, 1)
  // A task edit creates a durable new checkpoint identity without touching its source.
  fs.rename = originalRename
  const task = fixture('synthetic-task'); task.kind = 'task'; task.cwd = '/synthetic-workspace'; task.goal = 'Keep current files'; task.memoryPolicy = { useExisting: false, contribute: false }; task.phoneLocalOnly = true
  threads.push(task); await saveThreads(threads)
  const taskBefore = structuredClone(task)
  const taskInput = { ...host.prepare(task.id, 'u'), threadId: task.id, text: 'SYNTHETIC_TASK_BRANCH' }
  const branch = await host.edit(taskInput)
  assert.notEqual(branch.id, task.id)
  assert.deepEqual(task, taskBefore)
  assert.equal(threads[0].id, branch.id)
  assert.equal(branch.apiMessages.at(-1)?.content, 'SYNTHETIC_TASK_BRANCH')
  assert.equal(branch.parentId, task.id)
  assert.equal(branch.phoneLocalOnly, true)
  const savedBranches = await loadThreads()
  assert.equal(savedBranches.find((candidate) => candidate.id === task.id)?.apiMessages.length, 2)
  assert.equal(savedBranches.find((candidate) => candidate.id === branch.id)?.apiMessages.length, 1)
  // A failed secondary broadcast cannot misreport an already durable edit as rejected.
  const noBroadcast = createThreadMessageEditHost({ threads: () => threads, provider: () => undefined, aborters, guard, publish: () => { throw new Error('Synthetic closed observer') } })
  const broadcastInput = { ...noBroadcast.prepare(branch.id, 'u'), threadId: branch.id, text: 'SYNTHETIC_BROADCAST_SAVED' }
  const broadcastBranch = await noBroadcast.edit(broadcastInput)
  assert.equal(broadcastBranch.apiMessages.at(-1)?.content, 'SYNTHETIC_BROADCAST_SAVED')
  assert.equal((await loadThreads()).find((candidate) => candidate.id === broadcastBranch.id)?.apiMessages.at(-1)?.content, 'SYNTHETIC_BROADCAST_SAVED')
  // Incognito branch/cut never writes the normal store, even if it is quarantined.
  const privateChat = { ...fixture('private-chat'), incognito: true }
  const privateTask = { ...fixture('private-task'), kind: 'task' as const, incognito: true }
  threads.push(privateChat, privateTask)
  const normalBytes = await fs.readFile(threadsPath(), 'utf8')
  await host.edit({ ...host.prepare(privateChat.id, 'u'), threadId: privateChat.id, text: 'SYNTHETIC_PRIVATE_EDIT' })
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), normalBytes)
  const privateBranch = await host.edit({ ...host.prepare(privateTask.id, 'u'), threadId: privateTask.id, text: 'SYNTHETIC_PRIVATE_BRANCH' })
  assert.equal(privateBranch.incognito, true)
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), normalBytes)
  await fs.writeFile(threadsPath(), 'Synthetic malformed protected store', { mode: 0o600 })
  await loadThreads()
  await saveThreadMessageEdit(() => threads, { ...prepareThreadMessageEdit(privateChat, 'u'), threadId: privateChat.id, text: 'STILL_PRIVATE' }, () => {})
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), 'Synthetic malformed protected store')
  console.log('Actual sent-edit host + encrypted store: atomic API/timeline acceptance, failure rollback, queued saves, exclusive run lease, stale/cancel, durable task branch, preserved source and Incognito passed')
} finally {
  fs.rename = originalRename
  closeSessionSearchIndexes()
  await fs.rm(profile, { recursive: true, force: true })
}
