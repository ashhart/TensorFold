import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { enqueueJob, updateJob, recordJobApproval, resolveJobApproval } from '../agent/jobs'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-incognito-derived-'))
process.env.VONTRA_HOME = profile
const { saveJobs, loadJobs, readProtectedJson, saveThreads, saveProjects, saveRooms, saveTickets, closeSessionSearchIndexes } = await import('../store')
const { store } = await import('../main/state')
const { archiveEntity, archiveSnapshot, restoreEntity } = await import('../main/host/archive')
const secret = 'EXAMPLE-PRIVATE-TRANSCRIPT-427'

try {
  let privateResult = enqueueJob([], { kind: 'thread', owner: 'private-chat', threadId: 'private-chat', incognito: true, resumeAfterRestart: 'full_policy', metadata: { title: secret } })
  assert.equal(privateResult.job.incognito, true)
  assert.equal(privateResult.job.resumeAfterRestart, 'never')
  privateResult = { ...privateResult, ...updateJob(privateResult.jobs, privateResult.job.id, { incognito: false, resumeAfterRestart: 'full_policy', metadata: { result: secret } } as any) } as typeof privateResult
  assert.equal(privateResult.job.incognito, true, 'a privacy marker cannot be cleared by a generic job patch')
  assert.equal(privateResult.job.resumeAfterRestart, 'never')
  const pending = recordJobApproval(privateResult.jobs, privateResult.job.id, { callId: 'private-call', threadId: 'private-chat', tool: 'write_file', policy: { label: secret } as any })
  assert.equal(pending.job?.status, 'waiting_approval', 'Incognito approvals remain usable in live memory')
  const resolved = resolveJobApproval(pending.jobs, 'private-call', 'allow')
  assert.equal(resolved.approval?.decision, 'allow')
  assert.equal(resolved.job?.incognito, true)
  const normal = enqueueJob(resolved.jobs, { kind: 'thread', owner: 'private-chat', threadId: 'ordinary-chat', metadata: { title: 'Normal example' } })
  assert.equal(normal.created, true, 'ordinary and Incognito jobs never merge merely because an owner matches')
  await saveJobs(normal.jobs)
  assert.deepEqual((await loadJobs()).map((job) => job.id), [normal.job.id])
  assert.ok(!JSON.stringify(await readProtectedJson(path.join(profile, 'state/jobs.json'), [])).includes(secret), 'private job approvals/metadata/receipts never enter the encrypted collection')

  store.projects = [{ id: 'project-example', name: 'Normal example', createdAt: 1, updatedAt: 1 } as any]
  store.rooms = []; store.tickets = []
  store.threads = [
    { id: 'private-chat', projectId: 'project-example', incognito: true, title: secret, timeline: [{ id: 'private-answer', kind: 'assistant', text: secret }], createdAt: 1, updatedAt: 1 } as any,
    { id: 'ordinary-chat', projectId: 'project-example', title: 'Normal example', timeline: [], createdAt: 1, updatedAt: 1 } as any,
  ]
  await archiveEntity('thread', 'private-chat')
  assert.ok(archiveSnapshot().threads.some((thread) => thread.id === 'private-chat'), 'local memory-only archive remains accessible until app exit')
  assert.equal(await fs.stat(path.join(profile, 'archives/threads/private-chat.json')).then(() => true, () => false), false, 'archiving does not create a private transcript recovery file')
  const index: any = await readProtectedJson(path.join(profile, 'archives/index.json'), {})
  assert.ok(!JSON.stringify(index).includes(secret) && !JSON.stringify(index).includes('private-chat'), 'durable archive index excludes private rows')
  await restoreEntity('thread', 'private-chat')
  await archiveEntity('project', 'project-example')
  const project: any = await readProtectedJson(path.join(profile, 'archives/projects/project-example.json'), {})
  assert.deepEqual(project.threads.map((thread: any) => thread.id), ['ordinary-chat'])
  assert.ok(!JSON.stringify(project).includes(secret), 'project recovery snapshots never embed an Incognito child transcript')
  await Promise.all([saveThreads(store.threads), saveProjects(store.projects), saveRooms(store.rooms), saveTickets(store.tickets)])
  console.log('Incognito derived jobs, live approvals, archive index and project/thread recovery remain memory-only')
} finally {
  closeSessionSearchIndexes()
  await fs.rm(profile, { recursive: true, force: true })
}
