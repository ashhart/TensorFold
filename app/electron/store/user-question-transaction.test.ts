import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { Thread, UserQuestion } from '../agent/types'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-question-transaction-'))
process.env.VONTRA_HOME = profile
const { saveThreads, saveThreadQuestion, loadThreads, threadsPath, closeSessionSearchIndexes } = await import('../store')
const { createUserQuestionPersistence } = await import('../main/host/user-question-persistence')
const run = { threadId: 'chat-example', runId: 'run-example', itemId: 'question-item' }
const question: UserQuestion = { id: 'question-example', ...run, question: 'Which colour?', choices: ['Amber', 'Teal'], state: 'pending', answer: null, createdAt: 1, resolvedAt: null }
const item: any = { id: run.itemId, kind: 'tool', tool: 'ask_user', state: 'awaiting', question }
const threads = [{ id: run.threadId, title: 'Example chat', timeline: [item] }] as Thread[]
let events = 0
const persistence = createUserQuestionPersistence({ getThread: (id) => threads.find((thread) => thread.id === id) ?? null,
  save: (identity, patch) => saveThreadQuestion(() => threads, identity, patch), emit: () => { events++ },
})
const originalRename = fs.rename

try {
  await saveThreads(threads)
  const before = await fs.readFile(threadsPath(), 'utf8')
  let entered!: () => void, release!: () => void
  const enteredWrite = new Promise<void>((resolve) => { entered = resolve })
  const gate = new Promise<void>((resolve) => { release = resolve })
  let block = true
  fs.rename = async (from, to) => {
    if (String(to) === threadsPath() && block) { block = false; entered(); await gate }
    return originalRename(from, to)
  }
  const accepted = persistence.commit(run, { ...question, state: 'answered', answer: 'Amber', resolvedAt: 2 })
  await enteredWrite
  assert.equal(item.question.state, 'pending', 'threads:get sees only the prior durable state while the answer write is blocked')
  assert.equal(item.question.answer, null, 'an uncommitted answer never enters live thread memory')
  assert.equal(events, 0, 'no renderer acceptance is emitted before rename')
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), before, 'the prior encrypted snapshot remains intact until rename')
  const laterGenericSave = saveThreads(threads)
  release()
  await accepted
  await laterGenericSave
  assert.equal(item.question.state, 'answered')
  assert.equal(events, 1)
  assert.equal((await loadThreads())[0].timeline[0].question?.answer, 'Amber', 'a queued generic save cannot overwrite the durable answer with stale pending state')
  const raw = await fs.readFile(threadsPath(), 'utf8')
  assert.ok(!raw.includes('Amber') && !raw.includes('Which colour?'), 'question and answer remain encrypted on disk')

  item.question = question; item.state = 'awaiting'; delete item.output
  fs.rename = originalRename
  await saveThreads(threads)
  const pendingBytes = await fs.readFile(threadsPath(), 'utf8')
  let rejectEntered!: () => void, rejectWrite!: () => void
  const rejectionEntered = new Promise<void>((resolve) => { rejectEntered = resolve })
  const rejectedGate = new Promise<void>((resolve) => { rejectWrite = resolve })
  let fail = true
  fs.rename = async (from, to) => {
    if (String(to) === threadsPath() && fail) { fail = false; rejectEntered(); await rejectedGate; throw new Error('Example encrypted commit failure') }
    return originalRename(from, to)
  }
  const failed = persistence.commit(run, { ...question, state: 'answered', answer: 'Teal', resolvedAt: 3 })
  const failedAssertion = assert.rejects(failed, /encrypted commit failure/)
  await rejectionEntered
  assert.equal(item.question.state, 'pending', 'a failed write never announces acceptance through thread snapshots')
  const saveAfterFailure = saveThreads(threads)
  rejectWrite()
  await failedAssertion
  await saveAfterFailure
  assert.equal(item.question.answer, null)
  assert.equal((await loadThreads())[0].timeline[0].question?.state, 'pending')
  assert.equal(events, 1, 'a failed answer produces no acceptance event')
  assert.ok(!pendingBytes.includes('Teal'))

  fs.rename = originalRename
  const incognitoRun = { threadId: 'incognito-example', runId: 'incognito-run', itemId: 'incognito-item' }
  const incognitoQuestion = { ...question, ...incognitoRun, id: 'incognito-question' }
  const incognitoItem: any = { id: incognitoRun.itemId, kind: 'tool', tool: 'ask_user', question: incognitoQuestion, state: 'awaiting' }
  threads.push({ id: incognitoRun.threadId, title: 'Private example', incognito: true, timeline: [incognitoItem] } as Thread)
  const normalBytes = await fs.readFile(threadsPath(), 'utf8')
  await persistence.commit(incognitoRun, { ...incognitoQuestion, state: 'answered', answer: 'Private example', resolvedAt: 4 })
  assert.equal(incognitoItem.question.answer, 'Private example', 'incognito questions remain usable in live memory')
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), normalBytes, 'incognito acceptance never writes the unrelated protected thread file')
  assert.equal((await loadThreads()).length, 1, 'only the normal chat remains on disk')
  await fs.writeFile(threadsPath(), 'Example malformed protected file', { mode: 0o600 })
  await loadThreads()
  await persistence.commit(incognitoRun, { ...incognitoQuestion, state: 'answered', answer: 'Still private', resolvedAt: 5 })
  assert.equal(incognitoItem.question.answer, 'Still private', 'incognito answers still publish when the unrelated normal store is quarantined')
  assert.equal(await fs.readFile(threadsPath(), 'utf8'), 'Example malformed protected file', 'quarantined evidence is preserved unchanged')
  console.log('question transactions: live snapshots, encrypted queue ordering, failed saves and incognito passed')
} finally {
  fs.rename = originalRename
  closeSessionSearchIndexes()
  await fs.rm(profile, { recursive: true, force: true })
}
