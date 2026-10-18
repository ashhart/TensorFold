import assert from 'node:assert/strict'
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { store } from '../state'
import type { InputHistoryScope } from '../../../shared/input-history'
const root = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-history-'))
const previous = process.env.VONTRA_HOME
process.env.VONTRA_HOME = path.join(root, 'one')
const { getInputHistory, recallInputHistory, clearInputHistory } = await import('./input-history')
const { readProtectedJson } = await import('../../store/write-infra')
store.settings = { providers: [{ id: 'provider', accounts: [{ id: 'one' }, { id: 'two' }] }], defaultCwd: '/fixture/work' } as any
store.projects = [{ id: 'work', cwd: '/fixture/work' }] as any
const user = (id: string, text: string, extra = {}) => ({ id, kind: 'user', memorySource: 'user', text, createdAt: Date.now() - 100, ...extra })
const thread = (id: string, extra = {}) => ({ id, projectId: 'work', cwd: '/fixture/work', providerId: 'provider', providerAccountId: 'one', updatedAt: 2, timeline: [user(id, `${id}-DIRECT-PROMPT`)], ...extra })
store.threads = [thread('current'), thread('ordinary'), thread('private', { incognito: true }), thread('phone', { phoneLocalOnly: true }), thread('google', { googleLocalOnly: true }), thread('bot', { bot: {} }), thread('paired', { workspaceBoundary: 'home' }), thread('remote', { orchestratorHostId: 'remote' }), thread('ssh', { workspace: { kind: 'ssh', hostId: 'remote', path: '/fixture/work' } }), thread('other-account', { providerAccountId: 'two' }), thread('other-folder', { cwd: '/fixture/other' }), thread('archived', { archivedAt: 1 }), thread('external', { timeline: [user('external', 'EXTERNAL-DATA', { memorySource: 'external' })] }), thread('legacy', { timeline: [user('legacy', 'UNCLASSIFIED-DATA', { memorySource: undefined })] }), thread('quotes', { timeline: [user('quotes', 'QUOTED-PROMPT', { quotedMessages: [{ text: 'QUOTED-BODY' }] })] })] as any
const scope: InputHistoryScope = { draft: { surface: 'chat', conversationId: 'current', hostId: null, incognito: false }, projectId: 'work', cwd: '/fixture/work', providerId: 'provider', accountId: 'one' }
try {
  const first = await getInputHistory(scope)
  assert.deepEqual(first.entries.map(row => row.text).sort(), ['current-DIRECT-PROMPT', 'ordinary-DIRECT-PROMPT'])
  assert.equal(await recallInputHistory({ scope, id: first.entries[0].id, expectedRevision: 0 }), first.entries[0].text)
  store.threads[0].phoneLocalOnly = true
  await assert.rejects(() => recallInputHistory({ scope, id: first.entries[0].id, expectedRevision: 0 }), /cannot use/)
  store.threads[0].phoneLocalOnly = false
  const cleared = await clearInputHistory({ scope, expectedRevision: 0 })
  assert.deepEqual(cleared.entries, []); assert.equal(cleared.revision, 1)
  assert.deepEqual((await getInputHistory(scope)).entries, [])
  await assert.rejects(() => clearInputHistory({ scope, expectedRevision: 0 }), /changed/)
  await assert.rejects(() => recallInputHistory({ scope, id: first.entries[0].id, expectedRevision: 0 }), /changed/)
  const file = path.join(process.env.VONTRA_HOME!, 'state', 'input-history.json')
  const raw = await readFile(file, 'utf8')
  assert.doesNotMatch(raw, /DIRECT-PROMPT|fixture|provider/)
  const metadata = await readProtectedJson<any>(file, undefined, false)
  assert.deepEqual(Object.keys(metadata), ['entries'])
  assert.deepEqual(Object.keys(Object.values(metadata.entries)[0] as any).sort(), ['before', 'revision'], 'Only clearing metadata is retained, never a second copy of input text')
  const later = Date.now() + 2
  store.threads[1].timeline.push(user('new', 'NEW-DIRECT-PROMPT', { createdAt: later }) as any)
  assert.deepEqual((await getInputHistory(scope)).entries.map(row => row.text), ['NEW-DIRECT-PROMPT'])
  process.env.VONTRA_HOME = path.join(root, 'two')
  assert.equal((await getInputHistory(scope)).revision, 0, 'Clear preferences live under the active protected profile only')
  process.env.VONTRA_HOME = path.join(root, 'one')
  assert.equal((await getInputHistory(scope)).revision, 1)
  const pending = getInputHistory(scope)
  process.env.VONTRA_HOME = path.join(root, 'two')
  await assert.rejects(() => pending, /profile|scope changed/)
  process.env.VONTRA_HOME = path.join(root, 'one')
  for (const draft of [{ ...scope.draft, incognito: true }, { ...scope.draft, surface: 'assistant' as const }, { ...scope.draft, hostId: 'remote' }]) await assert.rejects(() => getInputHistory({ ...scope, draft }), /private|connector|remote/)
  await assert.rejects(() => getInputHistory({ ...scope, accountId: 'removed' }), /account/)
  store.settings.providers[0].accounts = []
  await assert.rejects(() => getInputHistory(scope), /account/)
  store.settings.providers[0].accounts = [{ id: 'one' }] as any
  const home = { ...scope, draft: { ...scope.draft, surface: 'home' as const, conversationId: null } }
  assert.ok((await getInputHistory(home)).entries.length)
  await assert.rejects(() => getInputHistory({ ...home, projectId: null, cwd: '/fixture/elsewhere' }), /Start a conversation/)
  // Corruption is a refusal, never an empty preference file that overwrites recovery bytes.
  await writeFile(file, '{malformed protected preferences')
  await assert.rejects(() => getInputHistory(scope), /unavailable/)
  await assert.rejects(() => clearInputHistory({ scope, expectedRevision: 1 }), /unavailable/)
  assert.equal(await readFile(file, 'utf8'), '{malformed protected preferences')
  console.log('Input history reads only direct same-workspace/account protected chat inputs, rejects sensitive/stale/profile/removed-account actions and stores only durable clear cutoffs')
} finally { if (previous === undefined) delete process.env.VONTRA_HOME; else process.env.VONTRA_HOME = previous; await rm(root, { recursive: true, force: true }) }
