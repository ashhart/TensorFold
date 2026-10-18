import assert from 'node:assert/strict'
import { createGoogleContextMarker, createGoogleWorkspaceToolRunner } from './google-workspace-tool-runtime'
import { GoogleWorkspaceService } from '../../agent/google-workspace'
import { requestedScopes } from '../../agent/google-workspace/scopes'
import type { GoogleWorkspacePrivateState } from '../../agent/google-workspace'

function setup(extraThread: Record<string, unknown> = {}, extraProvider: Record<string, unknown> = {}) {
  const thread: any = { id: 'synthetic-thread', kind: 'task', title: 'Example', cwd: '/synthetic-workspace', providerId: 'synthetic-local', model: 'synthetic', timeline: [], apiMessages: [], memoryPolicy: { useExisting: true, contribute: true }, memoryPolicyRevision: 0, ...extraThread }
  const provider: any = { id: 'synthetic-local', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:64000/v1', models: ['synthetic'], ...extraProvider }
  const controller = new AbortController()
  const run = { thread, provider, model: 'synthetic', signal: controller.signal }
  let liveThread = thread; let liveProvider = provider
  let calls = 0; let factories = 0; let recallBlocked = false; let research = false
  let disabledCapabilities: string[] = []; let disabledTools: string[] = []
  let resolveData!: (value: any) => void
  let begin!: () => void
  const started = new Promise<void>(resolve => { begin = resolve })
  let hold = false
  const deps = {
    getThread: () => liveThread, resolveProvider: () => liveProvider,
    service: () => { factories++; return { snapshot: async () => { calls++; begin(); if (hold) return new Promise(resolve => { resolveData = resolve }); return { clientConfigured: true, accounts: [] } } } as any },
    disabledCapabilities: () => disabledCapabilities, disabledTools: () => disabledTools,
    assertRecallSourcesAllowed: () => { if (recallBlocked) throw new Error('Recalled source privacy changed') },
    hasExternalResearchEnabled: () => research,
    markContext: async (thread: any) => { thread.googleLocalOnly = true },
  }
  return { thread, provider, controller, run, deps, runner: createGoogleWorkspaceToolRunner(run, deps), get calls() { return calls }, get factories() { return factories },
    setThread: (value: any) => { liveThread = value }, setProvider: (value: any) => { liveProvider = value }, blockRecall: () => { recallBlocked = true }, enableResearch: () => { research = true }, disableCapability: () => { disabledCapabilities = ['connector:google-workspace'] }, disableTool: () => { disabledTools = ['google_accounts'] },
    hold: () => { hold = true }, started, finish: () => resolveData({ clientConfigured: true, accounts: [{ id: 'synthetic-google', email: 'person-a@example.com', capabilities: ['gmail.read'], revision: 1, connectedAt: 1, status: 'connected' }] }),
  }
}
const allowed = setup()
assert.equal((await allowed.runner('google_accounts', {})).ok, true)
assert.equal(allowed.calls, 1)
assert.equal(allowed.thread.googleLocalOnly, true)
const noMemory = setup({ memoryPolicy: { useExisting: false, contribute: false } })
assert.equal((await noMemory.runner('google_accounts', {})).ok, true, 'an explicit current Google read does not recall saved memories')
const incognitoRead = setup({ incognito: true })
assert.equal((await incognitoRead.runner('google_accounts', {})).ok, true)

for (const fixture of [setup({}, { kind: 'cloud', baseUrl: 'https://model.example.com' }), setup({}, { kind: 'local', baseUrl: 'http://192.0.2.20:8080/v1' }), setup({}, { transport: 'codex-cli' }), setup({ bot: { executionTarget: 'vm' } }), setup({ orchestratorHostId: 'synthetic-remote' }), setup({ workspace: { kind: 'ssh', hostId: 'synthetic-ssh', path: '/synthetic' } }), setup({ phoneLocalOnly: true }), setup({ engine: 'omp' })]) {
  await assert.rejects(fixture.runner('google_accounts', {}))
  assert.equal(fixture.factories, 0, 'blocked routes cannot access the protected Google service')
}
const missing = setup()
await assert.rejects(createGoogleWorkspaceToolRunner(undefined, missing.deps)('google_accounts', {}), /explicitly bound/)
await assert.rejects(createGoogleWorkspaceToolRunner({ thread: missing.thread, provider: missing.provider, model: 'synthetic' }, missing.deps)('google_accounts', {}), /explicitly bound/)
assert.equal(missing.factories, 0)
const privateDraft = setup({ incognito: true })
await assert.rejects(privateDraft.runner('gmail_draft', { accountId: 'synthetic-google' }), /Incognito/)
await assert.rejects(privateDraft.runner('google_action_prepare', { accountId: 'synthetic-google' }), /Incognito/)
assert.equal(privateDraft.factories, 0)
const scoped = setup({ bot: { toolScope: { mode: 'selected', names: ['read_file'] } } })
await assert.rejects(scoped.runner('google_accounts', {}), /captured tool scope/)
assert.equal(scoped.factories, 0)

const mutations: ((fixture: ReturnType<typeof setup>) => void)[] = [
  fixture => fixture.controller.abort(),
  fixture => fixture.setThread({ ...fixture.thread }),
  fixture => { fixture.thread.archivedAt = 10 },
  fixture => { fixture.thread.memoryPolicyRevision++ },
  fixture => { fixture.thread.providerId = 'synthetic-other' },
  fixture => { fixture.thread.model = 'synthetic-other' },
  fixture => { fixture.thread.incognito = true },
  fixture => fixture.setProvider({ ...fixture.provider, baseUrl: 'https://model.example.com', kind: 'cloud' }),
  fixture => fixture.blockRecall(),
  fixture => fixture.enableResearch(),
  fixture => fixture.disableCapability(),
  fixture => fixture.disableTool(),
]
for (const mutate of mutations) {
  const fixture = setup(); fixture.hold()
  const pending = fixture.runner('google_accounts', {})
  const failure = assert.rejects(pending)
  await fixture.started
  mutate(fixture); fixture.finish()
  await failure
  assert.equal(fixture.calls, 1, 'private account data is discarded after a live boundary change')
}

// Check the actual service's draft transaction guard at the protected-store commit seam.
let protectedState: GoogleWorkspacePrivateState = { version: 1, accounts: [{ id: 'synthetic-google', subject: 'synthetic-subject', email: 'person-a@example.com', status: 'connected', revision: 1, capabilities: ['gmail.read'], connectedAt: 1, tokens: { accessToken: 'synthetic-access', expiresAt: Date.now() + 3_600_000, scopes: requestedScopes(['gmail.read']) } }], drafts: [], actions: [] }
let releaseWrite!: () => void; let writeBegin!: () => void
const writing = new Promise<void>(resolve => { writeBegin = resolve })
const release = new Promise<void>(resolve => { releaseWrite = resolve })
const guardedService = new GoogleWorkspaceService({ openExternal: async () => {}, store: {
  read: async () => structuredClone(protectedState),
  write: async (value, validateCommit) => { writeBegin(); await release; validateCommit?.(); protectedState = structuredClone(value) },
} })
const draftFixture = setup()
const draftRunner = createGoogleWorkspaceToolRunner(draftFixture.run, { ...draftFixture.deps, service: () => guardedService })
const draftWrite = draftRunner('gmail_draft', { accountId: 'synthetic-google', email: { to: ['person-b@example.com'], subject: 'Example', text: 'No persistence after cancellation' } })
const rejectedDraft = assert.rejects(draftWrite, /changed|active/)
await writing
draftFixture.thread.incognito = true
releaseWrite()
await rejectedDraft
assert.equal(protectedState.drafts.length, 0, 'privacy changes during disk IO cannot commit a local draft')
guardedService.dispose()
let completeMarker!: () => void
let markerWrites = 0; let markerPublished = 0
const markerThread = setup().thread
const markerWrite = new Promise<void>(resolve => { completeMarker = resolve })
const mark = createGoogleContextMarker({ persist: () => { markerWrites++; return markerWrite }, publish: () => { markerPublished++ } })
let firstReturned = false; let secondReturned = false
const firstMarker = mark(markerThread).then(() => { firstReturned = true })
const secondMarker = mark(markerThread).then(() => { secondReturned = true })
await Promise.resolve()
assert.equal(markerWrites, 1)
assert.equal(firstReturned || secondReturned, false, 'parallel reads cannot observe an in-memory marker as already durable')
completeMarker(); await Promise.all([firstMarker, secondMarker])
assert.equal(markerPublished, 1)
let failureWrites = 0
const failureThread = setup().thread
const retryMarker = createGoogleContextMarker({ persist: async () => { if (++failureWrites === 1) throw new Error('Synthetic disk failure') }, publish() {} })
await assert.rejects(retryMarker(failureThread), /disk failure/)
assert.equal(failureThread.googleLocalOnly, true, 'a failed disk write cannot reopen external routes in memory')
await retryMarker(failureThread)
assert.equal(failureWrites, 2, 'the next attempt must make the marker durable before returning data')
console.log('Google host-bound local routes, live privacy/abort guards and atomic draft admission tests passed')
