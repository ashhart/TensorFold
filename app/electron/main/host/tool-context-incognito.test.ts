import assert from 'node:assert/strict'
import { mkdtemp, readFile, readdir, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-incognito-tools-'))
process.env.VONTRA_HOME = home
const { defaultSettings } = await import('../../store')
const { store } = await import('../state')
const { createToolContextRuntime } = await import('./tool-context-runtime')
const { executeTool } = await import('../../agent/tools')
const { INCOGNITO_PERSISTENT_TOOLS } = await import('../../agent/incognito')
const { MANAGED_OUTPUT_REF_PREFIX } = await import('../../agent/context-governance')
const saved = { settings: store.settings, threads: store.threads, subAgents: store.subAgents, artifacts: store.artifacts, suggestions: store.suggestions, projects: store.projects, skills: store.skills, commands: store.commands }
const normal: any = { id: 'normal', cwd: home, kind: 'task', apiMessages: [], timeline: [] }
const privateThread = { ...normal, id: 'private', incognito: true }
let persistenceCalls = 0
store.settings = defaultSettings(); store.threads = [normal, privateThread]; store.projects = []; store.artifacts = []; store.suggestions = []; store.skills = []; store.commands = []
store.subAgents = [{ id: 'private-child', threadId: privateThread.id }, { id: 'normal-child', threadId: normal.id }] as any
const runtime = createToolContextRuntime({ sshManager: { backendForWorkspace: () => undefined } as any, persistSubAgents: () => { persistenceCalls++ }, persistThreads() {}, applyEventToThread() {}, emitToRenderer() {}, emitPluginEvent() {}, renderProjectMemoryBucket: () => '', recordEmailDraft() {}, recordEmailEvent() {}, recordCalendarEvent() {}, recordMessageEvent() {}, runBrowserAutomation() {}, mcpToolContext: () => ({}), cancelThreadRun() {} })
const provider: any = { id: 'local', kind: 'local', models: ['synthetic'] }
const privateContext = runtime.fullToolContext({ thread: privateThread, provider, model: 'synthetic', signal: new AbortController().signal })
const normalContext = runtime.fullToolContext({ thread: normal, provider, model: 'synthetic', signal: new AbortController().signal })
try {
  assert.deepEqual(privateContext.threads.map((thread: any) => thread.id), ['normal'])
  assert.deepEqual(normalContext.listSubAgents().map((agent: any) => agent.id), ['normal-child'])
  assert.throws(() => privateContext.registerArtifact({ title: 'private output', content: 'synthetic private body' }), /Incognito/)
  assert.throws(() => privateContext.fileSuggestion({ kind: 'skill', name: 'private', body: 'synthetic private body' }), /Incognito/)
  await assert.rejects(async () => privateContext.writeManagedOutput(`${MANAGED_OUTPUT_REF_PREFIX}private.txt`, 'synthetic private output'), /Incognito/)
  await assert.rejects(() => privateContext.launchSubAgent({ task: 'synthetic private task' }), /Incognito/)
  assert.throws(() => privateContext.sendSubAgentMessage({ agentId: 'private-child', text: 'synthetic', kind: 'question' }), /Incognito/)
  assert.throws(() => normalContext.readSubAgentMessages({ agentId: 'private-child', markRead: true }), /Private/)
  assert.equal(persistenceCalls, 0)
  assert.equal(store.artifacts.length, 0)
  assert.equal(store.suggestions.length, 0)
  assert.deepEqual(await readdir(home), [], 'private helper refusal happens before creating output/state directories')
  for (const name of INCOGNITO_PERSISTENT_TOOLS) {
    const result = await executeTool(name, {}, { ...privateContext, cwd: home, currentThread: privateThread })
    assert.equal(result.ok, false, `direct ${name} call must not bypass private admission`)
    assert.equal(result.meta, 'incognito-persistence-blocked')
  }
  const output = await executeTool('write_file', { path: 'explicit-output.txt', content: 'Owner-requested synthetic output.' }, { cwd: home, currentThread: privateThread })
  assert.equal(output.ok, true, 'Incognito still permits an explicitly approved workspace output')
  assert.equal(await readFile(path.join(home, 'explicit-output.txt'), 'utf8'), 'Owner-requested synthetic output.')
  const ref = await normalContext.writeManagedOutput(`${MANAGED_OUTPUT_REF_PREFIX}normal.txt`, 'ordinary saved output')
  assert.equal(await normalContext.readManagedOutput(ref), 'ordinary saved output', 'normal managed-output functionality remains intact')
  console.log('Actual Incognito tool context refuses durable helpers and private mailbox/history reads before persistence, with explicit workspace writes and ordinary managed-output positive controls.')
} finally {
  Object.assign(store, saved)
  await rm(home, { recursive: true, force: true })
}
