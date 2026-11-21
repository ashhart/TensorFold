import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { createServer } from 'node:http'
import { randomUUID } from 'node:crypto'

const home = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-room-child-effort-'))
const previousHome = process.env.VONTRA_HOME
process.env.VONTRA_HOME = home
const { store } = await import('../state')
const storage = await import('../../store')
const { createToolContextRuntime } = await import('./tool-context-runtime')
const { createThreadRuntime } = await import('./thread-runtime')
const { createSubAgentFromLaunch, parentForRoomSeat } = await import('../../agent/subagents')
const previousStore = { ...store }
const requests: any[] = []
const server = createServer((request, response) => {
  if (request.method === 'GET') { response.end(JSON.stringify({ data: [{ id: 'fixture' }] })); return }
  let raw = ''
  request.on('data', chunk => { raw += chunk })
  request.on('end', () => {
    requests.push(JSON.parse(raw))
    response.writeHead(200, { 'Content-Type': 'text/event-stream' })
    response.end(`data: ${JSON.stringify({ choices: [{ delta: { content: 'Synthetic child complete.' }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
const provider: any = { id: 'fixture', name: 'Synthetic API', kind: 'local', transport: 'chat-completions',
  baseUrl: `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`, apiKey: '', models: ['fixture'] }
const fold = createThreadRuntime({} as any).applyEventToThread
const runtime = createToolContextRuntime({
  sshManager: { backendForWorkspace: () => ({ kind: 'ssh' }) }, persistSubAgents() {}, persistThreads() {},
  applyEventToThread: fold, emitToRenderer() {}, emitPluginEvent() {}, renderProjectMemoryBucket: () => '',
  recordEmailDraft() {}, recordEmailEvent() {}, recordCalendarEvent() {}, recordMessageEvent() {},
  runBrowserAutomation() {}, mcpToolContext: () => ({}), cancelThreadRun() {},
} as any)
try {
  store.settings = { ...storage.defaultSettings(), providers: [provider], activeProviderId: provider.id, activeModel: 'fixture' }
  store.threads = []; store.projects = []; store.subAgents = []
  const cases = [['high', 'low'], ['low', 'high'], ['high', undefined]] as const
  for (const [global, configured] of cases) {
    store.settings.reasoningEffort = global
    const seat = { id: 'builder', name: 'Builder', role: 'Inspect input', providerId: provider.id, model: 'fixture' }
    const room: any = { id: 'room', projectId: 'project', name: 'Synthetic room', seats: [seat] }
    const created = createSubAgentFromLaunch(store.subAgents, { source: 'room_delegate', parent: parentForRoomSeat(room, seat),
      task: 'Inspect synthetic input.', providerId: provider.id, model: 'fixture', runId: randomUUID(), cwd: home })
    store.subAgents = created.agents
    const input = { task: 'Inspect synthetic input.', parentLabel: 'Synthetic room agent', cwd: home,
      signal: AbortSignal.timeout(5_000), reasoningEffort: configured }
    const result = await runtime.runReadOnlySubAgent(created.agent, provider, 'fixture', input)
    assert.equal(result.agent.status, 'succeeded', result.output)
  }
  assert.equal(requests.length, 3, 'every real child run reaches the HTTP provider once')
  assert.deepEqual(requests.map(request => request.reasoning_effort), ['low', 'high', 'high'],
    'the actual child runtime forwards its explicit Room effort and preserves the inherited fallback')
  console.log('Real Room child HTTP checks passed: independent Low/High requests and legacy app inheritance.')
} finally {
  server.closeAllConnections()
  await new Promise<void>(resolve => server.close(() => resolve()))
  while (storage.writeChains.size) await Promise.allSettled([...storage.writeChains.values()])
  Object.assign(store, previousStore)
  storage.closeSessionSearchIndexes()
  await fs.rm(home, { recursive: true, force: true, maxRetries: 10, retryDelay: 20 })
  if (previousHome === undefined) delete process.env.VONTRA_HOME
  else process.env.VONTRA_HOME = previousHome
}
