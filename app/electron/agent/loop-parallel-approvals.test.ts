import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { runAgent } from './loop'
import { approvalPolicyScope } from './approval-policy'
import { createApprovalRuntime } from '../main/host/approval-runtime'
import type { Thread } from './types'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'studio-parallel-approvals-'))
await fs.chmod(root, 0o700)
const pendingApprovals = new Map<string, any>(), pendingApprovalMeta = new Map<string, any>()
const runtime = createApprovalRuntime({ pendingApprovals, pendingApprovalMeta, findApprovalRule: async () => null, emitJob() {}, emitToRenderer() {}, patchJob: (job: any) => job, persistJobs() {}, persistThreads() {} } as any)
const events: any[] = [], requests: any[] = [], controllers: AbortController[] = [], runs: Promise<void>[] = []
const server = createServer((request, response) => {
  let raw = ''
  request.on('data', (value) => { raw += value })
  request.on('end', () => {
    const body = JSON.parse(raw)
    requests.push(body)
    response.writeHead(200, { 'Content-Type': 'text/event-stream' })
    const hasResult = body.messages.some((message: any) => message.role === 'tool')
    const delta = hasResult ? { content: 'done' } : { tool_calls: [{ index: 0, id: 'call-shared', type: 'function', function: { name: 'write_file', arguments: JSON.stringify({ path: 'README.md', content: `${body.model}\n` }) } }] }
    response.end(`data: ${JSON.stringify({ choices: [{ delta, finish_reason: hasResult ? 'stop' : 'tool_calls' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve))
const provider = { id: 'fixture', name: 'Fixture', kind: 'local', transport: 'chat-completions', baseUrl: `http://127.0.0.1:${(server.address() as any).port}/v1`, apiKey: '', models: [] }
const waitFor = async (predicate: () => boolean) => { const until = Date.now() + 2000; while (!predicate()) { if (Date.now() > until) throw new Error('Both worker approvals did not arrive'); await new Promise((resolve) => setTimeout(resolve, 5)) } }
async function start(name: string) {
  const cwd = path.join(root, name)
  await fs.mkdir(cwd)
  const thread: Thread = { id: name, title: 'Fixture', kind: 'task', cwd, providerId: 'fixture', model: name, createdAt: 1, updatedAt: 1, status: 'idle', apiMessages: [{ role: 'user', content: 'Write one reviewed file' }], timeline: [] }
  const controller = new AbortController()
  controllers.push(controller)
  const run = runAgent({ thread, provider, model: name, approvalMode: 'suggest', temperature: 0, reasoningEffort: 'low', signal: controller.signal, persist() {}, requestApproval: runtime.approvalResolver(controller), emit(event: any) {
    events.push(event)
    if (event.type === 'item_add') thread.timeline.push(event.item)
    if (event.type === 'item_update') Object.assign(thread.timeline.find((item) => item.id === event.id) ?? {}, event.patch)
    if (event.type === 'approval_request') {
      const item = thread.timeline.find((item) => item.id === event.itemId)!
      pendingApprovalMeta.set(event.callId, { threadId: thread.id, itemId: item.id, tool: item.tool, policy: approvalPolicyScope('tool', item.tool!, item.args ?? {}) })
    }
  } } as any)
  runs.push(run)
  return { thread, controller, run }
}
try {
  const [a, b] = await Promise.all([start('worker-a'), start('worker-b')])
  await waitFor(() => events.filter((event) => event.type === 'approval_request').length === 2)
  await new Promise<void>((resolve) => setImmediate(resolve))
  assert.equal(pendingApprovals.size, 2, 'provider call IDs cannot overwrite another worker approval')
  const requestFor = (id: string) => events.find((event) => event.type === 'approval_request' && event.threadId === id)
  const aKey = requestFor(a.thread.id).callId, bKey = requestFor(b.thread.id).callId
  assert.notEqual(aKey, bKey)
  assert.notEqual(aKey, 'call-shared')
  const saved = JSON.parse(JSON.stringify(a.thread)) as Thread
  const item = saved.timeline.find((item) => item.id === requestFor(a.thread.id).itemId)!
  assert.equal(item.approvalId, aKey, 'opaque identity survives a persisted timeline round trip')
  assert.equal(item.callId, 'call-shared', 'raw protocol identity survives separately')
  pendingApprovals.get(aKey)('allow')
  pendingApprovals.get(bKey)('deny')
  await Promise.all([a.run, b.run])
  assert.equal(await fs.readFile(path.join(a.thread.cwd, 'README.md'), 'utf8'), 'worker-a\n')
  await assert.rejects(() => fs.stat(path.join(b.thread.cwd, 'README.md')), { code: 'ENOENT' })
  assert.equal(pendingApprovals.size, 0)
  assert.equal(pendingApprovalMeta.size, 0)
  for (const request of requests.filter((request) => request.messages.some((message: any) => message.role === 'tool'))) {
    assert.ok(request.messages.filter((message: any) => message.role === 'tool').every((message: any) => message.tool_call_id === 'call-shared'), 'model wire IDs remain raw')
  }
  const [c, d] = await Promise.all([start('worker-c'), start('worker-d')])
  await waitFor(() => pendingApprovals.size === 2)
  c.controller.abort()
  await c.run
  assert.equal(pendingApprovals.size, 1, 'cancelling one worker preserves the other resolver')
  assert.ok(pendingApprovals.has(requestFor(d.thread.id).callId))
  pendingApprovals.get(requestFor(d.thread.id).callId)('allow')
  await d.run
  assert.equal(await fs.readFile(path.join(d.thread.cwd, 'README.md'), 'utf8'), 'worker-d\n')
  await assert.rejects(() => fs.stat(path.join(c.thread.cwd, 'README.md')), { code: 'ENOENT' })
  assert.equal(pendingApprovalMeta.size, 0)
  console.log('Actual worker/HTTP/approval chain preserves independent decisions and raw model IDs')
} finally {
  for (const controller of controllers) controller.abort()
  await Promise.allSettled(runs)
  server.closeAllConnections()
  await new Promise<void>((resolve) => server.close(() => resolve()))
  await fs.rm(root, { recursive: true, force: true })
}
