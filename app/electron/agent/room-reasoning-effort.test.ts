import assert from 'node:assert/strict'
import { createServer } from 'node:http'
import { runRoom, initSeatHeartbeat } from './rooms'
import { compactSeatRuntime, type SeatRuntime } from './room-seat-runtime'
import type { Project, Provider, Room, ReasoningEffortSetting } from './types'

const requests: { model: string; reasoning_effort?: string }[] = []
const reply = (model: string) => model === 'alpha'
  ? 'I checked the synthetic input.\nACTION: ASK Beta confirm the result.'
  : 'The synthetic result is confirmed.\nACTION: FINAL Synthetic review complete.'
let afterRequest = () => {}
const server = createServer((request, response) => {
  if (request.method === 'GET') {
    response.writeHead(200, { 'Content-Type': 'application/json' })
    response.end(JSON.stringify({ data: [{ id: 'alpha', object: 'model' }, { id: 'beta', object: 'model' }] }))
    return
  }
  let raw = ''
  request.on('data', chunk => { raw += chunk })
  request.on('end', () => {
    const body = JSON.parse(raw)
    requests.push(body)
    afterRequest()
    response.writeHead(200, { 'Content-Type': 'text/event-stream' })
    response.end(`data: ${JSON.stringify({ choices: [{ delta: { content: reply(body.model) }, finish_reason: 'stop' }] })}\n\ndata: [DONE]\n\n`)
  })
})
await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
const baseUrl = `http://127.0.0.1:${(server.address() as { port: number }).port}/v1`
const project = { id: 'project', name: 'Synthetic project', cwd: '/workspace', workspace: null } as Project
const cases = [['high', 'low', 'high', false], ['low', 'high', 'low', false],
  ['high', undefined, undefined, false], ['high', 'low', 'high', true]] as const
const observed: { transport: string; global: string; efforts: (string | undefined)[] }[] = []
const expected: typeof observed = []
try {
  for (const transport of ['http', 'cli'] as const) {
    for (const [global, alpha, beta, capturedByHost] of cases) {
      const seats = [
        { id: 'alpha', name: 'Alpha', role: 'Review input', providerId: 'alpha', model: 'alpha', reasoningEffort: alpha as ReasoningEffortSetting | undefined },
        { id: 'beta', name: 'Beta', role: 'Check review', providerId: 'beta', model: 'beta', reasoningEffort: beta as ReasoningEffortSetting | undefined },
      ]
      const room = { id: 'room', projectId: project.id, name: 'Synthetic room', problem: 'Review the synthetic input.', seats, events: [], memory: [],
        memorySummary: '', finalText: '', state: 'queued', seatHeartbeats: {}, deliberationEffort: 'quick',
        collaborationMode: 'balanced', createdAt: 1, updatedAt: 1 } as Room
      const providers: Provider[] = seats.map(seat => ({ id: seat.id, name: seat.name, kind: 'local', baseUrl,
        apiKey: '', models: [seat.model], transport: transport === 'http' ? 'chat-completions' : 'claude-cli' }))
      const captured = new Map(seats.map(seat => [seat.id, seat.reasoningEffort ?? global]))
      if (capturedByHost) for (const seat of seats) seat.reasoningEffort = 'medium'
      const start = requests.length
      const cliCalls: { reasoningEffort?: string }[] = []
      afterRequest = () => {
        if (requests.length === start + 1) for (const seat of seats) seat.reasoningEffort = 'medium'
      }
      const controller = new AbortController()
      const timer = setTimeout(() => controller.abort(), 5_000)
      try {
        await runRoom({ room, project, providers, problem: 'Review the synthetic input.', temperature: 0,
          reasoningEffort: global, emit() {}, signal: controller.signal, maxTurns: 4, turnTimeoutMs: 1_000,
          ...(capturedByHost ? { reasoningEffortForSeat: (seat) => captured.get(seat.id)! } : {}),
          cliSeatTurn: async (params, onDelta) => {
            cliCalls.push(params)
            if (cliCalls.length === 1) for (const seat of seats) seat.reasoningEffort = 'medium'
            const text = reply(params.model); onDelta(text); return text
          },
        })
      } finally { clearTimeout(timer) }
      assert.equal(room.state, 'complete', JSON.stringify(room.events))
      const efforts = transport === 'http' ? requests.slice(start).map(call => call.reasoning_effort)
        : cliCalls.map(call => call.reasoningEffort)
      assert.equal(efforts.length, 2, 'each real room run reaches both agents once')
      observed.push({ transport, global, efforts })
      expected.push({ transport, global, efforts: [alpha ?? global, beta ?? global] })
    }
  }
  assert.deepEqual(observed, expected, 'HTTP and CLI room requests must retain each agent effort captured before turns start')
  afterRequest = () => {}
  const seat = { id: 'compact', name: 'Compact', role: 'Review input', providerId: 'alpha', model: 'alpha' }
  const runtime: SeatRuntime = { seat, reasoningEffort: 'low', recent: [], heartbeat: initSeatHeartbeat(seat, 30, 1),
    messages: Array.from({ length: 22 }, () => ({ role: 'user', content: 'Synthetic earlier context. '.repeat(20) })) }
  const provider: Provider = { id: 'alpha', name: 'Synthetic API', kind: 'local', baseUrl, apiKey: '', models: ['alpha'] }
  const before = requests.length
  const compacted = await compactSeatRuntime(runtime, provider, 1, AbortSignal.timeout(2_000))
  assert.ok(compacted && compacted.folded > 0, 'real compaction folds earlier messages')
  assert.equal(requests.length, before + 1)
  assert.equal(requests.at(-1)?.reasoning_effort, 'low', 'private transcript compaction uses the same captured effort')
  console.log('Room effort checks passed: HTTP/CLI requests, opposite agent/app settings, legacy fallback and startup snapshots.')
} finally {
  server.closeAllConnections()
  await new Promise<void>(resolve => server.close(() => resolve()))
}
