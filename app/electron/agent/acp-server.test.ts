// Run: jiti electron/agent/acp-server.test.ts
//
// Tests the real ACP agent server against the SDK's own in-process composition
// (`ClientApp.connectWith(AgentApp, ...)` — no fake transport needed; the SDK
// itself supports connecting an agent and a client directly without a stream,
// "useful for tests and in-process examples"). This exercises the real JSON-RPC
// request/notification plumbing the SDK provides, while the actual turn logic
// (the equivalent of `runAgent`) is dependency-injected as a fake — mirroring
// how codex-app-server.test.ts injects a FakeTransport for protocol-level tests
// without spawning a real subprocess or calling a real model.
import assert from 'node:assert/strict'
import * as acp from '@agentclientprotocol/sdk'
import { buildTensorFoldStudioAcpAgentApp, type AcpTurnRunner, type AcpTurnDeps } from './acp-server'

// A fake turn runner standing in for the real loop.ts-backed adapter. Records every
// invocation and lets each test script exactly what it streams back / how it behaves
// on cancellation, the same dependency-injection shape codex-app-server.test.ts uses
// for its FakeTransport.
function fakeTurnRunner(script: (params: AcpTurnDeps) => Promise<void>): { runner: AcpTurnRunner; calls: AcpTurnDeps[] } {
  const calls: AcpTurnDeps[] = []
  const runner: AcpTurnRunner = async (params) => {
    calls.push(params)
    await script(params)
  }
  return { runner, calls }
}

// Minimal ACP client used to drive the agent under test. Auto-approves every
// permission request with the first "allow"-kind option offered, mirroring a
// real editor's default-allow test harness.
function buildTestClient() {
  const permissionRequests: acp.RequestPermissionRequest[] = []
  const client = acp.client({ name: 'acp-server-test-client' }).onRequest(acp.methods.client.session.requestPermission, async (ctx) => {
    permissionRequests.push(ctx.params)
    const allow = ctx.params.options.find((o) => o.kind === 'allow_once' || o.kind === 'allow_always') ?? ctx.params.options[0]
    return { outcome: { outcome: 'selected', optionId: allow.optionId } }
  })
  return { client, permissionRequests }
}

async function collectSessionUpdates<T>(op: (push: (u: acp.SessionUpdate) => void) => Promise<T>): Promise<{ result: T; updates: acp.SessionUpdate[] }> {
  const updates: acp.SessionUpdate[] = []
  const result = await op((u) => updates.push(u))
  return { result, updates }
}

// --- initialize returns sane capabilities ---
async function testInitialize() {
  const { runner } = fakeTurnRunner(async () => {})
  const agentApp = buildTensorFoldStudioAcpAgentApp({ runTurn: runner, cwd: '/repo' })
  const { client } = buildTestClient()

  const init = await client.connectWith(agentApp, async (ctx) => {
    return ctx.request('initialize', { protocolVersion: acp.PROTOCOL_VERSION })
  })

  assert.equal(init.protocolVersion, acp.PROTOCOL_VERSION)
  // loadSession is honestly false: this v1 server does not implement session/load
  // (no handler is registered for it), so advertising true would make a client's
  // session/load call fail with "method not found" instead of skipping it cleanly.
  assert.equal(init.agentCapabilities?.loadSession, false)
  assert.equal(init.agentCapabilities?.promptCapabilities?.embeddedContext, true)
  // Streaming prompt output requires the client be told the agent supports it implicitly
  // (ACP has no explicit "streaming" capability flag — session/update notifications are
  // always how output streams once a session exists), so just assert agentInfo identifies TensorFold.
  assert.equal(init.agentInfo?.name, 'TensorFold')
}

// --- a new session can be created ---
async function testNewSession() {
  const { runner } = fakeTurnRunner(async () => {})
  const agentApp = buildTensorFoldStudioAcpAgentApp({ runTurn: runner, cwd: '/repo' })
  const { client } = buildTestClient()

  const sessionId = await client.connectWith(agentApp, async (ctx) => {
    await ctx.request('initialize', { protocolVersion: acp.PROTOCOL_VERSION })
    const session = await ctx.buildSession('/repo').start()
    return session.sessionId
  })

  assert.ok(sessionId, 'newSession returns a non-empty sessionId')
  assert.equal(typeof sessionId, 'string')
}

// --- a prompt turn produces streamed output mapped from a fake/injected TensorFold run ---
async function testPromptStreamsOutput() {
  const { runner, calls } = fakeTurnRunner(async ({ emitContent, emitReasoning }) => {
    emitReasoning('thinking about it... ')
    emitContent('Hello ')
    emitContent('from TensorFold.')
  })
  const agentApp = buildTensorFoldStudioAcpAgentApp({ runTurn: runner, cwd: '/repo' })
  const { client } = buildTestClient()

  const { result, updates } = await client.connectWith(agentApp, async (ctx) => {
    await ctx.request('initialize', { protocolVersion: acp.PROTOCOL_VERSION })
    return ctx.buildSession('/repo').withSession(async (session) => {
      return collectSessionUpdates(async (push) => {
        session.prompt('hi there')
        for (;;) {
          const msg = await session.nextUpdate()
          if (msg.kind === 'stop') return msg.response
          push(msg.update)
        }
      })
    })
  })

  assert.equal(result.stopReason, 'end_turn')
  const agentChunks = updates.filter((u) => u.sessionUpdate === 'agent_message_chunk')
  const thoughtChunks = updates.filter((u) => u.sessionUpdate === 'agent_thought_chunk')
  assert.equal(agentChunks.length, 2)
  assert.equal((agentChunks[0] as any).content.text, 'Hello ')
  assert.equal((agentChunks[1] as any).content.text, 'from TensorFold.')
  assert.equal(thoughtChunks.length, 1)
  assert.equal((thoughtChunks[0] as any).content.text, 'thinking about it... ')

  // The turn runner received the prompt text and a working directory derived from the session.
  assert.equal(calls.length, 1)
  assert.equal(calls[0].promptText, 'hi there')
  assert.equal(calls[0].cwd, '/repo')
}

// --- a tool-approval request routes through session/request_permission ---
async function testPromptRequestsPermission() {
  const { runner } = fakeTurnRunner(async ({ requestApproval }) => {
    const decision = await requestApproval({ toolCallId: 'call_1', title: 'Edit config.json', kind: 'edit' })
    assert.equal(decision, 'allow')
  })
  const agentApp = buildTensorFoldStudioAcpAgentApp({ runTurn: runner, cwd: '/repo' })
  const { client, permissionRequests } = buildTestClient()

  const response = await client.connectWith(agentApp, async (ctx) => {
    await ctx.request('initialize', { protocolVersion: acp.PROTOCOL_VERSION })
    return ctx.buildSession('/repo').withSession(async (session) => session.prompt('edit a file'))
  })

  assert.equal(response.stopReason, 'end_turn')
  assert.equal(permissionRequests.length, 1)
  assert.equal(permissionRequests[0].toolCall.toolCallId, 'call_1')
  assert.equal(permissionRequests[0].toolCall.title, 'Edit config.json')
  // TensorFold's loop offers a tri-state decision (allow / allow_always / deny), so the
  // permission request surfaces all three as real, distinct ACP options rather than
  // collapsing "always allow" into a plain allow.
  const optionKinds = permissionRequests[0].options.map((o) => o.kind).sort()
  assert.deepEqual(optionKinds, ['allow_always', 'allow_once', 'reject_once'])
}

// --- a denied permission request maps back to 'deny' ---
async function testPromptPermissionDenied() {
  const { runner } = fakeTurnRunner(async ({ requestApproval }) => {
    const decision = await requestApproval({ toolCallId: 'call_2', title: 'Delete file', kind: 'delete' })
    assert.equal(decision, 'deny')
  })
  const agentApp = buildTensorFoldStudioAcpAgentApp({ runTurn: runner, cwd: '/repo' })
  const client = acp.client({ name: 'denying-client' }).onRequest(acp.methods.client.session.requestPermission, async (ctx) => {
    const reject = ctx.params.options.find((o) => o.kind.startsWith('reject'))!
    return { outcome: { outcome: 'selected', optionId: reject.optionId } }
  })

  const response = await client.connectWith(agentApp, async (ctx) => {
    await ctx.request('initialize', { protocolVersion: acp.PROTOCOL_VERSION })
    return ctx.buildSession('/repo').withSession(async (session) => session.prompt('delete a file'))
  })
  assert.equal(response.stopReason, 'end_turn')
}

// --- cancel actually stops an in-flight fake turn ---
async function testCancelStopsInFlightTurn() {
  let observedAbort = false
  const { runner } = fakeTurnRunner(async ({ signal, emitContent }) => {
    emitContent('starting work')
    await new Promise<void>((resolve) => {
      if (signal.aborted) { observedAbort = true; resolve(); return }
      signal.addEventListener('abort', () => { observedAbort = true; resolve() }, { once: true })
    })
  })
  const agentApp = buildTensorFoldStudioAcpAgentApp({ runTurn: runner, cwd: '/repo' })
  const { client } = buildTestClient()

  const response = await client.connectWith(agentApp, async (ctx) => {
    await ctx.request('initialize', { protocolVersion: acp.PROTOCOL_VERSION })
    return ctx.buildSession('/repo').withSession(async (session) => {
      const promptPromise = session.prompt('do a long task')
      // Drain the first update so we know the turn has actually started before cancelling.
      const first = await session.nextUpdate()
      assert.equal(first.kind, 'session_update')
      await ctx.notify('session/cancel', { sessionId: session.sessionId })
      return promptPromise
    })
  })

  assert.equal(response.stopReason, 'cancelled')
  assert.equal(observedAbort, true, 'the injected turn runner observed the abort signal firing')
}

// --- prompting an unknown session id is a clean error, not a crash ---
async function testPromptUnknownSession() {
  const { runner } = fakeTurnRunner(async () => {})
  const agentApp = buildTensorFoldStudioAcpAgentApp({ runTurn: runner, cwd: '/repo' })
  const { client } = buildTestClient()

  await assert.rejects(
    client.connectWith(agentApp, async (ctx) => {
      await ctx.request('initialize', { protocolVersion: acp.PROTOCOL_VERSION })
      return ctx.request('session/prompt', { sessionId: 'does-not-exist', prompt: [{ type: 'text', text: 'hi' }] })
    }),
    /session/i,
  )
}

async function main() {
  await testInitialize()
  await testNewSession()
  await testPromptStreamsOutput()
  await testPromptRequestsPermission()
  await testPromptPermissionDenied()
  await testCancelStopsInFlightTurn()
  await testPromptUnknownSession()
  console.log('acp-server tests passed')
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
