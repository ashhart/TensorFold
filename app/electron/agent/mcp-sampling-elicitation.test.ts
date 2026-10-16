// Run: npx jiti electron/agent/mcp-sampling-elicitation.test.ts
import assert from 'node:assert/strict'
import {
  clientSamplingCapabilities,
  handleClientRequest,
  isClientRequestMethod,
  parseSamplingParams,
  parseElicitationParams,
  sanitizeElicitationContent,
  type ClientRequestDeps,
  type SamplingModelInput,
  type SamplingModelResult,
  type ElicitationPromptResult,
  type JsonRpcResult,
  type JsonRpcErrorEnvelope,
} from './mcp-sampling-elicitation.ts'

let passed = 0
function ok(label: string, cond: boolean) {
  assert.ok(cond, label)
  passed++
}

function baseDeps(over: Partial<ClientRequestDeps> = {}): ClientRequestDeps {
  return { serverId: 's1', serverName: 'Echo MCP', approvalMode: 'ask', ...over }
}

async function main() {
  // ---- capabilities advertise only what deps can serve ----
  {
    assert.deepEqual(clientSamplingCapabilities({}), {})
    assert.deepEqual(
      clientSamplingCapabilities({ runModel: async () => ({ model: 'm', content: { type: 'text', text: '' } }) }),
      { sampling: {} },
    )
    assert.deepEqual(
      clientSamplingCapabilities({ promptElicitation: async () => ({ action: 'cancel' }) }),
      { elicitation: {} },
    )
    const both = clientSamplingCapabilities({
      runModel: async () => ({ model: 'm', content: { type: 'text', text: '' } }),
      promptElicitation: async () => ({ action: 'cancel' }),
    })
    assert.deepEqual(both, { sampling: {}, elicitation: {} })
    ok('capabilities reflect injected deps', true)
  }

  // ---- method predicate ----
  ok('isClientRequestMethod sampling', isClientRequestMethod('sampling/createMessage'))
  ok('isClientRequestMethod elicitation', isClientRequestMethod('elicitation/create'))
  ok('isClientRequestMethod rejects other', !isClientRequestMethod('tools/call'))

  // ---- param validation ----
  {
    const bad = parseSamplingParams({ messages: [] })
    assert.equal(bad.ok, false)
    const good = parseSamplingParams({ messages: [{ role: 'user', content: { type: 'text', text: 'hi' } }], maxTokens: 50.9 })
    assert.equal(good.ok, true)
    if (good.ok) assert.equal(good.value.maxTokens, 50) // floored
    // invalid content type rejected
    assert.equal(parseSamplingParams({ messages: [{ role: 'user', content: { type: 'nope' } }] }).ok, false)
    // image content requires data + mimeType
    assert.equal(parseSamplingParams({ messages: [{ role: 'user', content: { type: 'image', data: 'x' } }] }).ok, false)
    ok('parseSamplingParams validates shape', true)
  }
  {
    assert.equal(parseElicitationParams({ message: '', requestedSchema: { type: 'object', properties: {} } }).ok, false)
    assert.equal(parseElicitationParams({ message: 'give api key', requestedSchema: { type: 'array' } }).ok, false)
    const good = parseElicitationParams({ message: 'key?', requestedSchema: { type: 'object', properties: { key: { type: 'string' } }, required: ['key'] } })
    assert.equal(good.ok, true)
    if (good.ok) assert.deepEqual(good.value.requestedSchema.required, ['key'])
    ok('parseElicitationParams validates shape', true)
  }

  // ---- sanitize: only schema-declared primitives survive ----
  {
    const schema = { type: 'object' as const, properties: { name: {}, age: {} } }
    const cleaned = sanitizeElicitationContent(
      { name: 'alex', age: 30, extra: 'drop', nested: { a: 1 } as any },
      schema,
    )
    assert.deepEqual(cleaned, { name: 'alex', age: 30 })
    ok('sanitizeElicitationContent drops undeclared + non-primitive fields', true)
  }

  // ---- sampling: unsupported when no runModel ----
  {
    const res = (await handleClientRequest(1, 'sampling/createMessage', { messages: [{ role: 'user', content: { type: 'text', text: 'x' } }] }, baseDeps())) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32603)
    assert.match(res.error.message, /does not support sampling/)
    ok('sampling unsupported without runModel', true)
  }

  // ---- sampling: denied by approver → error, model never runs ----
  {
    let ran = false
    const res = (await handleClientRequest(2, 'sampling/createMessage', { messages: [{ role: 'user', content: { type: 'text', text: 'x' } }] }, baseDeps({
      approveSampling: async () => false,
      runModel: async () => { ran = true; return { model: 'm', content: { type: 'text', text: 'y' } } },
    }))) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32603)
    assert.match(res.error.message, /denied/)
    assert.equal(ran, false)
    ok('sampling denied gates the model', true)
  }

  // ---- sampling: ask-mode with no approver fails closed ----
  {
    let ran = false
    const res = (await handleClientRequest(3, 'sampling/createMessage', { messages: [{ role: 'user', content: { type: 'text', text: 'x' } }] }, baseDeps({
      runModel: async () => { ran = true; return { model: 'm', content: { type: 'text', text: 'y' } } },
    }))) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32603)
    assert.equal(ran, false)
    ok('sampling fails closed with no approver on ask-mode', true)
  }

  // ---- sampling: trusted mode skips approval, runs model, returns CreateMessageResult ----
  {
    let seen: SamplingModelInput | null = null
    const res = (await handleClientRequest(4, 'sampling/createMessage', {
      messages: [{ role: 'user', content: { type: 'text', text: 'ping' } }],
      systemPrompt: 'be terse',
      temperature: 0.2,
    }, baseDeps({
      approvalMode: 'trusted',
      runModel: async (input): Promise<SamplingModelResult> => {
        seen = input
        return { model: 'gpt-x', content: { type: 'text', text: 'pong' }, stopReason: 'endTurn' }
      },
    }))) as JsonRpcResult
    assert.ok('result' in res)
    assert.deepEqual(res.result, { model: 'gpt-x', role: 'assistant', content: { type: 'text', text: 'pong' }, stopReason: 'endTurn' })
    assert.ok(seen)
    assert.equal((seen as any).systemPrompt, 'be terse')
    assert.equal((seen as any).serverName, 'Echo MCP')
    ok('sampling trusted runs model and returns result', true)
  }

  // ---- sampling: approved (ask mode) runs; default stopReason applied ----
  {
    const res = (await handleClientRequest(5, 'sampling/createMessage', {
      messages: [{ role: 'user', content: { type: 'text', text: 'x' } }],
    }, baseDeps({
      approveSampling: async () => true,
      runModel: async () => ({ model: 'm', content: { type: 'text', text: 'ok' } }),
    }))) as JsonRpcResult
    assert.deepEqual((res.result as any).stopReason, 'endTurn')
    ok('approved sampling defaults stopReason=endTurn', true)
  }

  // ---- sampling: model throwing → internal error, not a throw ----
  {
    const res = (await handleClientRequest(6, 'sampling/createMessage', {
      messages: [{ role: 'user', content: { type: 'text', text: 'x' } }],
    }, baseDeps({
      approvalMode: 'trusted',
      runModel: async () => { throw new Error('boom') },
    }))) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32603)
    assert.match(res.error.message, /Sampling failed: boom/)
    ok('sampling wraps model errors', true)
  }

  // ---- sampling: invalid params → -32602 ----
  {
    const res = (await handleClientRequest(7, 'sampling/createMessage', { messages: [] }, baseDeps({
      approvalMode: 'trusted',
      runModel: async () => ({ model: 'm', content: { type: 'text', text: 'x' } }),
    }))) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32602)
    ok('sampling invalid params → -32602', true)
  }

  // ---- elicitation: accept returns sanitized content ----
  {
    let seenMessage = ''
    const res = (await handleClientRequest(8, 'elicitation/create', {
      message: 'What is your GitHub username?',
      requestedSchema: { type: 'object', properties: { username: { type: 'string' } }, required: ['username'] },
    }, baseDeps({
      promptElicitation: async (input): Promise<ElicitationPromptResult> => {
        seenMessage = input.message
        return { action: 'accept', content: { username: 'alex', injected: 'nope' } }
      },
    }))) as JsonRpcResult
    assert.deepEqual(res.result, { action: 'accept', content: { username: 'alex' } })
    assert.equal(seenMessage, 'What is your GitHub username?')
    ok('elicitation accept returns typed content', true)
  }

  // ---- elicitation: decline carries no content ----
  {
    const res = (await handleClientRequest(9, 'elicitation/create', {
      message: 'key?',
      requestedSchema: { type: 'object', properties: { key: { type: 'string' } } },
    }, baseDeps({
      promptElicitation: async () => ({ action: 'decline', content: { key: 'leak' } }),
    }))) as JsonRpcResult
    assert.deepEqual(res.result, { action: 'decline' })
    ok('elicitation decline drops content', true)
  }

  // ---- elicitation: unsupported without hook ----
  {
    const res = (await handleClientRequest(10, 'elicitation/create', {
      message: 'x', requestedSchema: { type: 'object', properties: {} },
    }, baseDeps())) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32603)
    assert.match(res.error.message, /does not support elicitation/)
    ok('elicitation unsupported without hook', true)
  }

  // ---- elicitation: hook throwing → internal error ----
  {
    const res = (await handleClientRequest(11, 'elicitation/create', {
      message: 'x', requestedSchema: { type: 'object', properties: {} },
    }, baseDeps({
      promptElicitation: async () => { throw new Error('ui gone') },
    }))) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32603)
    assert.match(res.error.message, /Elicitation failed: ui gone/)
    ok('elicitation wraps hook errors', true)
  }

  // ---- unknown method → -32601 ----
  {
    const res = (await handleClientRequest(12, 'roots/list', undefined, baseDeps())) as JsonRpcErrorEnvelope
    assert.equal(res.error.code, -32601)
    ok('unknown method → -32601', true)
  }

  console.log(`ok - mcp-sampling-elicitation (${passed} assertions)`)
}

main().catch((error) => {
  console.error(error)
  process.exit(1)
})
