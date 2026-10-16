// Run: jiti electron/agent/observer-hooks.test.ts
//
// Observer hooks (VST-012): sanitized bounded payloads, fail-open emission (throwing and
// rejecting hooks never break the emitter), visible auto-disable, AgentEvent mapping.
import assert from 'node:assert/strict'
import { createObserverHub, OBSERVER_DISABLE_AFTER, observerEventForAgentEvent, sanitizeObserverPayload } from './observer-hooks'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('observer-hooks tests:')

{
  const dirty = {
    tool: 'run_command',
    long: 'x'.repeat(2000),
    apiKey: 'sk-live-abcdefghijklmnopqrstuv',
    nested: { token: 'never', okay: 'fine', deeper: { too: { deep: true } } },
    email: 'Contact maya@example.com now',
    list: Array.from({ length: 50 }, (_, i) => i),
    fn: () => 'nope',
  }
  const clean = sanitizeObserverPayload(dirty)
  assert.equal((clean.long as string).length, 301, 'strings are bounded (300 + ellipsis)')
  assert.equal(clean.apiKey, undefined, 'secret-named keys never cross')
  assert.equal((clean.nested as Record<string, unknown>).token, undefined, 'nested secret keys are dropped too')
  assert.equal((clean.nested as Record<string, unknown>).okay, 'fine')
  assert.equal(((clean.nested as Record<string, unknown>).deeper as Record<string, unknown>).too, '[object]', 'depth is capped')
  assert.equal((clean.list as unknown[]).length, 10, 'arrays are capped')
  assert.ok(!(clean.email as string).includes('maya@example.com'), 'PII in values is redacted')
  assert.equal(clean.fn, undefined, 'functions never cross')
  ok('sanitize: bounded, secret keys dropped, values redacted, depth/array caps')
}

{
  const hub = createObserverHub({ now: () => 1000 })
  const seen: string[] = []
  hub.register('healthy', (event) => { seen.push(event.kind) })
  hub.register('thrower', () => { throw new Error('observer bug') })
  hub.register('rejector', async () => { throw new Error('async observer bug') })

  // Emission must never throw regardless of hook behavior — the run continues.
  assert.doesNotThrow(() => hub.emit('tool.start', { threadId: 't1', payload: { tool: 'read_file' } }))
  assert.deepEqual(seen, ['tool.start'], 'healthy hooks still fire when siblings fail')

  const settle = new Promise((resolve) => setTimeout(resolve, 20))
  await settle
  const status = hub.status()
  assert.equal(status.find((s) => s.name === 'healthy')?.delivered, 1)
  assert.equal(status.find((s) => s.name === 'thrower')?.failed, 1)
  assert.equal(status.find((s) => s.name === 'rejector')?.failed, 1)
  ok('fail-open: throwing and rejecting hooks are counted, never propagated')
}

{
  const hub = createObserverHub()
  hub.register('flaky', () => { throw new Error('always') })
  for (let i = 0; i < OBSERVER_DISABLE_AFTER + 5; i++) hub.emit('turn.start', { payload: {} })
  const status = hub.status()[0]
  assert.equal(status.disabled, true, 'a hook that keeps failing is disabled')
  assert.equal(status.failed, OBSERVER_DISABLE_AFTER, 'no further deliveries after disable')
  ok('auto-disable: persistent failure turns the hook off, visibly')
}

{
  const hub = createObserverHub()
  const seen: string[] = []
  const off = hub.register('once', (event) => { seen.push(event.kind) })
  hub.emit('session.start', { payload: {} })
  off()
  hub.emit('session.end', { payload: {} })
  assert.deepEqual(seen, ['session.start'])
  assert.equal(hub.recent().length, 2, 'the ring keeps recent events for the doctor')
  ok('unregister stops delivery; the recent ring keeps observing')
}

{
  assert.equal(observerEventForAgentEvent({ type: 'status', status: 'thinking', threadId: 't', runId: 'r' } as never)?.kind, 'turn.start')
  assert.equal(observerEventForAgentEvent({ type: 'status', status: 'streaming', threadId: 't', runId: 'r' } as never), null, 'streaming status is noise')
  assert.equal(observerEventForAgentEvent({ type: 'item_add', item: { id: 'i1', kind: 'tool', tool: 'read_file' }, threadId: 't', runId: 'r' } as never)?.kind, 'tool.start')
  assert.equal(observerEventForAgentEvent({ type: 'item_add', item: { id: 'i2', kind: 'assistant' }, threadId: 't', runId: 'r' } as never), null)
  assert.equal(observerEventForAgentEvent({ type: 'item_update', id: 'i1', patch: { state: 'ok' }, threadId: 't', runId: 'r' } as never)?.kind, 'tool.end')
  assert.equal(observerEventForAgentEvent({ type: 'item_update', id: 'i1', patch: { text: 'delta' }, threadId: 't', runId: 'r' } as never), null, 'text deltas never cross')
  assert.equal(observerEventForAgentEvent({ type: 'approval_request', callId: 'c', itemId: 'i', threadId: 't', runId: 'r' } as never)?.kind, 'approval.requested')
  assert.equal(observerEventForAgentEvent({ type: 'done', threadId: 't', runId: 'r' } as never)?.kind, 'turn.end')
  ok('mapper: structural facts only — deltas and transcripts never reach observers')
}

console.log(`  ${passed} passed`)
