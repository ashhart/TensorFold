import assert from 'node:assert/strict'
import { ModelQualificationService, validateQualificationInput } from './index'
import { runQualificationProbe } from './probe'
import type { ChatParams, StreamChunk } from '../openai'
import type { ModelQualificationReport } from '../../../shared/model-fit'

async function main() {
  let identity = 'a'.repeat(64), current = true
  let mode: 'valid' | 'malformed' | 'cancel' | 'changed' | 'oversize' = 'valid'
  let requests: ChatParams[] = []
  let saved: ModelQualificationReport[] = []
  const provider = { id: 'synthetic', name: 'Synthetic', kind: 'local' as const, baseUrl: 'http://127.0.0.1:9/v1', apiKey: '', models: ['fixture'] }
  const stream = async function* (params: ChatParams): AsyncGenerator<StreamChunk> {
    requests.push(params)
    assert.equal(params.messages.length, 1)
    assert.equal(params.messages[0].role, 'user')
    assert.equal('cwd' in params, false, 'Probe must never include a workspace')
    if (mode === 'changed') current = false
    if (mode === 'oversize') { yield { type: 'content', text: 'x'.repeat(9_000) }; return }
    if (mode === 'cancel') { await new Promise<void>(resolve => params.signal!.addEventListener('abort', () => resolve(), { once: true })); throw new Error('cancelled') }
    if (params.tools?.length) yield { type: 'tool_calls', toolCalls: [{ id: 'synthetic-tool', type: 'function', function: { name: 'tensorfold_qualification_echo', arguments: mode === 'malformed' ? '{broken' : '{"value":"tensorfold_fixed_probe_1"}' } }] }
    else if (Array.isArray(params.messages[0].content)) { assert.match(params.messages[0].content[1].type === 'image_url' ? params.messages[0].content[1].image_url.url : '', /^data:image\/png;base64,/); yield { type: 'content', text: 'red' } }
    else { if (params.reasoningEffort === 'low') yield { type: 'reasoning', text: 'Synthetic separate reasoning.' }; yield { type: 'content', text: 'OK' } }
  }
  const resolveRoute = async () => ({ provider, model: 'fixture', accountId: null, identity, allowedChecks: ['text', 'cancellation', 'tools', 'vision', 'reasoning'] as const as any, assertCurrent() { if (!current) throw new Error('Synthetic route changed') } })
  const service = new ModelQualificationService({ resolveRoute, stream, load: async () => saved, save: async reports => { saved = reports }, timeoutMs: 100, now: () => 1_000 })
  const input = { requestId: 'synthetic-first', providerId: provider.id, accountId: null, model: 'fixture', checks: ['text', 'cancellation', 'tools', 'vision', 'reasoning'] as const as any }
  const report = await service.check(input)
  for (const check of input.checks) assert.equal(report.checks[check as keyof typeof report.checks].state, 'observed', check)
  assert.equal(report.checks.cancellation.providerComputeStopped, false)
  assert.equal(report.checks.cancellation.transportDrained, true)
  assert.equal(report.checkedAt, 1_000)
  assert.equal(report.probeRevision, 1)
  const beforeRepeat = requests.length
  await service.check(input)
  assert.equal(requests.length, beforeRepeat, 'Repeated request identity cannot create another provider request')
  assert.equal(JSON.stringify(saved).includes('Synthetic separate reasoning'), false, 'No model output is persisted')
  mode = 'malformed'
  const bad = await service.check({ ...input, requestId: 'synthetic-malformed', checks: ['tools'] })
  assert.equal(bad.checks.tools.state, 'failed'); assert.match(bad.checks.tools.message, /malformed or schema-invalid/)
  assert.equal(bad.checks.text.state, 'observed', 'Independent prior text evidence is retained')
  mode = 'oversize'
  assert.equal((await service.check({ ...input, requestId: 'synthetic-bound', checks: ['text'] })).checks.text.state, 'failed')
  const oldIdentity = identity; identity = 'b'.repeat(64)
  assert.equal((await service.reports()).length, 0, 'Changed route identity invalidates old capability claims')
  await assert.rejects(service.check(input), /another route revision/)
  identity = oldIdentity
  mode = 'cancel'
  const pending = service.check({ ...input, requestId: 'synthetic-cancel', checks: ['text'] })
  await new Promise<void>(resolve => setImmediate(resolve))
  await assert.rejects(service.check({ ...input, requestId: 'other' }), /already running/)
  await service.cancel('synthetic-cancel')
  assert.equal((await pending).checks.text.state, 'cancelled')
  mode = 'changed'
  const beforeChanged = saved.length
  await assert.rejects(service.check({ ...input, requestId: 'synthetic-stale', checks: ['text'] }), /route changed/)
  assert.equal(saved.length, beforeChanged, 'A stale route cannot persist fresh-looking observations')
  current = true
  const restricted = new ModelQualificationService({ resolveRoute: async () => ({ ...await resolveRoute(), allowedChecks: ['text'] }), stream })
  requests = []
  const refused = await restricted.check({ ...input, requestId: 'synthetic-policy', checks: ['tools', 'vision'] })
  assert.equal(requests.length, 0); assert.equal(refused.checks.tools.state, 'unsupported'); assert.equal(refused.checks.vision.state, 'unsupported')
  assert.throws(() => validateQualificationInput({ ...input, baseUrl: 'https://example.invalid' } as any), /only a configured route/)
  assert.throws(() => validateQualificationInput({ ...input, checks: ['arbitrary'] } as any), /fixed-fixture checks/)
  const partial = await runQualificationProbe(provider, 'fixture', 'text', { signal: new AbortController().signal, assertCurrent() {}, stream: async function* () { yield { type: 'content', text: 'Partial synthetic answer' }; throw new Error('Synthetic connection loss') } })
  assert.equal(partial.state, 'failed', 'Partial output cannot become a completed capability observation')
  const restarted = new ModelQualificationService({ resolveRoute, load: async () => [{ ...report, state: 'requested', checks: { ...report.checks, tools: { state: 'requested', message: 'Interrupted' } } }] })
  const restartReports = await restarted.reports()
  assert.equal(restartReports[0].state, 'failed'); assert.equal(restartReports[0].checks.tools.state, 'unknown', 'Restart cannot promote an interrupted request to an observed claim')
  const unsupported = await runQualificationProbe({ ...provider, transport: 'codex-cli' }, 'fixture', 'text', { signal: new AbortController().signal, assertCurrent() {}, stream })
  assert.equal(unsupported.state, 'unsupported')
  const timed = await runQualificationProbe(provider, 'fixture', 'text', { signal: new AbortController().signal, assertCurrent() {}, timeoutMs: 5, stream: async function* (params) { await new Promise<void>(resolve => params.signal!.addEventListener('abort', () => resolve(), { once: true })); throw new Error('bounded') } })
  assert.equal(timed.state, 'failed'); assert.match(timed.message, /deadline/)
  console.log('PASS fixed-fixture qualification: independent text/tools/vision/reasoning/cancellation, malformed calls, no tool execution/output persistence, bounded cancellation, stale routes, permissions and no endpoint injection')
}
main().catch(error => { console.error(error); process.exitCode = 1 })
