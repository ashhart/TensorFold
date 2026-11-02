import assert from 'node:assert/strict'
import { QuickAskSessionHost, type AskSavedState } from './session'
import type { QuickAskRoute, QuickAskEvent } from '../../../shared/quick-ask'
import type { AskGeneration, AskStep } from './generate'

const baseRoute: QuickAskRoute = { id: 'route', providerId: 'example', accountId: null, model: 'example', label: 'Example', destination: 'device', location: 'This computer', state: 'ready', vision: 'unknown', binding: 'a'.repeat(64) }
let route = { ...baseRoute }, leased = 0, calls: AskGeneration[] = [], saved: AskSavedState | null = null, failSave = false, role = true, hold = false
let started!: () => void
let startSignal = new Promise<void>(resolve => { started = resolve })
const events: QuickAskEvent[] = []
const host = new QuickAskSessionHost({
  authorize: () => { if (!role) throw new Error('role revoked') },
  reserve: () => { leased++; return { release: () => leased-- } },
  save: async state => { if (failSave) throw new Error('disk unavailable'); saved = structuredClone(state) },
  routes: async () => [route], emit: event => events.push(event),
  handoff: async state => `ask-${state.session.id}`,
  generate: async function* (input, signal): AsyncGenerator<AskStep> {
    calls.push(input); started()
    yield { type: 'content', text: 'Partial ' }
    if (hold) await new Promise<void>((resolve) => { if (signal.aborted) resolve(); else signal.addEventListener('abort', () => resolve(), { once: true }) })
    signal.throwIfAborted()
    yield { type: 'content', text: 'answer.' }; yield { type: 'message', message: { role: 'assistant', content: 'Partial answer.' } }; yield { type: 'finish' }
  },
})
async function draft(text: string) { return host.saveDraft({ ...host.snapshot().draft, text, routeId: route.id }) }
await draft('First question')
const stale = host.snapshot().draft
await draft('Latest question')
await assert.rejects(host.saveDraft(stale), /changed/)
await host.send({ revision: host.snapshot().draft.revision, approvedRouteId: null })
await host.cancel()
assert.equal(leased, 0)
assert.equal(host.snapshot().draft.text, 'Latest question', 'Cancellation keeps the original question')
assert.equal(host.snapshot().turns[0].status, 'cancelled')
await host.reset(); await draft('A real question')
await host.send({ revision: host.snapshot().draft.revision, approvedRouteId: null })
await startSignal
await new Promise(resolve => setTimeout(resolve, 15))
assert.equal(host.snapshot().turns[0].status, 'complete')
assert.equal(host.snapshot().draft.text, '')
assert.equal(host.snapshot().turns[0].answer, 'Partial answer.')
await draft('Follow up')
await host.send({ revision: host.snapshot().draft.revision, approvedRouteId: null })
await new Promise(resolve => setTimeout(resolve, 15))
assert.equal(calls.at(-1)?.messages.length, 3, 'Follow-ups keep the first user and assistant messages')
const id = await host.handoff()
assert.equal(await host.handoff(), id, 'Handoff retries are idempotent')
await assert.rejects(host.send({ revision: host.snapshot().draft.revision, approvedRouteId: null }), /moved/)
await host.reset(); route = { ...baseRoute, destination: 'cloud', location: 'Example cloud' }; await draft('Private context')
const before = calls.length
await assert.rejects(host.send({ revision: host.snapshot().draft.revision, approvedRouteId: null }), /approve/)
assert.equal(calls.length, before, 'Cloud refusal happens before a model request')
await host.send({ revision: host.snapshot().draft.revision, approvedRouteId: route.id })
await new Promise(resolve => setTimeout(resolve, 15))
route = { ...route, binding: 'b'.repeat(64) }; await draft('Should stay pinned')
await assert.rejects(host.send({ revision: host.snapshot().draft.revision, approvedRouteId: route.id }), /destination changed/)
await host.reset(); route = { ...baseRoute }; hold = true; await draft('Keep this if stopped')
startSignal = new Promise(resolve => { started = resolve })
await host.send({ revision: host.snapshot().draft.revision, approvedRouteId: null }); await startSignal
await assert.rejects(host.send({ revision: host.snapshot().draft.revision, approvedRouteId: null }), /already/)
await host.cancel()
assert.equal(host.snapshot().turns[0].answer, 'Partial ')
assert.match(String(host.saved().messages.at(-1)?.content), /Interrupted answer/, 'A follow-up or handoff retains the interrupted answer as context')
assert.equal(host.snapshot().draft.text, 'Keep this if stopped')
assert.equal(leased, 0)
const restored = new QuickAskSessionHost((host as any).deps)
restored.restore(saved)
assert.equal(restored.snapshot().turns[0].status, 'cancelled')
await host.reset(); failSave = true
await assert.rejects(draft('Unsaved edit'), /disk/)
assert.equal(host.snapshot().draft.text, 'Unsaved edit', 'A disk failure retains the in-memory draft')
failSave = false; role = false
await assert.rejects(draft('No access'), /role/)
role = true; await host.quiesce()
await assert.rejects(draft('Stopping'), /stopping/)
host.resume(); await host.purge()
assert.equal(host.snapshot().turns.length, 0)
assert.equal(host.snapshot().draft.text, '')
console.log('Quick Ask state, cancellation, privacy, follow-up, persistence and handoff checks passed')
