import assert from 'node:assert/strict'
import { harness, jsx, load, nodes, text } from '../reading/reading-test-harness'
import { resolveAssistantModel } from '../../lib/assistant-model'

const hooks = harness(), calls: any[] = [], cleared: unknown[] = [], opened: unknown[] = []
const draft: any = { text: 'Plan a useful afternoon', ready: true, scopeKey: 'fixture',
  capture: () => ({ text: draft.text }), clearAccepted: async (value: unknown) => { cleared.push(value) },
  setText: (value: string) => { draft.text = value } }
let reply: any = { ok: false, error: 'Fixture model is unavailable' }
const state: any = { settings: { defaultCwd: '', approvalMode: 'ask', activeProviderId: null, activeModel: null },
  threads: [], launch: async (payload: unknown) => { calls.push(payload); return reply },
  setView: (value: unknown) => opened.push(value), openThread: (id: string) => opened.push(id) }
const Composer = () => null
const { AssistantOverview } = load('src/components/assistant/AssistantOverview.tsx', {
  react: hooks.hooks, 'react/jsx-runtime': jsx,
  '../../lib/store': { useStore: (select: any) => select(state) },
  '../../lib/assistant-model': { resolveAssistantModel },
  '../../lib/use-conversation-draft': { useConversationDraft: () => draft },
  '../../lib/desktop-drafts': { desktopDraftPort: null },
  '../OriginalDiagnostic': { OriginalDiagnostic: 'diagnostic' }, '../ui': { Button: 'button' },
  './Composer': { Composer },
  '../work-activity/WorkActivityMenu': { WorkActivityMenu: () => null },
})
const render = () => hooks.render(() => AssistantOverview({ google: { snapshot: { accounts: [{
  id: 'account', email: 'private-person@example.com', status: 'connected' }] } }, onGoogle() {}, onInbox() {} }))
const composer = (tree: any) => nodes(tree).find(node => node.type === Composer)!
assert.equal(composer(render()).props.contextLabel, 'No inbox conversation is attached')
await composer(render()).props.onStartTask(); await new Promise(resolve => setImmediate(resolve))
assert.equal(calls.length, 0)
assert.match(text(render()), /Choose a model in Models/)
assert.equal(draft.text, 'Plan a useful afternoon')
state.settings = { ...state.settings, activeProviderId: 'fallback', activeModel: 'fallback-model',
  assistantProviderId: 'chosen', assistantProviderAccountId: 'chosen-account', assistantModel: 'chosen-model' }
await composer(render()).props.onStartTask(); await new Promise(resolve => setImmediate(resolve))
assert.equal(calls.length, 1)
assert.equal(calls[0].providerId, 'chosen')
assert.equal(calls[0].providerAccountId, 'chosen-account')
assert.equal(calls[0].model, 'chosen-model')
assert.ok(calls[0].text.includes(draft.text))
assert.match(calls[0].text, /draft-first/)
assert.equal(calls[0].persistActiveModel, false)
assert.doesNotMatch(JSON.stringify(calls[0]), /private-person/)
assert.equal(cleared.length, 0, 'a failed launch keeps the exact draft')
reply = { ok: true }
await composer(render()).props.onStartTask(); await new Promise(resolve => setImmediate(resolve))
assert.deepEqual(cleared, [{ text: 'Plan a useful afternoon' }])
hooks.cleanup()
console.log('Assistant overview preserves failed drafts, honors the Assistant model and never silently attaches account data')
