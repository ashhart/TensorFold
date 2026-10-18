import assert from 'node:assert/strict'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'
import type { InputHistoryScope } from '../../../shared/input-history'
const hooks = harness(), inserted: string[] = [], calls: string[] = []
const first: InputHistoryScope = { draft: { surface: 'chat', conversationId: 'one', hostId: null, incognito: false }, projectId: 'project', cwd: '/fixture', providerId: 'provider', accountId: null }
let scope = first, disabled = false, result: any, resolve!: (value: any) => void
const port = {
  getInputHistory: async () => { calls.push('get'); return { revision: 0, entries: [{ id: 'selected', text: 'Earlier input', createdAt: 1 }], partial: false } },
  recallInputHistory: async () => { calls.push('recall'); return await new Promise<string>(done => { resolve = done }) },
  clearInputHistory: async () => { calls.push('clear'); return { revision: 1, entries: [], partial: false } },
}
const { PromptHistoryPanel } = load('src/components/composer/PromptHistory.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, '../../lib/icons': { Icon: 'icon' }, '../ui': { Menu: 'menu' } })
const render = () => { result = hooks.render(() => PromptHistoryPanel({ scope, port, disabled, close() {}, onInsert: (value: string) => inserted.push(value) })); hooks.flush(); return result }
const click = (label: string) => { const node = find(render(), item => item.type === 'button' && (item.props['aria-label'] === label || text(item) === label)); assert.ok(node, label); node.props.onClick(); return node }
render(); await settle(); assert.ok(text(render()).includes('Earlier input'))
const insert = click('Insert'); insert.props.onClick(); assert.equal(calls.filter(value => value === 'recall').length, 1, 'Stale duplicate actions are guarded before entering an async operation')
resolve('Earlier input'); await settle(); assert.deepEqual(inserted, ['Earlier input'])
render(); await settle(); click('Insert'); disabled = true; render(); resolve('Blocked stale response'); await settle()
assert.deepEqual(inserted, ['Earlier input'], 'A run starting while recall is pending prevents a late insertion')
disabled = false; render(); click('Insert'); scope = { ...first, accountId: 'another-account' }; render(); resolve('Wrong account input'); await settle()
assert.deepEqual(inserted, ['Earlier input'], 'An old account callback cannot change the new account draft')
// A normal React key remount discards the old panel; exercise the cleanup fence directly too.
hooks.cleanup()
console.log('Actual recent-prompt panel handles duplicate recall, changed account and newly disabled state without sending or overwriting a draft')

const fresh = harness()
const { PromptHistoryPanel: FreshPanel } = load('src/components/composer/PromptHistory.tsx', { react: fresh.hooks, 'react/jsx-runtime': jsx, '../../lib/icons': { Icon: 'icon' }, '../ui': { Menu: 'menu' } })
const freshRender = () => { const tree = fresh.render(() => FreshPanel({ scope: first, port, onInsert: () => { throw new Error('Clearing must not modify the current draft') }, close() {} })); fresh.flush(); return tree }
freshRender(); await settle()
find(freshRender(), item => item.type === 'button' && text(item) === 'Clear recent prompts for this workspace and account').props.onClick()
await settle()
assert.ok(text(freshRender()).includes('No recent prompts'))
assert.equal(calls.filter(value => value === 'clear').length, 1)
fresh.cleanup()
