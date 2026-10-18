import { reviewedLanguageImport } from './preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import { renderToStaticMarkup } from 'react-dom/server'

let states: any[] = [], refs: any[] = [], effects: any[] = [], queued: any[] = []
let cursor = 0, refCursor = 0, effectCursor = 0
let selected = 'A', fail = '', hold: (() => void) | null = null, holdMutation = false, holdRead = false, failRead = false
const calls: any[] = []
const entries: any[] = ['A', 'B'].map((projectId) => ({ id: `m${projectId}`, projectId, text: `Fact for ${projectId}`, kind: 'convention', confidence: 'high', privacyScope: 'project', sourceType: 'manual', sourceRef: 'synthetic.txt', createdAt: 10, updatedAt: 20, trashedAt: null, stateRevision: 1, revisions: [] }))
let candidates: any[] = []
const api: any = {
  platform: 'darwin',
  getProjectMemory: async (id: string) => { if (failRead) throw new Error('Synthetic refresh failed'); const result = structuredClone(entries.filter((item) => item.projectId === id && item.trashedAt == null)); if (holdRead && id === 'A') { holdRead = false; await new Promise<void>((resolve) => { hold = resolve }) } return result },
  getProjectMemoryTrash: async (id: string) => structuredClone(entries.filter((item) => item.projectId === id && item.trashedAt != null)),
  getProjectMemoryCandidates: async (id: string) => candidates.filter((item) => item.projectId === id),
  memoryCandidateDiff: async (id: string) => ({ candidateId: id, action: 'create', changes: [] }),
  async trashProjectMemory(input: any) { calls.push(['trash', input]); if (fail) throw new Error(fail); if (holdMutation) { holdMutation = false; await new Promise<void>((resolve) => { hold = resolve }) } const entry = entries.find((item) => item.id === input.entryId); entry.trashedAt = 50; entry.stateRevision++; return entry },
  async restoreProjectMemory(input: any) { calls.push(['restore', input]); const entry = entries.find((item) => item.id === input.entryId); entry.trashedAt = null; entry.stateRevision++; return entry },
  async deleteProjectMemory(id: string, guard: any) { calls.push(['forget', id, guard]); entries.splice(entries.findIndex((item) => item.id === id), 1) },
  async saveProjectMemory(input: any) { calls.push(['save', input]); if (fail) throw new Error(fail); Object.assign(entries.find((item) => item.id === input.id), input) },
  async rollbackMemoryEntry(id: string) { calls.push(['rollback', id]) },
  async approveProjectMemoryCandidate(id: string) { calls.push(['approve', id]); candidates = candidates.filter((item) => item.id !== id) },
  async rejectProjectMemoryCandidate(id: string) { calls.push(['reject', id]); candidates = candidates.filter((item) => item.id !== id) },
}
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const imports: Record<string, unknown> = {
  react: {
    useState(initial: any) { const i = cursor++; if (!(i in states)) states[i] = typeof initial === 'function' ? initial() : initial; return [states[i], (value: any) => { states[i] = typeof value === 'function' ? value(states[i]) : value }] },
    useRef(initial: any) { const i = refCursor++; return refs[i] ??= { current: initial } },
    useEffect(effect: any, deps: any[]) { const i = effectCursor++, previous = effects[i]; if (!previous || deps.some((value, index) => value !== previous.deps[index])) queued.push(() => { previous?.cleanup?.(); effects[i] = { deps, cleanup: effect() } }) },
  },
  'react/jsx-runtime': jsx, '../lib/ipc': { api }, '../lib/icons': { Icon: () => null },
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'MemoryEntries.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed import ${id}`); return imports[id] }, loaded, loaded.exports)
function render(id = selected) { selected = id; cursor = 0; refCursor = 0; effectCursor = 0; const node = loaded.exports.MemoryEntries({ projectId: id }); const work = queued; queued = []; work.forEach((effect) => effect()); return node }
function all(node: any, predicate: any): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function text(node: any): string { return renderToStaticMarkup(node) }
function button(node: any, label: string) { const match = all(node, (item: any) => item.type === 'button' && renderToStaticMarkup(item).includes(label)); assert.ok(match.length, label); return match[0] }
function textarea(node: any) { return all(node, (item: any) => item.type === 'textarea')[0] }
async function settle() { for (let i = 0; i < 20; i++) await Promise.resolve(); await new Promise<void>((resolve) => setImmediate(resolve)) }
let confirmed = false
Object.assign(globalThis, { confirm: () => confirmed })
render(); await settle(); let node = render()
assert.match(text(node), /Fact for A/); assert.doesNotMatch(text(node), /Fact for B/)
fail = 'Synthetic encrypted write rejected'; button(node, 'Move to Trash').props.onClick(); await settle(); node = render()
assert.match(text(node), /Synthetic encrypted write rejected/); assert.match(text(node), /Fact for A/)
fail = ''; failRead = true; button(node, 'Move to Trash').props.onClick(); await settle(); node = render()
assert.doesNotMatch(text(node), /Fact for A/); assert.match(text(node), /Synthetic refresh failed/); assert.match(text(node), /Moved to Trash/)
failRead = false; button(node, 'Retry').props.onClick(); await settle(); node = render()
assert.deepEqual(calls.at(-1), ['trash', { projectId: 'A', entryId: 'mA', expectedRevision: 1 }])
assert.doesNotMatch(text(node), /Fact for A/); button(node, 'Trash (1)').props.onClick(); node = render()
assert.match(text(node), /Fact for A/); assert.match(text(node), /synthetic.txt/)
button(node, 'Restore').props.onClick(); await settle(); node = render()
assert.deepEqual(calls.at(-1), ['restore', { projectId: 'A', entryId: 'mA', expectedRevision: 2 }])
button(node, 'Active (1)').props.onClick(); node = render(); button(node, 'Edit').props.onClick(); node = render()
textarea(node).props.onChange({ target: { value: 'Revised A' } }); node = render(); fail = 'Save failed'
button(node, 'Save').props.onClick(); await settle(); node = render(); assert.equal(textarea(node).props.value, 'Revised A'); assert.match(text(node), /Save failed/)
fail = ''; button(node, 'Save').props.onClick(); await settle(); node = render()
assert.equal(textarea(node), undefined); assert.match(text(node), /Revised A/)
assert.deepEqual(Object.keys(calls.at(-1)[1]).sort(), ['confidence', 'id', 'kind', 'privacyScope', 'projectId', 'text'])
// A pending operation cannot clear B's editor or cross project boundaries; duplicates are synchronous guarded.
button(node, 'Edit').props.onClick(); node = render()
holdMutation = true; const staleTrash = button(node, 'Move to Trash'); staleTrash.props.onClick(); staleTrash.props.onClick()
const count = calls.filter((call) => call[0] === 'trash').length
node = render('B'); assert.doesNotMatch(text(node), /Revised A/); await settle(); node = render(); button(node, 'Edit').props.onClick(); node = render()
textarea(node).props.onChange({ target: { value: 'B unfinished' } }); node = render(); (hold as (() => void) | null)?.(); await settle(); node = render()
assert.equal(calls.filter((call) => call[0] === 'trash').length, count)
assert.equal(textarea(node).props.value, 'B unfinished'); assert.doesNotMatch(text(node), /Revised A/)
// A stale read resolves last, but B's loaded state remains visible.
holdRead = true; render('A'); node = render('B'); await settle(); (hold as (() => void) | null)?.(); await settle(); node = render()
assert.match(text(node), /Fact for B/); assert.doesNotMatch(text(node), /Revised A/)
node = render('A'); await settle(); node = render(); assert.equal(textarea(node), undefined); button(node, 'Trash (1)').props.onClick(); node = render()
button(node, 'Forget permanently').props.onClick(); await settle(); assert.equal(calls.some((call) => call[0] === 'forget'), false)
confirmed = true; button(node, 'Forget permanently').props.onClick(); await settle(); node = render()
assert.deepEqual(calls.at(-1), ['forget', 'mA', { projectId: 'A', expectedRevision: 4 }]); assert.match(text(node), /Trash is empty/)
api.platform = 'web-remote'; node = render(); assert.match(text(node), /desktop app on its owning host/); assert.equal(all(node, (item: any) => item.type === 'button').length, 0)
console.log('memory UI: durable failure, Trash/Restore CAS, scoped Forget, retained failed edits, duplicate actions and stale project loads/completions, remote ownership')
