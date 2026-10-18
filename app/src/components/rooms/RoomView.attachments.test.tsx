import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import * as attachmentHelpers from '../../lib/attachments'
import * as roomModes from '../../lib/room-modes'
import { defaultSettings } from '../../../electron/store/defaults'

let states: any[] = [], refs: any[] = [], effects: Array<{ deps: unknown[]; cleanup?: () => void }> = []
let cursor = 0, refCursor = 0, effectCursor = 0, pendingEffects: Array<() => void> = []
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const helperModule = { exports: {} as any }
const helperImports: Record<string, unknown> = {
  '../../lib/room-modes': roomModes,
  '../../lib/format': { relativeTime: () => '' }, '../../lib/policy': { allowedProviders: () => [], projectDefaultModel: () => null }, '../../lib/provider-accounts': { providerAccountChoices: () => [] }, 'react/jsx-runtime': jsx,
}
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'helpers.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, helperImports); if (reviewed.matched) return reviewed.value;  assert.ok(id in helperImports, `Reviewed helper import ${id}`); return helperImports[id] }, helperModule, helperModule.exports)
const roomHelpers = helperModule.exports
const components = new Proxy({}, { get: (_target, name) => name === '__esModule' ? true : () => null })
const AttachmentChips = () => null
let startCalls: any[] = [], rejectStart = false
const store: any = {
  rooms: ['A', 'B'].map((id) => ({ id, name: id, projectId: 'p', seats: [{ id: 'seat', name: 'Builder', providerId: 'provider', model: 'model' }], events: [], state: 'idle', problem: '', memory: [] })),
  projects: [{ id: 'p', name: 'Project', root: '/workspace' }], settings: defaultSettings(), roomStatus: { A: 'idle', B: 'idle' }, roomLive: {}, tickets: [], jobs: [], rightPanelHidden: true,
  startRoom: async (id: string, text: string, attachments: any[]) => { startCalls.push({ id, text, attachments }); store.roomStatus[id] = rejectStart ? 'error' : 'running' },
  steerRoom: async () => {},
}
const useStore = Object.assign((selector: any) => selector(store), { getState: () => store, setState: (patch: any) => Object.assign(store, patch) })
const imports: Record<string, unknown> = {
  react: {
    useState: (initial: any) => { const index = cursor++; if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial; return [states[index], (value: any) => { states[index] = typeof value === 'function' ? value(states[index]) : value }] },
    useRef: (initial: any) => { const index = refCursor++; return refs[index] ??= { current: initial } }, useEffect() {},
    useLayoutEffect: (effect: () => void | (() => void), deps: unknown[]) => { const index = effectCursor++; const previous = effects[index]; if (!previous || deps.some((item, i) => item !== previous.deps[i])) pendingEffects.push(() => { previous?.cleanup?.(); const cleanup = effect(); effects[index] = { deps, ...(cleanup ? { cleanup } : {}) } }) },
  },
  'react/jsx-runtime': jsx,
  '../../lib/store': { useStore }, '../../lib/icons': components, '../ui': components,
  '../AttachmentChips': { AttachmentChips }, '../../lib/attachments': attachmentHelpers,
  '../../lib/policy': { allowedProviders: () => [] }, '../../lib/voice': { speakText() {} },
  './helpers': roomHelpers, './RoomPanels': components, './RoomGitBanner': components,
  './RoomTimeline': components, './RoomPlans': components,
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'RoomView.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed import ${id}`); return imports[id] }, loaded, loaded.exports)
function render(id = 'A') { cursor = 0; refCursor = 0; effectCursor = 0; const node = loaded.exports.RoomView({ id }); const effects = pendingEffects; pendingEffects = []; effects.forEach((effect) => effect()); return node }
function all(node: any, predicate: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function fileInput(node: any) { return all(node, (entry) => entry.type === 'input' && entry.props.type === 'file')[0] }
function chips(node: any) { return all(node, (entry) => entry.type === AttachmentChips)[0] }
function prompt(node: any) { return all(node, (entry) => entry.type === 'textarea' && entry.props.rows === 1)[0] }
function start(node: any) { return all(node, (entry) => entry.type === 'button' && entry.props['aria-label'] === 'Start')[0] }
async function settle() { for (let i = 0; i < 12; i++) await Promise.resolve(); await new Promise<void>((resolve) => setImmediate(resolve)) }
render()
let node = render()
const broken = new File(['bad'], 'unreadable.txt', { type: 'text/plain' })
Object.defineProperty(broken, 'slice', { value: () => { throw new Error('selected file is unreadable') } })
fileInput(node).props.onChange({ target: { files: [new File(['first'], 'first.txt', { type: 'text/plain' }), broken, new File(['third'], 'third.txt', { type: 'text/plain' })], value: '' } })
await settle(); node = render()
assert.deepEqual(chips(node).props.items.map((item: any) => item.name), ['first.txt', 'unreadable.txt', 'third.txt'])
assert.deepEqual(chips(node).props.items.map((item: any) => item.status), ['ready', 'error', 'ready'])
prompt(node).props.onChange({ target: { value: 'Room task' } }); node = render()
assert.equal(start(node).props.disabled, true)
start(node).props.onClick(); await settle()
assert.equal(startCalls.length, 0)
assert.match(store.notice.message, /files need attention/)
const failedId = chips(node).props.items[1].id
chips(node).props.onRemove(failedId); node = render()
assert.equal(start(node).props.disabled, false)
rejectStart = true; start(node).props.onClick(); await settle(); node = render()
assert.equal(chips(node).props.items.length, 2, 'failed room launch retains successful files')
assert.equal(prompt(node).props.value, 'Room task')
store.roomStatus.A = 'idle'; rejectStart = false; node = render()
start(node).props.onClick(); await settle(); node = render()
assert.equal(startCalls.at(-1).attachments.length, 2)
assert.equal(chips(node), undefined)

// A file that finishes after navigation cannot move into the next room or resurrect a removed row.
store.roomStatus.A = 'idle'; node = render()
let release: (() => void) | undefined
const slow = new File(['slow'], 'slow.txt', { type: 'text/plain' })
Object.defineProperty(slow, 'slice', { value: () => ({ text: () => new Promise<string>((resolve) => { release = () => resolve('slow') }) }) })
fileInput(node).props.onChange({ target: { files: [slow], value: '' } }); node = render()
assert.equal(chips(node).props.items[0].status, 'processing')
node = render('B'); node = render('B'); release?.(); await settle(); node = render('B')
assert.equal(chips(node), undefined)
node = render('A'); node = render('A')
assert.equal(chips(node), undefined)
fileInput(node).props.onChange({ target: { files: [slow], value: '' } }); node = render()
const removedId = chips(node).props.items[0].id
chips(node).props.onRemove(removedId); release?.(); await settle(); node = render()
assert.equal(chips(node), undefined)
console.log('room files retain independent failures, block incomplete submission, retain failed launch inputs, and reject stale scope/removal completions')
