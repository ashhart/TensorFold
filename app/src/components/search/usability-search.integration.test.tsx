import assert from 'node:assert/strict'
import React from 'react'
import { createConversationSearchClient } from '../../lib/conversation-search-client'
import { createMockConversationSearchApi } from '../../lib/mock/conversation-search'
import { scrollTranscriptTarget } from './scroll-transcript-target'
import * as events from './search-events'
import * as filters from '../../../shared/conversation-search-filters'
import { searchFiltersFromDraft, emptySearchFilterDraft } from './search-filter-draft'
import * as windowing from '../../lib/transcript-window'
import { find, jsx, load, nodes, settle, text } from '../reading/reading-test-harness'

function harness() {
  const values: any[] = [], dependencies: any[][] = [], effects: (() => void)[] = [], cleanups: any[] = []
  let cursor = 0
  const changed = (index: number, deps?: any[]) => !deps || !dependencies[index] || deps.some((value, position) => value !== dependencies[index][position])
  const hooks: any = {
    useState(initial: any) { const index = cursor++; if (!(index in values)) values[index] = typeof initial === 'function' ? initial() : initial; return [values[index], (next: any) => { values[index] = typeof next === 'function' ? next(values[index]) : next }] },
    useRef(initial: any) { const index = cursor++; return values[index] ??= { current: initial } },
    useMemo(build: any, deps: any[]) { const index = cursor++; if (changed(index, deps)) { values[index] = build(); dependencies[index] = deps }; return values[index] },
    useEffect(effect: any, deps?: any[]) { const index = cursor++; if (changed(index, deps)) { effects.push(() => { cleanups[index]?.(); cleanups[index] = effect() }); if (deps) dependencies[index] = deps } },
    useLayoutEffect(effect: any, deps?: any[]) { hooks.useEffect(effect, deps) },
  }
  return { hooks, render<T>(build: () => T): T { cursor = 0; return build() }, flush() { while (effects.length) effects.shift()!() }, cleanup() { cleanups.forEach((cleanup) => cleanup?.()) } }
}
const original = new Map(['window', 'document', 'setTimeout', 'clearTimeout', 'requestAnimationFrame', 'cancelAnimationFrame'].map((key) => [key, Object.getOwnPropertyDescriptor(globalThis, key)]))
const listeners = new Map<string, Set<any>>(), timers = new Map<number, () => void>(), frames = new Map<number, () => void>()
let serial = 0
const elements = new Map<string, any>()
const win = { addEventListener(name: string, callback: any) { if (!listeners.has(name)) listeners.set(name, new Set()); listeners.get(name)!.add(callback) }, removeEventListener(name: string, callback: any) { listeners.get(name)?.delete(callback) }, dispatchEvent(event: any) { listeners.get(event.type)?.forEach((callback) => callback(event)) } }
for (const [key, value] of Object.entries({ window: win, document: { body: {}, querySelector: () => null, getElementById: (id: string) => elements.get(id) ?? null }, setTimeout: (callback: () => void) => { const id = ++serial; timers.set(id, callback); return id }, clearTimeout: (id: number) => timers.delete(id), requestAnimationFrame: (callback: () => void) => { const id = ++serial; frames.set(id, callback); return id }, cancelAnimationFrame: (id: number) => frames.delete(id) })) Object.defineProperty(globalThis, key, { configurable: true, value })
const tickTimers = () => { const callbacks = [...timers.values()]; timers.clear(); callbacks.forEach((callback) => callback()) }
const tickFrames = () => { const callbacks = [...frames.values()]; frames.clear(); callbacks.forEach((callback) => callback()) }
const pending: Array<{ request: any; resolve(result: any): void }> = [], opened: string[] = []
const store: any = { view: { name: 'thread', id: 'current' }, projects: [], openThread: (id: string) => opened.push(id), setNotice() {} }
const hooks = harness()
const { ConversationSearchHost } = load('src/components/search/ConversationSearchHost.tsx', {
  './SearchFilterControls': { SearchFilterControls: 'filter-controls' }, './search-filter-draft': { searchFiltersFromDraft, emptySearchFilterDraft }, '../../../shared/conversation-search-filters': filters,
  react: hooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node },
  '../../lib/preferences/hooks': { useUiText: () => () => 'Search messages' }, '../../lib/ipc': { api: { platform: 'browser', conversationSearch: (request: any) => new Promise((resolve) => pending.push({ request, resolve })) } },
  '../../lib/store': { useStore: Object.assign((select: any) => select(store), { getState: () => store }) }, '../../lib/conversation-search-client': { createConversationSearchClient },
  '../navigation/useAccessibleDialog': { useAccessibleDialog() {} }, '../ui': { Button: 'button' }, './SearchResultRow': { SearchResultRow: 'result-fixture' }, './search-events': events,
})
let navigationBlocked = false
const render = () => hooks.render(() => ConversationSearchHost({ navigationBlocked }))
const commit = async () => { let tree = render(); hooks.flush(); await settle(); tree = render(); return tree }
const input = () => find(render(), (node) => node.type === 'input' && node.props.role === 'combobox')
const match = (query: string, threadId = 'visible'): any => ({ threadId, title: `Synthetic ${query}`, kind: 'chat', projectId: null, updatedAt: 1, itemId: `item-${query}`, source: 'timeline', excerpt: `Visible ${query}`, highlights: [{ start: 8, end: 8 + query.length }], score: 1 })
const resolve = (index: number, query: string, matches = [match(query)]) => pending[index].resolve({ query, scope: pending[index].request.scope, matches, omitted: 0, truncated: false })
render(); hooks.flush()
events.requestConversationSearch({ scope: 'all', query: 'alpha' }); await commit(); tickTimers(); await commit()
resolve(0, 'alpha'); await settle(); let tree = await commit()
const alpha = find(tree, (node) => node.type === 'result-fixture'); assert.ok(alpha)
input().props.onChange({ target: { value: 'beta' } })
tree = render(); assert.equal(nodes(tree).some((node) => node.type === 'result-fixture'), false, 'Old snippets disappear in the first render before passive cleanup after query changes')
hooks.flush(); alpha.props.onOpen(); assert.deepEqual(opened, [], 'A stale result callback cannot navigate after its client epoch was cleared')
tickTimers(); await commit(); resolve(1, 'beta'); await settle(); tree = await commit(); assert.ok(find(tree, (node) => node.type === 'result-fixture' && node.props.match.excerpt === 'Visible beta'))
const betaRow = find(tree, (node) => node.type === 'result-fixture')
const controls = find(tree, (node) => node.type === 'filter-controls')
controls.props.onChange({ ...emptySearchFilterDraft(), model: 'synthetic-model', attachments: true })
tree = render(); assert.equal(nodes(tree).some((node) => node.type === 'result-fixture'), false, 'Changing a filter hides old snippets synchronously')
betaRow.props.onOpen(); assert.deepEqual(opened, [], 'A callback from an old filter cannot navigate even before passive cleanup')
hooks.flush(); tickTimers(); await commit(); assert.deepEqual(pending[2].request.filters, { model: 'synthetic-model', hasAttachments: true })
resolve(2, 'beta'); await settle(); tree = await commit()
const scope = find(tree, (node) => node.type === 'select' && node.props['aria-label'] === 'Conversation search scope')
scope.props.onChange({ target: { value: 'thread' } }); tree = render()
assert.equal(nodes(tree).some((node) => node.type === 'result-fixture'), false, 'Global snippets disappear before thread-scope effects run')
hooks.flush(); tickTimers(); await commit(); resolve(3, 'beta', [match('beta', 'wrong-thread')]); await settle(); tree = await commit()
assert.equal(nodes(tree).some((node) => node.type === 'result-fixture'), false); assert.match(text(tree), /could not finish/)
input().props.onChange({ target: { value: 'late' } }); await commit(); tickTimers(); await commit()
find(render(), (node) => node.type === 'button' && text(node) === 'Close search').props.onClick(); await commit()
resolve(4, 'late', [match('late', 'current')]); await settle(); assert.equal(render(), null, 'Closing search discards delayed private snippets')
hooks.cleanup()

const anchorHooks = harness(), windowHooks = harness(), revealed: string[] = [], focused: string[] = []
const { useConversationSearchTarget } = load('src/components/search/useConversationSearchTarget.ts', { react: anchorHooks.hooks, './search-events': events, './scroll-transcript-target': { scrollTranscriptTarget } })
const { useTranscriptWindow } = load('src/components/search/useTranscriptWindow.ts', { react: windowHooks.hooks, '../../lib/transcript-window': windowing })
const turn: any = { key: 'work-turn', work: [{ id: 'exact-tool', kind: 'tool', output: 'Synthetic recorded work.' }] }
let rows: any[] = Array.from({ length: 900 }, (_, index) => index === 23 ? { type: 'work', turn } : { type: 'item', item: { id: `message-${index}` } })
const outerViewport = { scrollTop: 133 }
const viewport: any = { scrollTop: 250, scrollHeight: 3000, clientHeight: 600, getBoundingClientRect: () => ({ top: 50 }), contains: (node: any) => node.inside === true }
const scrollRef = { current: viewport }, followRef = { current: true }
let currentThread = 'anchor-thread'
const exact: any = { inside: true, getBoundingClientRect: () => ({ top: 1000, height: 40 }), focus: (options: any) => { assert.equal(options.preventScroll, true); focused.push('exact-tool') }, scrollIntoView: () => { outerViewport.scrollTop = 777 } }
const anchorInput = () => ({ threadId: currentThread, timeline: turn.work, rows, scrollRef, followRef, revealWork: (key: string) => { revealed.push(key); elements.set('ti-exact-tool', exact) } })
const renderAnchor = () => anchorHooks.render(() => useConversationSearchTarget(anchorInput()))
renderAnchor(); anchorHooks.flush()
events.revealConversationItem({ threadId: currentThread, itemId: 'exact-tool' }, 'outline')
let anchor = renderAnchor(); anchorHooks.flush(); anchor = renderAnchor()
const renderWindow = () => windowHooks.render(() => useTranscriptWindow(currentThread, rows, anchor.itemId, scrollRef))
let window = renderWindow(); windowHooks.flush()
assert.equal(followRef.current, false); assert.deepEqual(revealed, ['work-turn']); assert.ok(window.segments.some((segment: any) => segment.kind === 'visible' && segment.start <= 23 && segment.end > 23), 'An early collapsed work match enters the actual long-history window')
tickFrames(); assert.deepEqual(focused, ['exact-tool']); assert.equal(exact.tabIndex, -1); assert.equal(outerViewport.scrollTop, 133, 'Source reveal must not scroll an enclosing app viewport or displace its docked composer'); assert.equal(viewport.scrollTop, 920, 'Only the transcript scroll container is centered on the exact source')
const total = rows.length; rows = [...rows, { type: 'item', item: { id: 'streamed-later' } }]
window = renderWindow(); windowHooks.flush()
assert.ok(window.segments.some((segment: any) => segment.kind === 'visible' && segment.start <= 23 && segment.end > 23)); assert.ok(window.segments.some((segment: any) => segment.kind === 'visible' && segment.end === total + 1), 'Streaming appends retain earlier searched rows')
const beforeLoad = viewport.scrollTop; window.load(0, 150); viewport.scrollHeight += 400; renderWindow(); windowHooks.flush(); assert.equal(viewport.scrollTop, beforeLoad + 400, 'Loading earlier rows preserves the reading position')
currentThread = 'different-thread'; anchor = renderAnchor(); anchorHooks.flush(); anchor = renderAnchor(); assert.equal(anchor.itemId, null)
rows = Array.from({ length: 900 }, (_, index) => ({ type: 'item', item: { id: `revision-${index}` } }))
renderWindow(); windowHooks.flush(); window = renderWindow()
assert.equal(window.segments.filter((segment: any) => segment.kind === 'visible').reduce((sum: number, segment: any) => sum + segment.end - segment.start, 0), 150)
rows = rows.slice(0, 20)
window = renderWindow(); windowHooks.flush(); window = renderWindow()
assert.equal(window.segments.filter((segment: any) => segment.kind === 'visible').reduce((sum: number, segment: any) => sum + segment.end - segment.start, 0), 20, 'Editing or reverting to a shorter revision keeps its surviving conversation visible rather than hiding every row behind an old window')
anchorHooks.cleanup(); windowHooks.cleanup()

const ordinary: any = { id: 'ordinary', title: 'Synthetic chat', kind: 'chat', updatedAt: 1, timeline: [{ id: 'unicode', kind: 'assistant', text: 'Café Привет 東京 ＡＢＣ' }] }
const privateKinds = ['incognito', 'phoneLocalOnly', 'googleLocalOnly', 'bot']
const mock = createMockConversationSearchApi(() => [ordinary, ...privateKinds.map((kind) => ({ ...ordinary, id: kind, [kind]: kind === 'bot' ? { presetId: 'fixture' } : true })), { ...ordinary, id: 'remote', orchestratorHostId: 'fixture-host' }])
const global = await mock.conversationSearch({ scope: 'all', query: 'Привет' }); assert.deepEqual([...new Set(global.matches.map((item) => item.threadId))], ['ordinary'])
await mock.conversationSearchSetActiveThread('incognito')
assert.deepEqual([...new Set((await mock.conversationSearch({ scope: 'thread', threadId: 'incognito', query: '東京' })).matches.map((item) => item.threadId))], ['incognito'])
await assert.rejects(mock.conversationSearch({ scope: 'thread', threadId: 'ordinary', query: 'Café' }), /current conversation/)
const compatibility = await mock.conversationSearch({ scope: 'all', query: 'ＡＢＣ' })
assert.equal(compatibility.matches.length, 1, 'Synthetic preview search must normalize both query and original text for compatibility-script forms without inventing highlight offsets')
for (const [name, descriptor] of original) { if (descriptor) Object.defineProperty(globalThis, name, descriptor); else Reflect.deleteProperty(globalThis, name) }
console.log('Actual search client/host and exact work reveal/window hooks pass stale snippets, scope/close fences, early-work anchors, retained streaming pages and reading position; synthetic mock keeps global privacy and Unicode compatibility forms.')
