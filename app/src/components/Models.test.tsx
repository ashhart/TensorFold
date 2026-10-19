import { reviewedLanguageImport } from './preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import { renderToStaticMarkup } from 'react-dom/server'
import { orderFavoritesFirst } from '../lib/favorite-models'
import * as hubFailure from '../../shared/hub-failure'

let slots: any[] = [], refs: any[] = [], effects: any[] = [], queued: any[] = []
let cursor = 0, refCursor = 0, effectCursor = 0
let busyLearning = false, requested: any = undefined, minimal = false
let calls: any[] = [], hold: (() => void) | null = null, delayLoad = false, failLoad = false
const model = (repo: string, title: string, fitsMemory = true) => ({ repo, title, fitsMemory, family: 'Synthetic family', format: '4-bit', weightsBytes: 1024, memoryFloorBytes: 2048, source: 'download', contextTokens: 32768 })
const snapshot: any = { status: { state: 'idle', available: true, model: null, totalMemoryBytes: 1e11, memoryLimitBytes: 7e10 }, installed: [model('fixture/Qwen', 'Qwen'), model('fixture/Nemotron', 'Nemotron'), model('fixture/Large', 'Large', false)], pulls: [], hubCache: '/synthetic/cache', suggestions: [{ repo: 'fixture/New', title: 'New model', revision: 'pinned-revision', note: '' }] }
const state: any = { view: { name: 'models' }, settings: { desktopOnboardingComplete: true }, threads: [], status: {}, setNotice: (notice: any) => calls.push(['notice', notice]), newChat: async () => calls.push(['chat']), openThread: (id: string) => calls.push(['thread', id]) }
const favorites: any = { favorites: [{ providerId: 'tensorfold', accountId: null, model: 'fixture/Nemotron' }], ready: true, error: null, isFavorite: (route: any) => favorites.favorites.some((item: any) => item.model === route.model), toggle: (route: any) => { calls.push(['star', route]); return true } }
const placeholders = Object.fromEntries(['HubRow', 'InstalledRow', 'EngineRuntimePanel', 'EngineApiPanel', 'LearningPanel', 'WorkersPanel', 'ModelRecommendationPanel', 'ModelDetailsDrawer', 'ModelComparisonPanel', 'ModelsDrawer', 'DownloadCentre', 'ModelFoldersSettings', 'ModelBenchmarkDrawer', 'ClusterWorkspace'].map((name) => [name, () => null])) as Record<string, any>
const Button = (props: any) => React.createElement('button', props, props.children)
const Input = (props: any) => React.createElement('input', props)
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const api = { enginePreflight: async (...args: any[]) => { calls.push(['check', ...args]); return { verdict: 'ready' } }, engineSearch: async (query: string) => { calls.push(['search', query]); return [{ repo: 'fixture/New' }] }, enginePull: async (...args: any[]) => { calls.push(['download', ...args]) }, engineUnload: async () => calls.push(['unload']), engineDelete: async (id: string) => calls.push(['delete', id]), engineCancelPull: async (id: string) => calls.push(['cancel', id]) }
const useStore = Object.assign((selector: any) => selector({ ...state, view: { name: 'models', tab: requested } }), { getState: () => state })
const imports: any = {
  react: {
    useState(initial: any) { const i = cursor++; if (!(i in slots)) slots[i] = typeof initial === 'function' ? initial() : initial; return [slots[i], (next: any) => { slots[i] = typeof next === 'function' ? next(slots[i]) : next }] },
    useRef(initial: any) { const i = refCursor++; return refs[i] ??= { current: initial } }, useCallback: (fn: any) => fn,
    useEffect(effect: any, deps: any[]) { const i = effectCursor++, old = effects[i]; if (!old || deps.some((value, index) => value !== old.deps[index])) queued.push(() => { old?.cleanup?.(); effects[i] = { deps, cleanup: effect() } }) },
  },
  'react/jsx-runtime': jsx, clsx: (...values: any[]) => values.filter(Boolean).join(' '), '../lib/ipc': { api }, '../lib/store': { useStore },
  '../lib/engine': { ENGINE_PROVIDER_ID: 'tensorfold', useEngine: () => ({ snapshot }), refreshEngine: async () => {}, loadEngineModel: async (repo: string) => { calls.push(['use', repo]); if (delayLoad) { delayLoad = false; await new Promise<void>((resolve) => { hold = resolve }) } if (failLoad) throw Error('Synthetic load failure'); return { state: 'ready', model: repo } } },
  '../lib/favorite-models': { useFavoriteModels: () => favorites, orderFavoritesFirst }, './ui': { Button, Input }, '../lib/icons': { Icon: () => null },
  './models/ModelRows': { HubRow: placeholders.HubRow, InstalledRow: placeholders.InstalledRow },
  '../lib/learning': { useLearning: () => ({ snapshot: {} }), learningBusy: () => busyLearning },
  './models/workers': { isLocalWorker: () => false }, '../lib/model-recommendation': { recommendEngineModel: () => ({ status: 'ready', hardwareLabel: 'Synthetic Mac', selected: { repo: 'fixture/New', title: 'New', action: 'download' } }) },
  './onboarding-model': { shouldShowOnboarding: () => false }, './models/download-display': { isDownloadPending: (pull: any) => ['queued', 'running', 'verifying'].includes(pull.state), downloadStateLabel: (pull: any) => pull.state }, '../lib/turn-stats-line': { repoDisplayName: (name: string) => name },
}
for (const name of Object.keys(placeholders)) if (!['HubRow', 'InstalledRow', 'ModelBenchmarkDrawer', 'ClusterWorkspace'].includes(name)) imports[`./${name === 'ModelFoldersSettings' ? 'settings' : 'models'}/${name}`] = { [name]: placeholders[name] }
imports['./clusters/ClusterWorkspace'] = { ClusterWorkspace: placeholders.ClusterWorkspace }
imports['../../shared/hub-failure'] = hubFailure
imports['./models/benchmark/ModelBenchmarkDrawer'] = { ModelBenchmarkDrawer: placeholders.ModelBenchmarkDrawer }
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'Models.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, id); return imports[id] }, loaded, loaded.exports)
function render() { cursor = 0; refCursor = 0; effectCursor = 0; const tree = loaded.exports.Models({ minimal }); const work = queued; queued = []; work.forEach((fn) => fn()); return tree }
function all(node: any, predicate: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function component(node: any, name: string) { return all(node, (item) => item.type === placeholders[name])[0] }
function button(node: any, label: string) { const items = all(node, (item) => (item.type === 'button' || item.type === Button) && renderToStaticMarkup(item).includes(label)); assert.ok(items.length, label); return items[0] }
function rows(node: any) { return all(node, (item) => item.type === placeholders.InstalledRow) }
async function settle() { for (let i = 0; i < 20; i++) await Promise.resolve(); await new Promise<void>((resolve) => setImmediate(resolve)) }
let node = render(); await settle(); node = render()
assert.equal(button(node, 'My models').props['aria-selected'], true)
button(node, 'Compute and clusters').props.onClick(); node = render(); assert.ok(component(node, 'ClusterWorkspace')); component(node, 'ClusterWorkspace').props.onCloseBlockedChange(true); node = render(); assert.equal(component(node, 'ModelsDrawer').props.closeDisabled, true); component(node, 'ClusterWorkspace').props.onCloseBlockedChange(false); node = render(); component(node, 'ModelsDrawer').props.onClose(); node = render(); assert.equal(component(node, 'ClusterWorkspace'), undefined)
assert.equal(component(node, 'EngineRuntimePanel').props.compact, true)
assert.equal(component(node, 'EngineApiPanel'), undefined, 'API credentials view is not mounted while the library is open')
assert.deepEqual(rows(node).map((entry) => entry.props.model.repo), ['fixture/Nemotron', 'fixture/Qwen', 'fixture/Large'])
const libraryPanel = all(node, (entry) => entry.props?.id === 'local-panel-Models')[0]
assert.equal(component(libraryPanel, 'ModelRecommendationPanel'), undefined, 'recommendations do not displace a populated library')
rows(node)[0].props.onFavorite(); assert.equal(calls.at(-1)[0], 'star'); assert.equal(calls.some((call) => call[0] === 'use'), false)
rows(node)[0].props.onSettings(); node = render(); assert.equal(component(node, 'ModelDetailsDrawer').props.model.repo, 'fixture/Nemotron'); component(node, 'ModelDetailsDrawer').props.onClose(); node = render()
button(node, 'Show favourite models only').props.onClick(); node = render(); assert.equal(rows(node).length, 1)
button(node, 'Show favourite models only').props.onClick(); node = render()
const search = all(node, (entry) => entry.type === Input && entry.props['aria-label'] === 'Search your models')[0]
search.props.onChange({ target: { value: 'qwen' } }); node = render(); assert.deepEqual(rows(node).map((entry) => entry.props.model.repo), ['fixture/Qwen'])
search.props.onChange({ target: { value: '' } }); node = render()
button(node, 'Benchmarks').props.onClick(); node = render(); assert.equal(component(node, 'ModelBenchmarkDrawer').props.models.length, 3); assert.equal(component(node, 'ModelBenchmarkDrawer').props.preferredModel, null); component(node, 'ModelBenchmarkDrawer').props.onClose(); node = render();
button(node, 'Compare models').props.onClick(); node = render(); assert.equal(component(node, 'ModelComparisonPanel').props.models.length, 3); component(node, 'ModelComparisonPanel').props.onClose(); node = render()
button(node, 'Discover').props.onClick(); node = render(); const input = all(node, (entry) => entry.type === Input && entry.props['aria-label'] === 'Search Hugging Face')[0]; input.props.onChange({ target: { value: 'fixture/New' } }); node = render(); all(node, (entry) => entry.type === 'form')[0].props.onSubmit({ preventDefault() {} }); await settle(); node = render(); component(node, 'HubRow').props.onDownload(); await settle(); assert.deepEqual(calls.at(-1), ['download', 'fixture/New', 'pinned-revision'])
button(node, 'API').props.onClick(); node = render(); assert.ok(component(node, 'EngineApiPanel')); button(node, 'My models').props.onClick(); node = render(); assert.equal(component(node, 'EngineApiPanel'), undefined)
delayLoad = true; const use = rows(node)[0].props.onLoad; use(); use(); assert.equal(calls.filter((call) => call[0] === 'use').length, 1); (hold as (() => void) | null)?.(); await settle(); assert.equal(calls.filter((call) => call[0] === 'chat').length, 1)
state.threads = [{ id: 'running', providerId: 'tensorfold', status: 'running' }]; state.status = { running: 'streaming' }; node = render(); assert.equal(button(node, 'Compare models').props.disabled, true); rows(node)[0].props.onLoad(); await settle(); assert.equal(calls.filter((call) => call[0] === 'use').length, 1)
button(node, 'Benchmarks').props.onClick(); node = render(); assert.match(component(node, 'ModelBenchmarkDrawer').props.blockedReason, /active local/); component(node, 'ModelBenchmarkDrawer').props.onClose();
state.threads = []; state.status = {}; busyLearning = true; node = render(); rows(node)[0].props.onLoad(); await settle(); assert.equal(calls.filter((call) => call[0] === 'use').length, 1)
busyLearning = false; failLoad = true; node = render(); rows(node)[0].props.onLoad(); await settle(); assert.equal(calls.filter((call) => call[0] === 'chat').length, 1); assert.match(calls.at(-1)[1].message, /Synthetic load failure/)
failLoad = false; delayLoad = true; node = render(); rows(node)[0].props.onLoad(); effects.forEach((effect) => effect?.cleanup?.()); (hold as (() => void) | null)?.(); await settle(); assert.equal(calls.filter((call) => call[0] === 'chat').length, 1, 'late load never navigates after unmount')
console.log('Models: library-first layout, scoped favourites/search/details, lazy API, pinned discovery downloads, comparison gates, single use/chat action and late/error guards')
