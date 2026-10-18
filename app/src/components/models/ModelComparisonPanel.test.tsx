import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import ts from 'typescript'
import { renderToStaticMarkup } from 'react-dom/server'
import { repoDisplayName } from '../../lib/turn-stats-line'
import type { EngineInstalledModel, EngineModelComparisonSnapshot } from '../../lib/types'

let states: any[] = [], refs: any[] = [], effects: Array<{ deps: unknown[]; cleanup?: () => void }> = []
let cursor = 0, refCursor = 0, effectCursor = 0, pendingEffects: Array<() => void> = []
let calls: any[] = [], cancelled: string[] = [], closed = 0, confirms = 0, accepted = false
let release: ((value: EngineModelComparisonSnapshot) => void) | null = null, reject: ((value: Error) => void) | null = null
let pollValue: EngineModelComparisonSnapshot | null = null
let timer: (() => void) | null = null
let engineEvent: ((value: any) => void) | null = null
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }, ...React.Children.toArray(props.children)), Fragment: React.Fragment }
const hooks = {
  useState(initial: any) { const i = cursor++; if (!(i in states)) states[i] = typeof initial === 'function' ? initial() : initial; return [states[i], (value: any) => { states[i] = typeof value === 'function' ? value(states[i]) : value }] },
  useRef(initial: any) { const i = refCursor++; return refs[i] ??= { current: initial } },
  useEffect(effect: () => void | (() => void), deps: unknown[]) { const i = effectCursor++, prior = effects[i]; if (!prior || deps.some((value, j) => value !== prior.deps[j])) pendingEffects.push(() => { prior?.cleanup?.(); const cleanup = effect(); effects[i] = { deps, ...(cleanup ? { cleanup } : {}) } }) },
}
const imports: Record<string, unknown> = {
  react: hooks, 'react/jsx-runtime': jsx,
  '../../lib/turn-stats-line': { repoDisplayName },
  './ModelsDrawer': { ModelsDrawer: ({ children, footer, ...props }: any) => React.createElement('section', { 'data-close-disabled': props.closeDisabled }, children, footer) },
  '../../lib/ipc': { api: {
    onEngineEvent: (handler: (value: any) => void) => { engineEvent = handler; return () => { engineEvent = null } },
    engineCompareModels: async (input: any) => { calls.push(input); return new Promise<EngineModelComparisonSnapshot>((resolve, rejectValue) => { release = resolve; reject = rejectValue }) },
    engineCancelComparison: async (id: string) => { cancelled.push(id); return true },
    engineComparisonStatus: async () => pollValue,
  } },
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'ModelComparisonPanel.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed import ${id}`); return imports[id] }, loaded, loaded.exports)
const models = ['Fixture/A', 'Fixture/B', 'Fixture/TooBig'].map((repo, index) => ({ repo, title: `Model ${index + 1}`, fitsMemory: index !== 2 })) as EngineInstalledModel[]
function render(locked = false, available = models) { cursor = 0; refCursor = 0; effectCursor = 0; const node = loaded.exports.ModelComparisonPanel({ models: available, activeModel: 'Fixture/A', locked, onClose: () => { closed++ } }); const queued = pendingEffects; pendingEffects = []; queued.forEach((effect) => effect()); return node }
function all(node: any, predicate: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate)), ...React.Children.toArray(node.props?.footer).flatMap((child) => all(child, predicate))] }
const button = (node: any, label: string) => { const value = all(node, (item) => item.type === 'button' && item.props.children === label)[0]; assert.ok(value, label); return value }
const prompt = (node: any) => all(node, (item) => item.type === 'textarea')[0]
const select = (node: any, label: string) => all(node, (item) => item.type === 'select' && item.props['aria-label'] === label)[0]
async function settle() { for (let i = 0; i < 12; i++) await Promise.resolve() }
function complete(id: string): EngineModelComparisonSnapshot { return { id, revision: 5, state: 'complete', startedAt: 1, finishedAt: 2, prompt: 'Synthetic', maxOutputTokens: 256, temperature: 0, previousModel: 'Fixture/Prior', restored: true, restoreError: null, rows: ['Fixture/A', 'Fixture/B'].map((model) => ({ model, state: 'complete', error: null, result: { model, text: 'Synthetic result', finishReason: 'length', elapsedMs: 1, loadMs: 2, firstTokenMs: null, decodeTokensPerSecond: null, completionTokens: null, promptTokens: null, cachedTokens: null, peakMemoryBytes: null, prefillTokensPerSecond: null, profile: null, startedAt: 1, finishedAt: 2 } })) as EngineModelComparisonSnapshot['rows'] } }
const originalWindow = globalThis.window
Object.defineProperty(globalThis, 'window', { configurable: true, value: { confirm: () => { confirms++; return accepted }, setInterval: (value: () => void) => { timer = value; return 1 }, clearInterval: () => { timer = null } } })
try {
  let node = render(true)
  assert.equal(button(node, 'Compare models').props.disabled, true)
  button(node, 'Compare models').props.onClick(); await settle(); assert.equal(calls.length, 0)
  const untitled = [{ ...models[0], repo: 'mlx-community/Qwen3.5-2B-MLX-4bit', title: undefined }, { ...models[1], title: undefined }]
  node = render(false, untitled)
  const option = all(node, (item) => item.type === 'option' && item.props.value === untitled[0].repo)[0]
  assert.equal(option.props.children, 'Qwen3.5 2B'); assert.equal(option.props.title, untitled[0].repo)
  node = render()
  assert.doesNotMatch(renderToStaticMarkup(node), /Model 3/, 'incompatible model cannot be selected')
  select(node, 'Comparison model B').props.onChange({ target: { value: 'Fixture/A' } }); node = render()
  assert.equal(button(node, 'Compare models').props.disabled, true)
  select(node, 'Comparison model B').props.onChange({ target: { value: 'Fixture/B' } }); prompt(node).props.onChange({ target: { value: 'Unsent private comparison prompt' } }); node = render()
  button(node, 'Close').props.onClick(); assert.equal(confirms, 1); assert.equal(closed, 0)
  assert.equal(prompt(render()).props.value, 'Unsent private comparison prompt')
  button(node, 'Compare models').props.onClick(); button(node, 'Compare models').props.onClick(); await settle(); node = render()
  assert.equal(calls.length, 1, 'synchronous ref closes duplicate start admission')
  assert.deepEqual(calls[0].models, ['Fixture/A', 'Fixture/B']); assert.equal(calls[0].prompt, 'Unsent private comparison prompt')
  assert.equal(node.props.closeDisabled, true); assert.equal(prompt(node).props.disabled, true)
  button(node, 'Close').props.onClick(); assert.equal(closed, 0)
  pollValue = { ...complete(calls[0].id), state: 'restoring', restored: false }
  ;(timer as (() => void) | null)?.(); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /Restoring your previous model/)
  pollValue = { ...complete(calls[0].id), revision: 4, state: 'running', restored: false }
  ;(timer as (() => void) | null)?.(); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /Restoring your previous model/, 'late older polls cannot regress a newer host revision')
  button(node, 'Cancel comparison').props.onClick(); await settle(); assert.deepEqual(cancelled, [calls[0].id])
  ;(release as ((value: EngineModelComparisonSnapshot) => void) | null)?.(complete(calls[0].id)); await settle(); node = render()
  assert.equal(node.props.closeDisabled, false); assert.match(renderToStaticMarkup(node), /Not reported/); assert.doesNotMatch(renderToStaticMarkup(node), /winner|0 tokens\/s/)
  assert.match(renderToStaticMarkup(node), /previous model has been restored/)
  const untitledResults = render(false, models.map((model) => ({ ...model, title: undefined })))
  const resultTitle = all(untitledResults, (item) => item.type === 'h3')[0]
  assert.equal(resultTitle.props.title, 'Fixture/A'); assert.equal(React.Children.toArray(resultTitle.props.children).join(''), 'A · A')
  assert.match(renderToStaticMarkup(node), /Reply limit reached \(256 tokens\); this answer may be incomplete/)
  // Failure keeps selections and typed prompt, permits retry, and shows the real error.
  button(node, 'Compare models').props.onClick(); await settle()
  ;(reject as ((value: Error) => void) | null)?.(new Error('Synthetic busy request refused')); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /Synthetic busy request refused/); assert.equal(prompt(node).props.value, 'Unsent private comparison prompt')
  button(node, 'Compare models').props.onClick(); await settle(); node = render()
  const purged = calls.at(-1).id
  ;(engineEvent as ((value: any) => void) | null)?.({ kind: 'comparison-cleared' }); node = render()
  assert.match(renderToStaticMarkup(node), /cleared with local memory/)
  assert.notEqual(prompt(node).props.value, 'Unsent private comparison prompt')
  ;(release as ((value: EngineModelComparisonSnapshot) => void) | null)?.(complete(purged)); await settle(); node = render()
  assert.doesNotMatch(renderToStaticMarkup(node), /Synthetic result/, 'pre-purge IPC completion cannot resurrect erased results')
  button(node, 'Compare models').props.onClick(); await settle(); node = render()
  const last = calls.at(-1).id
  effects.forEach((effect) => effect.cleanup?.()); const frozen = structuredClone(states)
  await settle(); assert.ok(cancelled.includes(last), 'unmount cancels exactly its own job')
  ;(release as ((value: EngineModelComparisonSnapshot) => void) | null)?.(complete(last)); await settle(); assert.deepEqual(states, frozen, 'stale completion cannot update an unmounted panel')
} finally { Object.defineProperty(globalThis, 'window', { configurable: true, value: originalWindow }) }
console.log('Actual comparison renderer handlers: compatible selection, identical prompt, duplicate/locked guards, dirty close, polling/restoration, cancellation, honest unknown metrics, failed retry and unmount races passed.')
