import { reviewedLanguageImport } from '../../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import { createBenchmarkController } from './benchmark-controller'
import * as display from './benchmark-display'
import { fixtureCell, fixtureModel, fixtureRun } from './benchmark-test-fixtures'
import { formatBytes } from '../../../lib/engine-format'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = (_props: any) => null, Toggle = (_props: any) => null, Frame = (_props: any) => null, Results = (_props: any) => null, Metadata = (_props: any) => null
let current = null as ReturnType<typeof fixtureRun> | null
let history: ReturnType<typeof fixtureRun>[] = []
const calls: any[] = []
const controller = createBenchmarkController({ status: async () => current, history: async () => history, start: async (input) => { calls.push(['start', input]); current = fixtureRun({ ...input, serial: !!input.serial, state: 'preparing', revision: 0 }); return current }, cancel: async (id) => { calls.push(['cancel', id]); return true }, export: async (id) => { calls.push(['export', id]); return { saved: true } }, copy: async (text) => { calls.push(['copy', text]); return { ok: true } } }, () => {})
const slots: any[] = []
let cursor = 0
const imports: Record<string, unknown> = {
  react: { useState: (initial: any) => { const index = cursor++; if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial; return [slots[index], (next: any) => { slots[index] = typeof next === 'function' ? next(slots[index]) : next }] }, useEffect: (work: () => void) => work() },
  'react/jsx-runtime': jsx, '../../ui': { Button, Toggle }, '../ModelsDrawer': { ModelsDrawer: Frame }, './BenchmarkResults': { BenchmarkResults: Results }, './BenchmarkMetadata': { BenchmarkMetadata: Metadata }, './benchmark-display': display,
  './useModelBenchmark': { useModelBenchmark: () => ({ state: controller.snapshot(), controller }) }, '../../../lib/ipc': { api: { openExternal: async (url: string) => calls.push(['community', url]) } },
}
function load(file: string, bindings: Record<string, unknown>) {
  const output = ts.transpileModule(readFileSync(path.join(__dirname, file + '.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
  const loaded = { exports: {} as any }
  new Function('require', 'module', 'exports', output)((name: string) => { const reviewed = reviewedLanguageImport(name, bindings); if (reviewed.matched) return reviewed.value;  assert.ok(name in bindings, `Reviewed benchmark UI import ${name}`); return bindings[name] }, loaded, loaded.exports)
  return loaded.exports[file]
}
const Drawer = load('ModelBenchmarkDrawer', imports)
const props = { models: [fixtureModel('fixture/linked', { source: 'library' }), fixtureModel('fixture/too-large', { fitsMemory: false }), fixtureModel()], activeModel: 'fixture/cached-model', blockedReason: null as string | null, onClose: () => calls.push(['close']) }
function render(extra = {}) { cursor = 0; return Drawer({ ...props, ...extra }) }
function all(node: any, predicate: (item: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate)), ...all(node.props?.footer, predicate)] }
const button = (node: any, label: string) => { const found = all(node, (item) => item.type === Button && item.props.children === label)[0]; assert.ok(found, label); return found }
async function settle() { for (let i = 0; i < 12; i++) await Promise.resolve() }
await controller.refresh()
let view = render()
assert.equal(all(view, (item) => item.type === 'select' && item.props['aria-label'] === 'Benchmark model')[0].props.value, 'fixture/cached-model')
assert.equal(all(view, (item) => item.type === 'option').length, 2, 'only compatible cached target plus placeholder is offered')
assert.equal(calls.length, 0, 'drawer mount never starts work, exports, copies or opens the community page')
assert.equal(button(render({ blockedReason: 'Finish active local work.' }), 'Run benchmark').props.disabled, true)
button(render({ blockedReason: 'Finish active local work.' }), 'Run benchmark').props.onClick(); assert.equal(calls.length, 0)
all(view, (item) => item.type === Toggle)[0].props.onChange(true)
view = render(); button(view, 'Run benchmark').props.onClick(); button(view, 'Run benchmark').props.onClick(); await settle()
assert.equal(calls.filter((call) => call[0] === 'start').length, 1)
assert.equal(calls[0][1].serial, true)
assert.equal(calls[0][1].model, 'fixture/cached-model')
view = render(); button(view, 'Close and keep running').props.onClick()
assert.equal(calls.filter((call) => call[0] === 'cancel').length, 0)
slots.length = 0; view = render(); view = render()
assert.equal(all(view, (item) => item.type === Toggle)[0].props.checked, true, 'reopened controls reflect the active serial job rather than new defaults')
assert.equal(all(view, (item) => item.type === 'select' && item.props['aria-label'] === 'Benchmark model')[0].props.value, current!.model)
button(view, 'Cancel benchmark').props.onClick(); await settle()
assert.equal(calls.filter((call) => call[0] === 'cancel').length, 1)
current = fixtureRun({ id: current!.id, state: 'failed', restored: false, restoreError: 'Synthetic restore failure', protocolComplete: false, saved: false, cells: [{ ...fixtureCell(), complete: false, successful: 1, errors: 4, deliveryTokensPerSecond: { median: 999, min: 998, max: 1000, spread: 2 } }] })
history = [current, fixtureRun({ id: 'benchmark_history_fixture_0123' })]
await controller.poll(); view = render()
assert.equal(all(view, (item) => item.type === Results)[0].props.run.state, 'failed')
button(view, 'Copy basic results').props.onClick(); await settle()
button(view, 'Review and export JSON').props.onClick(); await settle()
assert.equal(calls.filter((call) => call[0] === 'copy').length, 1)
assert.equal(calls.filter((call) => call[0] === 'export').length, 1)
button(render(), 'Community benchmarks').props.onClick(); await settle()
assert.deepEqual(calls.at(-1), ['community', 'https://tensorfold.dev/benchmarks'])
all(render(), (item) => item.type === 'select' && item.props['aria-label'] === 'Benchmark result history')[0].props.onChange({ target: { value: history[1].id } })
assert.equal(all(render(), (item) => item.type === Results)[0].props.run.id, history[1].id)
controller.dispose()

const ResultsView = load('BenchmarkResults', { 'react/jsx-runtime': jsx, './benchmark-display': display })
const failedMarkup = renderToStaticMarkup(React.createElement(ResultsView, { run: current }))
assert.match(failedMarkup, /Incomplete/); assert.match(failedMarkup, /Not reported/); assert.match(failedMarkup, /Restoration failed/); assert.doesNotMatch(failedMarkup, /999 tokens/)
const pending = fixtureRun({ state: 'running', progress: { completedRequests: null, totalRequests: 24, fixture: null, temperature: null, repeat: null, phase: null }, cells: [] })
const pendingMarkup = renderToStaticMarkup(React.createElement(ResultsView, { run: pending }))
assert.match(pendingMarkup, /indeterminate/); assert.doesNotMatch(pendingMarkup, /<progress[^>]+value=/, 'unknown request count never becomes a fabricated zero/percent')
const completeMarkup = renderToStaticMarkup(React.createElement(ResultsView, { run: fixtureRun() }))
assert.match(completeMarkup, /25 tokens\/s/); assert.match(completeMarkup, /0.2 s/); assert.match(completeMarkup, /Engine decode speed/); assert.match(completeMarkup, /Not reported/)
const MetadataView = load('BenchmarkMetadata', { 'react/jsx-runtime': jsx, '../../../lib/engine-format': { formatBytes } })
assert.match(renderToStaticMarkup(React.createElement(MetadataView, { run: fixtureRun() })), /not available/)
const recorded = fixtureRun({ metadata: { runtime: { tensorfold_version: '1.0.12', backend: 'mlx', python_version: '3.11', platform: 'macos', dependencies: { mlx: 'fixture-version' } }, model: { repo_id: 'fixture/recorded-model', revision: 'a'.repeat(40), family: 'fixture', config_sha256: 'b'.repeat(64), tokenizer_sha256: null, quantization: '4-bit', drafter_repo_id: null, local: false }, hardware: { cpu_model: 'Synthetic recorded CPU', system_memory_bytes: 8 * 1024 ** 3, memory_type: 'unified', gpus: [{ name: 'Synthetic recorded GPU', memory_bytes: null, cores: null }], gpu_count: 1, source: 'partial' }, settings: { tokens: 256, repetitions: 5, temperatures: [1, 0], serial: false, context_tokens: 4096, managed: true, rank_count: 1 } } })
const recordedMarkup = renderToStaticMarkup(React.createElement(MetadataView, { run: recorded }))
assert.match(recordedMarkup, /Synthetic recorded CPU/); assert.match(recordedMarkup, /Synthetic recorded GPU/); assert.match(recordedMarkup, /TensorFold 1.0.12/); assert.match(recordedMarkup, /belong to this recorded run/); assert.doesNotMatch(recordedMarkup, /peak|power|watts/i)
assert.match(display.benchmarkBasicSummary(recorded), /community-v1/, 'old snapshots retain their original protocol label')
const v2 = fixtureRun({ metadata: { ...recorded.metadata!, suite: { id: 'community-v2', sha256: 'fixture' } } })
assert.match(display.benchmarkBasicSummary(v2), /community-v2/)
const v2Markup = renderToStaticMarkup(React.createElement(ResultsView, { run: v2 }))
assert.match(v2Markup, /All community-v2 cells/); assert.doesNotMatch(v2Markup, /community-v1/)
console.log('Benchmark drawer actual handlers: cached defaults, blocked/duplicate Run, serial selection, close/cancel, history, guarded export/copy/community link and truthful incomplete/missing metrics passed')
