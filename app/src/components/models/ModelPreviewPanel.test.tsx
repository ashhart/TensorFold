import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import type { EngineEvent, EngineInstalledModel, EngineModelPreviewResult } from '../../lib/types'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = ({ children, ...props }: any) => React.createElement('button', props, children)
let states: any[] = [], refs: any[] = [], effects: any[] = [], queued: Array<() => void> = [], stateIndex = 0, refIndex = 0, effectIndex = 0
let tries: string[] = [], loads: string[] = [], cancels = 0
const pending = { resolveTry: null as ((value: EngineModelPreviewResult) => void) | null, resolveUse: null as (() => void) | null, rejectTry: null as ((failure: Error) => void) | null, event: null as ((event: EngineEvent) => void) | null }
let mode = 'pending', busyChanges: boolean[] = []
const imports: Record<string, any> = {
  react: {
    useState: (initial: any) => { const i = stateIndex++; if (!(i in states)) states[i] = typeof initial === 'function' ? initial() : initial; return [states[i], (next: any) => { states[i] = typeof next === 'function' ? next(states[i]) : next }] },
    useRef: (initial: any) => refs[refIndex++] ??= { current: initial },
    useEffect: (fn: any, deps: any[]) => { const i = effectIndex++; const old = effects[i]; if (!old || deps.some((value, n) => value !== old.deps[n])) queued.push(() => { old?.cleanup?.(); effects[i] = { deps, cleanup: fn() } }) },
  }, 'react/jsx-runtime': jsx, '../ui': { Button }, '../../lib/ipc': { api: {
    engineTryModel: async (repo: string) => { tries.push(repo); if (mode === 'reject') throw new Error('Synthetic engine refused preview'); return new Promise<EngineModelPreviewResult>((resolve, reject) => { pending.resolveTry = resolve; pending.rejectTry = reject }) },
    engineCancelTryModel: async () => { cancels++ },
    onEngineEvent: (callback: (event: EngineEvent) => void) => { pending.event = callback; return () => { if (pending.event === callback) pending.event = null } },
  } }, '../../lib/engine': { loadEngineModel: async (repo: string) => { loads.push(repo); await new Promise<void>((resolve) => { pending.resolveUse = resolve }) } },
}
const moduleObject = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'ModelPreviewPanel.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed import ${id}`); return imports[id] }, moduleObject, moduleObject.exports)
const base: EngineInstalledModel = { repo: 'example/model-a', title: 'Model A', family: 'qwen', modelType: 'qwen3_5', format: 'mlx', contextTokens: 8192, weightsBytes: 1, memoryFloorBytes: 1, fitsMemory: true, drafter: null, drafterInstalled: null }
const second = { ...base, repo: 'example/model-b', title: 'Model B' }
const result: EngineModelPreviewResult = { model: second.repo, startedAt: 1, finishedAt: 2, text: 'Synthetic local sample', decodeTokensPerSecond: 12.5, firstTokenMs: 250, loadMs: 100, completionTokens: 12, cachedTokens: 0, elapsedMs: 1000, prefillTokensPerSecond: null, promptTokens: null, peakMemoryBytes: null, profile: null }
const onBusyChange = (value: boolean) => { busyChanges.push(value) }
function render(extra: any = {}) { stateIndex = 0; refIndex = 0; effectIndex = 0; const tree = moduleObject.exports.ModelPreviewPanel({ models: [base, second], activeModel: base.repo, locked: false, fixedModel: second.repo, onBusyChange, ...extra }); for (const effect of queued.splice(0)) effect(); return tree }
function all(node: any, predicate: (entry: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function button(node: any, label: string) { const found = all(node, (entry) => entry.type === Button && entry.props.children === label)[0]; assert.ok(found, label); return found }
async function settle() { for (let i = 0; i < 8; i++) await Promise.resolve() }
function dispose() { for (const effect of effects) effect?.cleanup?.() }
function reset() { dispose(); states = []; refs = []; effects = []; queued = []; tries = []; loads = []; cancels = 0; pending.resolveTry = null; pending.resolveUse = null; pending.rejectTry = null; pending.event = null; mode = 'pending'; busyChanges = [] }

let node = render()
assert.equal(all(node, (entry) => entry.type === 'select').length, 0, 'Details preview cannot silently select another model')
button(node, 'Try model').props.onClick(); button(node, 'Try model').props.onClick(); await settle(); node = render()
assert.deepEqual(tries, [second.repo], 'Exact chosen model is measured once even when active model differs')
assert.equal(busyChanges.at(-1), true)
button(node, 'Cancel preview').props.onClick(); await settle(); assert.equal(cancels, 1)
pending.resolveTry?.(result); await settle(); node = render()
assert.match(renderToStaticMarkup(node), /12.5 tok\/s/); assert.match(renderToStaticMarkup(node), /0.25 s/); assert.match(renderToStaticMarkup(node), /Synthetic local sample/)
button(node, 'Use this model').props.onClick(); button(node, 'Use this model').props.onClick(); await settle(); assert.deepEqual(loads, [second.repo], 'Model selection is synchronously duplicate guarded')
pending.resolveUse?.(); await settle(); node = render(); assert.equal(busyChanges.at(-1), false)

reset(); node = render({ models: [base], fixedModel: second.repo }); assert.equal(button(node, 'Try model').props.disabled, true); button(node, 'Try model').props.onClick(); await settle(); assert.equal(tries.length, 0)
reset(); node = render({ models: [{ ...second, fitsMemory: false }] }); assert.equal(button(node, 'Try model').props.disabled, true); button(node, 'Try model').props.onClick(); await settle(); assert.equal(tries.length, 0); assert.match(renderToStaticMarkup(node), /exceeds the available memory budget/)
reset(); node = render({ locked: true, lockReason: 'Save or cancel your Settings edits before measuring this model.' }); button(node, 'Try model').props.onClick(); await settle(); assert.equal(tries.length, 0); assert.match(renderToStaticMarkup(node), /Save or cancel your Settings edits/)
reset(); mode = 'reject'; node = render(); button(node, 'Try model').props.onClick(); await settle(); node = render(); assert.match(renderToStaticMarkup(node), /Synthetic engine refused preview/); assert.equal(button(node, 'Try model').props.disabled, false)
reset(); node = render(); button(node, 'Try model').props.onClick(); await settle(); dispose(); assert.equal(cancels, 1, 'Only a preview owned by this panel is cancelled on unmount'); pending.resolveTry?.(result); await settle(); assert.equal(states[4], null, 'Unmounted preview does not install a late result')
reset(); node = render(); button(node, 'Try model').props.onClick(); await settle(); pending.resolveTry?.(result); await settle(); node = render(); assert.match(renderToStaticMarkup(node), /Synthetic local sample/)
pending.event?.({ kind: 'comparison-cleared' }); node = render(); assert.equal(states[4], null); assert.doesNotMatch(renderToStaticMarkup(node), /Synthetic local sample/); assert.match(renderToStaticMarkup(node), /Preview data was cleared with local memory/)
reset(); node = render(); button(node, 'Try model').props.onClick(); await settle(); const oldResolve = pending.resolveTry
pending.event?.({ kind: 'comparison-cleared' }); node = render(); button(node, 'Try model').props.onClick(); await settle(); const newResolve = pending.resolveTry
oldResolve?.(result); await settle(); node = render(); assert.equal(states[4], null, 'Pre-purge completion cannot restore cleared text'); assert.equal(states[1], true, 'Pre-purge finally cannot unlock a newer preview')
newResolve?.(result); await settle(); node = render(); assert.equal((states[4] as EngineModelPreviewResult | null)?.model, second.repo); assert.equal(states[1], false)
reset(); node = render(); button(node, 'Try model').props.onClick(); await settle(); const oldReject = pending.rejectTry
pending.event?.({ kind: 'comparison-cleared' }); oldReject?.(new Error('Synthetic stale sample error')); await settle(); node = render(); assert.doesNotMatch(renderToStaticMarkup(node), /Synthetic stale sample error/); assert.equal(states[4], null)
reset(); node = render({ fixedModel: undefined }); assert.equal(all(node, (entry) => entry.type === 'select').length, 1, 'Standalone selector remains available'); dispose(); assert.equal(cancels, 0)
console.log('Actual model preview handlers: fixed identity, measured metrics, rejection/retry, memory/dirty locks, duplicate preview/use, cancel/unmount, memory-purge clearing/late-result fencing and standalone selector passed')
