import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import * as format from '../../lib/engine-format'
import * as display from './download-display'
import { repoDisplayName } from '../../lib/turn-stats-line'
import type { PullProgress } from '../../lib/types'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
function load(file: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }
  new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, file), 'utf8'), { compilerOptions }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed renderer import ${id}`); return imports[id] }, module, module.exports)
  return module.exports
}
const { Button } = load('../ui.tsx', { react: React, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: () => null }, clsx, '../lib/icons': { Icon: () => null } })
let slots: any[] = [], cursor = 0
const hooks = {
  useState(initial: any) { const slot = cursor++; if (!(slot in slots)) slots[slot] = typeof initial === 'function' ? initial() : initial; return [slots[slot], (value: any) => { slots[slot] = typeof value === 'function' ? value(slots[slot]) : value }] },
  useRef(initial: any) { const slot = cursor++; if (!(slot in slots)) slots[slot] = { current: initial }; return slots[slot] },
  useEffect() {},
}
const imports = { react: hooks, 'react/jsx-runtime': jsx, clsx, '../../lib/icons': { Icon: () => null }, '../ui': { Button }, '../../lib/engine': format, '../../lib/turn-stats-line': { repoDisplayName }, './download-display': display }
const { PullRow, HubRow } = load('ModelRows.tsx', imports)
const { DownloadCentre } = load('DownloadCentre.tsx', { react: hooks, 'react/jsx-runtime': jsx, '../../lib/types': {}, './ModelRows': { PullRow }, './download-display': display, '../ui': { Button } })
const base: PullProgress = { repo: 'Org/Example', revision: null, state: 'running', phase: 'Downloading Org/Example', receivedBytes: 1024 ** 3, totalBytes: 4 * 1024 ** 3, lastLine: null, message: null, startedAt: 0, finishedAt: null, updatedAt: Date.now(), bytesPerSecond: 1024 ** 2, etaSeconds: 120 }
const reset = () => { slots = []; cursor = 0 }
const renderRow = (pull: PullProgress, onCancel: () => unknown = () => {}, onRetry: () => unknown = () => {}) => { cursor = 0; return PullRow({ pull, onCancel, onRetry }) }
function button(node: any, label: string): any {
  if (!node || typeof node !== 'object') return null
  if (node.type === Button && node.props.children === label) return node
  for (const child of React.Children.toArray(node.props?.children)) { const found = button(child, label); if (found) return found }
  return null
}
reset()
let markup = renderToStaticMarkup(renderRow(base))
assert.match(markup, /about 2 min left/)
assert.match(markup, /aria-valuenow="25"/)
assert.match(markup, /aria-label="Cancel download Org\/Example"/)
reset()
const queued = { ...base, state: 'queued' as const, queuePosition: 2 }
markup = renderToStaticMarkup(renderRow(queued))
assert.match(markup, /Queued · 2 ahead/); assert.doesNotMatch(markup, /about|progressbar|Measuring/)
assert.ok(button(renderRow(queued), 'Remove'))
reset()
const verifying = { ...base, state: 'verifying' as const }
markup = renderToStaticMarkup(renderRow(verifying))
assert.match(markup, /Verifying checkpoint/); assert.equal(button(renderRow(verifying), 'Cancel'), null)
assert.equal(button(renderRow(verifying), 'Resume'), null)
reset()
assert.match(renderToStaticMarkup(renderRow({ ...base, state: 'failed', message: 'Synthetic network unavailable' })), /Synthetic network unavailable/)
let calls = 0, reject!: (error: Error) => void
const cancel = () => { calls++; return new Promise<void>((_resolve, no) => { reject = no }) }
const tree = renderRow(base, cancel)
button(tree, 'Cancel').props.onClick(); button(tree, 'Cancel').props.onClick()
assert.equal(calls, 1, 'two synchronous clicks admit one action before React rerenders')
assert.equal(button(renderRow(base, cancel), 'Cancel').props.disabled, true)
reject(new Error('Synthetic cancellation rejected'))
for (let i = 0; i < 10; i++) await Promise.resolve()
markup = renderToStaticMarkup(renderRow(base, cancel))
assert.match(markup, /role="alert"/); assert.match(markup, /Synthetic cancellation rejected/)
assert.equal(button(renderRow(base, cancel), 'Cancel').props.disabled, false)
reset()
assert.match(renderToStaticMarkup(renderRow({ ...base, state: 'done' })), /Ready to load/)
assert.equal(button(renderRow({ ...base, state: 'done' }), 'Resume'), null)
assert.equal(button(renderRow({ ...base, state: 'done' }), 'Load'), null, 'download completion cannot start a model')
reset()
assert.equal(DownloadCentre({ pulls: [], onCancel() {}, onRetry() {} }), null)
reset()
markup = renderToStaticMarkup(React.createElement(DownloadCentre, { pulls: [queued, base], onCancel() {}, onRetry() {} }))
assert.match(markup, /1 downloading · 1 queued/)
assert.match(markup, /One download at a time/)
for (const state of ['queued', 'running', 'verifying'] as const) {
  reset()
  const row = HubRow({ repo: base.repo, pull: { ...base, state }, preflight: { loading: false, result: { verdict: 'installed', reasons: [], fitsMemory: true, fitsDisk: true } }, onCheck() {}, onDownload() {} })
  const expected = state === 'queued' ? 'Queued' : state === 'verifying' ? 'Verifying' : 'Downloading'
  assert.equal(button(row, expected).props.disabled, true, 'hub cannot enqueue duplicate pending model downloads')
}
console.log('Actual download controls: queue/verification/progress, measured ETA, duplicate prevention, handled errors, admission and no automatic model loading passed')
