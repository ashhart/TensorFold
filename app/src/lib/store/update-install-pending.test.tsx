import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import { registerUpdatePreparationBlocker, rendererUpdateBlockReason } from '../update-safety'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
function load(file: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }
  const output = ts.transpileModule(readFileSync(path.join(process.cwd(), file), 'utf8'), { compilerOptions }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports, `Reviewed production import: ${id}`); return imports[id] }, module, module.exports)
  return module.exports
}
const ready = { state: 'ready', latestVersion: '1.0.0-beta.6', currentVersion: '1.0.0-beta.5', signatureVerified: true, integrity: 'artifact_verified', automaticInstall: { supported: true, reason: null }, checkedAt: null, releaseNotes: 'Synthetic release notes.', error: null }
let state: any, calls = 0, resolve!: (result: { ok: boolean }) => void, reject!: (reason: Error) => void
const api = { installAppUpdate() {
  calls++
  assert.equal(state.updateModalOpen, true, 'The real slice opens the modal before entering the async native API')
  assert.equal(state.updateInstallPending, true, 'The real slice marks pending before entering the async native API')
  return new Promise<{ ok: boolean }>((yes, no) => { resolve = yes; reject = no })
} }
const storage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: () => null, setItem() {} } })
const { updatesSlice } = load('src/lib/store/slices/updatesSlice.ts', { '../../ipc': { api } })
function reset() {
  state = { ...updatesSlice((patch: any) => { state = { ...state, ...(typeof patch === 'function' ? patch(state) : patch) } }, () => state), updateStatus: { ...ready } }
  calls = 0
}
reset()
assert.equal(state.updateInstallPending, false); assert.equal(state.updateModalOpen, false)
const first = state.installAppUpdate(), second = state.installAppUpdate()
assert.equal(calls, 1, 'Two synchronous explicit install calls share one native request')
assert.equal(state.updateStatus.state, 'ready', 'The regression exercises async native preflight before an installing push')
state.closeUpdateModal(); assert.equal(state.updateModalOpen, true, 'Even an old Close callback cannot dismiss preflight')
await second
resolve({ ok: false }); await first
assert.equal(state.updateInstallPending, false); assert.equal(state.updateStatus.state, 'ready'); assert.equal(state.updateModalOpen, true)
state.closeUpdateModal(); assert.equal(state.updateModalOpen, false, 'A resolved refusal releases UI dismissal without dropping the ready artifact')
reset()
const failed = state.installAppUpdate(); reject(new Error('Synthetic native preflight rejected.'))
await assert.rejects(failed, /preflight rejected/)
assert.equal(state.updateInstallPending, false); assert.equal(state.updateStatus.state, 'ready')
state.closeUpdateModal(); assert.equal(state.updateModalOpen, false)
reset()
const accepted = state.installAppUpdate(); state.applyUpdateStatus({ ...ready, state: 'installing' }); resolve({ ok: true }); await accepted
assert.equal(state.updateInstallPending, false); state.closeUpdateModal(); assert.equal(state.updateModalOpen, true, 'Host installing keeps the modal protected after the API promise settles')
await state.installAppUpdate(); assert.equal(calls, 1, 'A settled installing handoff cannot initiate another native request')

const { Button } = load('src/components/ui.tsx', { react: React, 'react/jsx-runtime': jsx, 'react-dom': { createPortal() { throw new Error('No portals in these production update controls.') } }, clsx, '../lib/icons': { Icon: () => null } })
const useStore = Object.assign((select: any) => select(state), { getState: () => state })
let draftError: string | null = null
const { UpdateActions } = load('src/components/UpdateActions.tsx', {
  react: { useState: () => [draftError, (value: string | null) => { draftError = value }] }, 'react/jsx-runtime': jsx,
  '../lib/store': { useStore }, '../lib/update-safety': { rendererUpdateBlockReason }, './ui': { Button },
})
function find(node: any, predicate: (node: any) => boolean): any {
  if (!node || typeof node !== 'object') return undefined
  if (predicate(node)) return node
  for (const child of React.Children.toArray(node.props?.children)) { const found = find(child, predicate); if (found) return found }
}
const action = (label: string) => find(UpdateActions({ status: state.updateStatus }), (n) => n.type === Button && n.props.children === label)
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve() }
reset()
const removeBlocker = registerUpdatePreparationBlocker(() => 'Save the synthetic editor first.')
action('Install and restart').props.onClick(); await settle()
assert.equal(calls, 0); assert.match(draftError!, /synthetic editor/); assert.equal(state.updateModalOpen, false)
removeBlocker(); draftError = null
const installButton = action('Install and restart'); installButton.props.onClick(); installButton.props.onClick()
assert.equal(calls, 1); assert.equal(action('Saving and preparing restart…').props.disabled, true)
assert.equal(action('Install and restart'), undefined, 'The existing modal exposes no second install control during pending preflight')
reject(new Error('Synthetic host refusal.')); await settle()
assert.equal(state.updateInstallPending, false); assert.match(renderToStaticMarkup(UpdateActions({ status: state.updateStatus })), /Synthetic host refusal/)
assert.ok(action('Install and restart'), 'A rejected preflight is handled and leaves a retry control')

const listeners: { type: string; fn: any; capture: boolean }[] = []
const cleanups: (() => void)[] = [], layout: (() => (() => void) | void)[] = [], effects: (() => (() => void) | void)[] = []
let focused = 0
const element = { focus() { focused++ }, contains(target: unknown) { return target === element } }
const oldWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
Object.defineProperty(globalThis, 'window', { configurable: true, value: {
  addEventListener(type: string, fn: any, capture = false) { listeners.push({ type, fn, capture }) },
  removeEventListener(type: string, fn: any, capture = false) { const index = listeners.findIndex((entry) => entry.type === type && entry.fn === fn && entry.capture === capture); if (index >= 0) listeners.splice(index, 1) },
} })
const { UpdateModal } = load('src/components/UpdateModal.tsx', {
  react: { useMemo: (build: any) => build(), useRef: () => ({ current: element }), useEffect: (effect: any) => effects.push(effect), useLayoutEffect: (effect: any) => layout.push(effect) },
  'react/jsx-runtime': jsx, '../lib/store': { useStore }, '../lib/icons': { Icon: () => null }, './ui': { Button },
  './UpdateActions': { UpdateActions }, '../lib/update-status': { updateEmptyMessage: () => 'Synthetic empty notes.' },
})
reset(); const pending = state.installAppUpdate()
const tree = UpdateModal()
for (const effect of [...layout, ...effects]) { const cleanup = effect(); if (cleanup) cleanups.push(cleanup) }
assert.equal(focused, 1, 'Pending modal synchronously takes keyboard focus before painting')
const dialog = find(tree, (n) => n.props.role === 'dialog')
assert.equal(dialog.props['aria-busy'], true); assert.equal(dialog.props.tabIndex, -1)
assert.equal(find(tree, (n) => n.props['aria-label'] === 'Close').props.disabled, true)
assert.equal(find(tree, (n) => n.type === Button && n.props.children === 'Close').props.disabled, true)
tree.props.onClick(); assert.equal(state.updateModalOpen, true, 'Pending backdrop cannot dismiss the modal')
const captureKey = listeners.find((entry) => entry.type === 'keydown' && entry.capture)!.fn
for (const key of ['Escape', 'Tab', 'n']) {
  let stopped = 0, prevented = 0
  captureKey({ key, metaKey: key === 'n', stopPropagation() { stopped++ }, preventDefault() { prevented++ } })
  assert.equal(stopped, 1, 'Background shortcuts cannot receive pending-modal keyboard events')
  assert.equal(prevented, key === 'n' ? 0 : 1)
}
listeners.find((entry) => entry.type === 'focusin' && entry.capture)!.fn({ target: {} })
assert.equal(focused, 4, 'Tab/Escape/background focus remain on the pending dialog')
resolve({ ok: false }); await pending
let staleStops = 0
captureKey({ key: 'n', stopPropagation() { staleStops++ } })
assert.equal(staleStops, 0, 'The pending keyboard barrier reads current state even before React effect cleanup')
for (const cleanup of cleanups) cleanup()
assert.equal(listeners.length, 0, 'Focus and key barriers are removed after pending state ends')
const ordinary = UpdateModal(); ordinary.props.onClick(); assert.equal(state.updateModalOpen, false, 'Refusal restores ordinary modal dismissal')
if (oldWindow) Object.defineProperty(globalThis, 'window', oldWindow); else Reflect.deleteProperty(globalThis, 'window')
if (storage) Object.defineProperty(globalThis, 'localStorage', storage); else Reflect.deleteProperty(globalThis, 'localStorage')
console.log('Actual update slice/API + modal/actions: synchronous protected preflight, one duplicate request, refusal/rejection retry, host installing continuity, renderer draft blocker, disabled dismissal and pending keyboard/focus isolation pass.')
