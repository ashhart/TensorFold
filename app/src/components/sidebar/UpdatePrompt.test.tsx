import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import { registerUpdatePreparationBlocker, rendererUpdateBlockReason } from '../../lib/update-safety'
import type { AppUpdateStatus } from '../../lib/types'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
function load(file: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }
  const output = ts.transpileModule(readFileSync(path.join(__dirname, file), 'utf8'), { compilerOptions }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed component import: ${id}`); return imports[id] }, module, module.exports)
  return module.exports
}
const { Button } = load('../ui.tsx', { react: React, 'react/jsx-runtime': jsx, 'react-dom': { createPortal() { throw new Error('No portals in sidebar control.') } }, clsx, '../lib/icons': { Icon: () => null } })
const base: AppUpdateStatus = {
  state: 'available', currentVersion: '1.0.0-beta.5', latestVersion: '1.0.0-beta.6', releaseName: 'Synthetic release',
  releaseNotes: 'Synthetic notes.', downloadUrl: 'https://example.com/installer.dmg', checksum: { algorithm: 'sha256', value: '0'.repeat(64) },
  signature: 'synthetic', signatureVerified: true, integrity: 'signature_verified', feedUrl: 'https://example.com/update.json', checkedAt: 1, error: null,
  automaticInstall: { supported: true, reason: null },
}
const calls = { download: 0, install: 0, cancel: 0, open: 0, details: 0 }
let status: AppUpdateStatus | null = { ...base }
let download: () => Promise<void> = async () => {}
let install: () => Promise<void> = async () => {}
let cancel: () => Promise<void> = async () => {}
const state = {
  get updateStatus() { return status },
  downloadAppUpdate: async () => { calls.download++; await download() },
  installAppUpdate: async () => { calls.install++; await install() },
  cancelAppUpdateDownload: async () => { calls.cancel++; await cancel() },
  openUpdateDownload: async () => { calls.open++ },
  openUpdateModal: () => { calls.details++ },
}
const store = Object.assign((selector: (s: typeof state) => unknown) => selector(state), { getState: () => state })
let slots: unknown[] = [], cursor = 0
const hooks = {
  useRef(initial: unknown) { const slot = cursor++; if (!(slot in slots)) slots[slot] = { current: initial }; return slots[slot] },
  useState(initial: unknown) { const slot = cursor++; if (!(slot in slots)) slots[slot] = initial; return [slots[slot], (next: unknown) => { slots[slot] = next }] },
}
const { UpdatePrompt, actionableSidebarUpdate, sidebarUpdateProgress } = load('UpdatePrompt.tsx', {
  react: hooks, 'react/jsx-runtime': jsx, '../../lib/store': { useStore: store }, '../../lib/update-safety': { rendererUpdateBlockReason }, '../ui': { Button },
})
const render = () => { cursor = 0; return UpdatePrompt() }
const markup = () => renderToStaticMarkup(render())
function button(node: any, label: string): any {
  if (!node || typeof node !== 'object') return undefined
  if ((node.type === 'button' || node.type === Button) && node.props.children === label) return node
  for (const child of React.Children.toArray(node.props?.children)) { const found = button(child, label); if (found) return found }
}
const click = (label: string, tree = render()) => { const found = button(tree, label); assert.ok(found, `Visible control: ${label}`); found.props.onClick() }
const settle = async () => { for (let i = 0; i < 12; i++) await Promise.resolve() }
function reset(patch: Partial<AppUpdateStatus> = {}) {
  slots = []; status = { ...base, ...patch }; download = install = cancel = async () => {}
  for (const name of Object.keys(calls) as (keyof typeof calls)[]) calls[name] = 0
}

for (const state of ['not_configured', 'not_checked', 'checking', 'current', 'error'] as const) assert.equal(actionableSidebarUpdate({ ...base, state }), null)
for (const patch of [{ signatureVerified: false }, { latestVersion: null }, { latestVersion: ' ' }, { latestVersion: base.currentVersion }, { latestVersion: `v${base.currentVersion}` }, { currentVersion: '' }]) assert.equal(actionableSidebarUpdate({ ...base, ...patch }), null)
for (const state of ['ready', 'installing'] as const) assert.equal(actionableSidebarUpdate({ ...base, state }), null, 'Ready controls require verified artifact state')
assert.equal(actionableSidebarUpdate(null), null)
for (const [receivedBytes, totalBytes, expected] of [[50, 100, 50], [-1, 100, 0], [200, 100, 100], [1, 0, null], [1, -1, null], [NaN, 100, null], [1, Infinity, null], [1, null, null]] as const) assert.equal(sidebarUpdateProgress({ ...base, downloadProgress: { receivedBytes, totalBytes } }), expected)

reset()
assert.match(markup(), /New version available/); assert.match(markup(), /v1\.0\.0-beta\.6/)
assert.equal(button(render(), 'Update').props.variant, 'default', 'The compact action uses semantic ink/card colors for readable dark and light text')
assert.equal(calls.details, 0, 'Mount is quiet: no modal, checks, download or restart')
click('Details'); assert.equal(calls.details, 1)
let finishDownload!: () => void
download = () => new Promise<void>((resolve) => { finishDownload = resolve })
const available = render(); click('Update', available); click('Update', available)
assert.equal(calls.download, 1, 'Two synchronous clicks share one download admission')
assert.ok(button(render(), 'Starting…').props.disabled)
status = { ...base, state: 'downloading', downloadProgress: { receivedBytes: 25, totalBytes: 100 } }
assert.match(markup(), /25%/)
let finishCancel!: () => void
cancel = () => new Promise<void>((resolve) => { finishCancel = resolve })
const downloading = render(); click('Cancel', downloading); click('Cancel', downloading)
assert.equal(calls.cancel, 1, 'Cancellation remains available while the download promise is pending, with its own duplicate guard')
finishCancel(); finishDownload(); await settle()
status = { ...base, error: 'Download cancelled; your current app is unchanged.' }
assert.match(markup(), /Download cancelled/); assert.ok(button(render(), 'Update'))

reset({ state: 'ready', integrity: 'artifact_verified' })
assert.equal(button(render(), 'Restart to update').props.variant, 'default')
const removeBlocker = registerUpdatePreparationBlocker(() => 'Save your synthetic draft first.')
click('Restart to update'); await settle()
assert.equal(calls.install, 0); assert.match(markup(), /Save your synthetic draft first/)
assert.equal(status?.state, 'ready', 'A renderer blocker leaves the ready artifact intact')
removeBlocker()
let finishInstall!: () => void
install = () => new Promise<void>((resolve) => { finishInstall = resolve })
const ready = render(); click('Restart to update', ready); click('Restart to update', ready)
assert.equal(calls.install, 1); assert.ok(button(render(), 'Preparing…').props.disabled)
finishInstall(); await settle()
install = async () => { throw new Error('Synthetic workspace save failed.') }
click('Restart to update'); await settle()
assert.match(markup(), /Synthetic workspace save failed/); assert.ok(button(render(), 'Restart to update'))
assert.equal(status?.state, 'ready')

reset({ state: 'ready', integrity: 'artifact_verified' })
const stale = render()
status = { ...base, state: 'available' }
click('Restart to update', stale); await settle(); assert.equal(calls.install, 0)
reset()
const oldRelease = render()
status = { ...base, latestVersion: '1.0.0-beta.7' }
click('Update', oldRelease); await settle(); assert.equal(calls.download, 0, 'A stale rendered version cannot start another release')

reset({ automaticInstall: { supported: false, reason: 'Synthetic manual build.' } })
assert.match(markup(), /Downloads, verifies and opens the installer/)
click('Update'); await settle(); assert.equal(calls.download, 1); assert.equal(calls.open, 0, 'Available manual mode still uses the verified download pipeline')
status = { ...status!, state: 'ready', integrity: 'artifact_verified' }
assert.equal(button(render(), 'Open installer').props.variant, 'default')
click('Open installer'); await settle(); assert.equal(calls.open, 1); assert.equal(calls.install, 0)
reset({ state: 'installing', integrity: 'artifact_verified' })
assert.ok(button(render(), 'Restarting…').props.disabled)
assert.equal(calls.install, 0)
assert.doesNotMatch(readFileSync(path.join(__dirname, '../../App.tsx'), 'utf8'), /openUpdateModal/, 'App launch does not open release notes automatically')
const sidebar = readFileSync(path.join(__dirname, '../Sidebar.tsx'), 'utf8')
assert.ok(sidebar.indexOf('<UpdatePrompt />') < sidebar.indexOf('{engineReady && <EngineChip'), 'The compact prompt precedes the engine and footer controls')
console.log('Actual sidebar update controls: quiet admission, signed host states, progress bounds, duplicate download/cancel/restart prevention, live-state recheck, renderer restart blocker, ready recovery and verified manual fallback passed.')
