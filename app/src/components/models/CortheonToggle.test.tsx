import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import * as policy from '../../../shared/release-feature-policy'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
// Use the real Button/Toggle renderer; only unused portal/icon collaborators are inert.
const uiImports: Record<string, unknown> = { react: React, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: () => { throw new Error('No portal in this static control qualification.') } }, clsx, '../lib/icons': { Icon: () => null } }
const ui = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, '..', 'ui.tsx'), 'utf8'), { compilerOptions }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, uiImports); if (reviewed.matched) return reviewed.value;  assert.ok(id in uiImports); return uiImports[id] }, ui, ui.exports)
const { Button, Toggle } = ui.exports
const calls: boolean[] = [], errors: unknown[] = []
const source = readFileSync(path.join(__dirname, 'CortheonToggle.tsx'), 'utf8')
const output = ts.transpileModule(source, { compilerOptions }).outputText
const imports: Record<string, unknown> = {
  react: { useState: (value: unknown) => [value, (next: unknown) => errors.push(next)] },
  'react/jsx-runtime': jsx,
  '../../lib/ipc': { api: { engineSetCortheon: async (enabled: boolean) => { calls.push(enabled) } } },
  '../../lib/markdown-link': { MarkdownLink: () => { throw new Error('The paused release must not offer a live research setup link.') } },
  '../ui': { Button, Toggle }, '../../../shared/release-feature-policy': policy,
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', output)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed component import ${id}`); return imports[id] }, loaded, loaded.exports)
const CortheonToggle = loaded.exports.CortheonToggle
function find(node: any, type: any): any {
  if (!node || typeof node !== 'object') return undefined
  if (node.type === type) return node
  for (const child of React.Children.toArray(node.props?.children)) { const found = find(child, type); if (found) return found }
}
for (const status of [undefined, { available: false }, { available: true, state: 'ready', cortheon: { supported: true, enabled: true, active: false } }, { available: true, state: 'starting', cortheon: { supported: false, enabled: false, active: false } }]) {
  const node = CortheonToggle({ status, activeRuns: 0 })
  const markup = renderToStaticMarkup(node)
  assert.match(markup, /temporarily disabled in this beta/)
  assert.match(markup, /role="switch"/)
  const toggle = find(node, Toggle)
  assert.equal(toggle.props.disabled, true)
  assert.equal(toggle.props.checked, false, 'stale enabled preferences do not look active')
  toggle.props.onChange(true)
  await Promise.resolve()
  assert.deepEqual(calls, [], 'even manually invoking the disabled enable handler cannot call the bridge')
  assert.ok(errors.includes(policy.CORTHEON_RELEASE_POLICY.reason))
}
const active = CortheonToggle({ status: { available: true, state: 'ready', cortheon: { supported: true, enabled: false, active: true } }, activeRuns: 0 })
assert.match(renderToStaticMarkup(active), /Research is active in an existing engine/)
assert.equal(find(active, Toggle).props.checked, true, 'observed legacy research remains visible')
const off = find(active, Button)
assert.equal(off.props.children, 'Turn off')
assert.equal(off.props.disabled, false)
off.props.onClick()
await Promise.resolve()
assert.deepEqual(calls, [false], 'the only admitted control disables an observed active instance')
const busy = CortheonToggle({ status: { available: true, state: 'ready', cortheon: { supported: true, enabled: false, active: true } }, activeRuns: 1 })
assert.equal(find(busy, Button).props.disabled, true, 'recovery does not bypass active-request protection')
console.log('Cortheon paused renderer: disabled accessible toggle, stale status, existing active recovery and no enable bridge calls')
