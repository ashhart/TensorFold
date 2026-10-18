import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import * as failures from '../../../shared/hub-failure'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }),
  jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const source = readFileSync(path.join(__dirname, 'ModelRecommendationPanel.tsx'), 'utf8')
const output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022,
  module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText
function MockButton(props: { children?: React.ReactNode; disabled?: boolean }) {
  return React.createElement('button', { disabled: props.disabled }, props.children)
}
const imports: Record<string, unknown> = {
  'react/jsx-runtime': jsx, '../../lib/engine': { formatBytes: () => '1 GiB' },
  '../ui': { Button: MockButton },
  '../../../shared/hub-failure': failures,
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', output)((id: string) => { assert.ok(id in imports, id); return imports[id] }, loaded, loaded.exports)
const Panel = loaded.exports.ModelRecommendationPanel
const recommendation = { status: 'checking', selected: null, fallback: null, reason: 'Checking models.', hardwareLabel: 'Apple silicon Mac' }
for (const kind of ['policy', 'transport', 'offline', 'http', 'check'] as const) {
  const html = renderToStaticMarkup(React.createElement(Panel, { recommendation, checkFailed: true, failureKind: kind,
    locked: false, starting: false, onStart: () => {}, onDownload: () => {}, onRetry: () => {} }))
  assert.ok(html.includes(failures.hubFailureCopy(kind).title))
  assert.ok(html.includes('Retry checks'))
  assert.ok(!html.includes('Download model'), 'a failed check cannot authorize a download')
  assert.doesNotMatch(html, /reconnect|credentials/i)
}
const policyUnavailable = renderToStaticMarkup(React.createElement(Panel, { recommendation: { ...recommendation, status: 'unavailable' },
  checkFailed: true, failureKind: 'policy', locked: false, starting: false, onStart: () => {}, onDownload: () => {}, onRetry: () => {} }))
assert.ok(policyUnavailable.includes(failures.hubFailureCopy('policy').title))
const installed = { repo: 'Org/Installed', title: 'Installed model', installed: true, action: 'load', memoryBytes: 1024 }
const verified = renderToStaticMarkup(React.createElement(Panel, { recommendation: { ...recommendation, status: 'ready', selected: installed },
  checkFailed: true, failureKind: 'policy', locked: false, starting: false, onStart: () => {}, onDownload: () => {}, onRetry: () => {} }))
assert.ok(verified.includes('Recommended: Installed model'))
assert.ok(!verified.includes(failures.hubFailureCopy('policy').title), 'verified installed models remain usable')
