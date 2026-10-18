import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
const jsx = { jsx: (type, props) => React.createElement(type, props), jsxs: (type, props) => React.createElement(type, props) }
const imports = {
  'react/jsx-runtime': jsx,
  '../../lib/preferences/useAuthoredText': { useAuthoredText: () => (source, params = {}) => source.replace(/\{(\w+)\}/g, (token, key) => Object.hasOwn(params, key) ? String(params[key] ?? '') : token) },
  '../../lib/icons': { Icon: () => null },
}
const compiled = ts.transpileModule(readFileSync(new URL('./TurnOutcome.tsx', import.meta.url), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
const loaded = { exports: {} }
new Function('require', 'module', 'exports', compiled)(id => { assert.ok(Object.hasOwn(imports, id), `Reviewed outcome component import ${id}`); return imports[id] }, loaded, loaded.exports)
const { TurnOutcome } = loaded.exports
const turn = { running: false, work: [] }
assert.equal(renderToStaticMarkup(React.createElement(TurnOutcome, { turn, onShowActivity() {} })), '')
function output(verdict, extra = {}) {
  const receipt = { verdict, needsReview: !['verified', 'not_applicable'].includes(verdict), warnings: [], errors: 3, summary: 'One action was denied.', ...extra }
  return renderToStaticMarkup(React.createElement(TurnOutcome, { turn: { ...turn, work: [{ kind: 'notice', runReceipt: receipt }] }, onShowActivity() {} }))
}
assert.match(output('failed'), /Actions need review/)
assert.match(output('failed'), /Recorded errors: 3\. Verify the result before relying on the answer\./)
assert.match(output('blocked'), /One action was denied/)
assert.match(output('unverified'), /Changes need verification/)
assert.match(output('cancelled'), /Earlier actions may have completed/)
assert.match(output('not_applicable', { warnings: ['A repeat guard stopped an action.'] }), /Safety checks recorded a warning/)
assert.equal(output('verified'), '')
assert.equal(output('not_applicable'), '')
assert.match(output('failed', { warnings: undefined, errors: undefined, needsReview: false }), /The run recorded a failure/, 'older receipt fields never crash the conversation or hide the failed verdict')
assert.ok(!/automatically retry|nothing changed|rolled back/i.test(output('failed')), 'recorded failure must not imply replay or rollback')
console.log('recorded turn outcome: actual render passed')
