import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { renderToStaticMarkup } from 'react-dom/server'
import * as React from 'react'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
import { flowTextFor } from '../../lib/preferences/flow-catalogue'
const jsx = { jsx: (type: any, props: any, key?: any) => React.createElement(type, { ...props, key }), jsxs: (type: any, props: any, key?: any) => React.createElement(type, { ...props, key }), Fragment: React.Fragment }
const imports: Record<string, any> = { '../../lib/preferences/useFlowText': { useFlowText: () => flowTextFor('en') }, 'react/jsx-runtime': jsx, '../ui': { Button: ({ variant: _variant, size: _size, ...props }: any) => React.createElement('button', props, props.children) } }
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'RunRecoveryCard.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports, name); return imports[name] }, loaded, loaded.exports)
const RunRecoveryCard = loaded.exports.RunRecoveryCard
import type { RunRecoverySnapshot } from '../../../shared/run-recovery'

const base: RunRecoverySnapshot = { runId: 'synthetic', state: 'stopping', reason: 'user_stop', message: 'Stop requested; waiting for started work.', completedTools: 2, completedMutations: 1, pendingOperations: 1, pendingMutations: 1, delivery: 'complete', retry: 'unavailable', retryReason: '', canStartNewRun: false, updatedAt: 1 }
const stopping = renderToStaticMarkup(React.createElement(RunRecoveryCard, { recovery: base, onEditRequest() {}, onRetry() {} }))
assert.match(stopping, /Stopping/)
assert.match(stopping, /1 operation still active/)
assert.match(stopping, /may still change files/)
assert.match(stopping, /disabled=""/)
assert.doesNotMatch(stopping, /Retry refused request/)
const unknown = renderToStaticMarkup(React.createElement(RunRecoveryCard, { recovery: { ...base, state: 'needs_review', pendingOperations: 0, pendingMutations: 0, delivery: 'unknown', retry: 'requires_review', retryReason: 'Request acceptance is unknown.', canStartNewRun: true }, onRetry() {}, onReviewEffects() {} }))
assert.match(unknown, /Review before continuing/)
assert.match(unknown, /Request acceptance is unknown/)
assert.match(unknown, /Review outcome/)
assert.doesNotMatch(unknown, /Retry refused request/)
const refused = renderToStaticMarkup(React.createElement(RunRecoveryCard, { recovery: { ...base, state: 'failed', pendingOperations: 0, pendingMutations: 0, delivery: 'refused', retry: 'safe', canStartNewRun: true }, onRetry() {} }))
assert.match(refused, /Retry refused request/)
assert.equal(renderToStaticMarkup(React.createElement(RunRecoveryCard, { recovery: { ...base, state: 'completed' } })), '')
console.log('PASS actual recovery card labels stopping work, blocks early actions and exposes retry only for safe refusal')
