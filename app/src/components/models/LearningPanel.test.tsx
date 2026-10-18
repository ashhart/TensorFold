import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import * as graph from '../../lib/learning-graph'
import type { EngineLearningFact } from '../../lib/types'

const facts: EngineLearningFact[] = [
  { id: 1, text: 'Synthetic retained fact.', state: 'learned', recalled: true, source: 'chat', at: 1, links: [] },
  { id: 2, text: 'Synthetic pending deletion.', state: 'deleting', recalled: true, source: 'chat', at: 2, links: [] },
  { id: 3, text: 'Synthetic deleted fact.', state: 'deleted', recalled: true, deleted: 1234, source: 'chat', at: 3, links: [] },
]
let state: any[] = [null, '', 0], cursor = 0
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = ({ size: _size, variant: _variant, children, ...props }: any) => React.createElement('button', props, children)
const imports: Record<string, unknown> = {
  react: { useMemo: (fn: () => unknown) => fn(), useState: (initial: any) => { const index = cursor++; if (!(index in state)) state[index] = typeof initial === 'function' ? initial() : initial; return [state[index], (value: any) => { state[index] = typeof value === 'function' ? value(state[index]) : value }] } },
  'react/jsx-runtime': jsx, clsx,
  '../../lib/learning': { useLearning: () => ({ snapshot: { active: false, graph: { next: 4, facts } } }) },
  '../../lib/learning-graph': graph,
  '../ui': { Button, Input: (props: any) => React.createElement('input', props) },
  './LearningControls': { LearningControls: () => null },
}
const module = { exports: {} as any }
const source = readFileSync(path.join(__dirname, 'LearningPanel.tsx'), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText
new Function('require', 'module', 'exports', compiled)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed import ${id}`); return imports[id] }, module, module.exports)
function render() { cursor = 0; return module.exports.LearningPanel({ activeRuns: 0 }) }
function all(node: any, predicate: (entry: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }

let node = render()
let markup = renderToStaticMarkup(node)
assert.match(markup, />Deleting<\/text>/)
assert.match(markup, />Deleted<\/text>/)
assert.match(markup, /1 learned facts have a recorded passing recall check/, 'stale recalled=true on deleting/deleted rows must not claim they are learned')
const svg = all(node, (entry) => entry.type === 'svg' && entry.props.role === 'group')[0]
assert.equal(svg.props.viewBox.split(' ')[2], String(graph.buildLearningLayout(facts).width))
assert.equal(svg.props.style.minWidth, graph.buildLearningLayout(facts).width, 'all native state columns remain scrollable inside the graph')

const deleted = all(node, (entry) => entry.type === 'g' && entry.props['aria-label']?.startsWith('Fact 3, deleted'))[0]
deleted.props.onClick(); node = render(); markup = renderToStaticMarkup(node)
assert.match(markup, /Recall not applicable to a deleted lesson/)
assert.match(markup, /Deleted lesson; excluded from learned and recall totals/)
assert.match(markup, /Removed /)
assert.doesNotMatch(markup, /Recall passed/, 'the selected deleted lesson cannot present an old recall result as current')

const deleting = all(node, (entry) => entry.type === 'g' && entry.props['aria-label']?.startsWith('Fact 2, deleting'))[0]
let prevented = false
deleting.props.onKeyDown({ key: 'Enter', preventDefault: () => { prevented = true } }); node = render(); markup = renderToStaticMarkup(node)
assert.equal(prevented, true)
assert.match(markup, /Recall not current while deletion is in progress/)
assert.doesNotMatch(markup, /Recall passed/)
console.log('Actual Living Weights panel renders native deletion states, scrollable graph columns and non-current recall evidence without claiming deleted lessons remain learned')
