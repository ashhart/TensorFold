import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import path from 'node:path'
import { createRequire } from 'node:module'
const app = path.resolve(import.meta.dirname, '../../..')
const require = createRequire(path.join(app, 'package.json'))
const React = require('react'), { renderToStaticMarkup } = require('react-dom/server'), ts = require('typescript'), clsx = require('clsx')
const { createJiti } = require('jiti');const jiti = createJiti(path.join(app, 'package.json'), { moduleCache: true, fsCache: false })
const { builtinChatLoadingPhase } = await jiti.import(path.join(app, 'src/lib/chat-loading-phase.ts'))
const { buildTurnRows, workRowLabel } = await jiti.import(path.join(app, 'src/lib/turn-blocks.ts'))
const source = await readFile(new URL('./TurnWorkRow.tsx', import.meta.url), 'utf8')
const ast = ts.createSourceFile('TurnWorkRow.tsx', source, ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const printer = ts.createPrinter()
const functions = ast.statements.filter(node => ts.isFunctionDeclaration(node) && ['useNow', 'TurnWorkRow'].includes(node.name?.text))
assert.equal(functions.length, 2)
const body = functions.map(node => printer.printNode(ts.EmitHint.Unspecified, node, ast).replace(/\bexport\s+(?=function\s+)/g, '')).join('\n')
const compiled = ts.transpileModule(body, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None, jsx: ts.JsxEmit.React } }).outputText
let locale = 'en'
const useStore = select => select({ settings: { uiPreferences: { locale } } })
const Row = new Function('React', 'useState', 'useEffect', 'useStore', 'workRowLabel', 'clsx', 'Icon', 'Spinner', 'WorkItems', `${compiled};return TurnWorkRow;`)(React, React.useState, React.useEffect, useStore, workRowLabel, clsx, () => null, () => null, () => null)
const provider = { id: 'tensorfold', source: 'tensorfold', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1/v1', models: ['fixture-model'], apiKey: '', name: 'TensorFold' }
const engine = { available: true, state: 'starting', model: 'fixture-model' }
const timeline = [{ id: 'user', kind: 'user', text: 'Synthetic first message', at: Date.now() - 1000 }]
const turn = buildTurnRows(timeline, { running: true }).find(row => row.type === 'work').turn
const before = JSON.stringify({ provider, engine, turn, timeline })
const render = (p = provider, e = engine, status = 'thinking', remoteExecution = false, model = 'fixture-model') => renderToStaticMarkup(React.createElement(Row, { turn, status, phaseLabel: builtinChatLoadingPhase({ provider: p, engine: e, model, status, remoteExecution }) }))
const loading = render();assert.match(loading, /Loading model/);assert.doesNotMatch(loading, /Thinking/);assert.match(loading, /role="status"/)
const ready = render(provider, { ...engine, state: 'ready' });assert.doesNotMatch(ready, /Loading model/);assert.match(ready, /Thinking/)
for (const p of [{ ...provider, id: 'other-local', source: 'omlx' }, { ...provider, transport: 'codex-cli' }, { ...provider, baseUrl: 'https://model.example.test/v1' }]) { const html = render(p);assert.doesNotMatch(html, /Loading model/);assert.match(html, /Thinking/) }
assert.doesNotMatch(render(provider, engine, 'thinking', true), /Loading model/)
assert.doesNotMatch(render(provider, engine, 'thinking', false, 'different-model'), /Loading model/)
assert.doesNotMatch(render(provider, { ...engine, available: false }), /Loading model/)
assert.doesNotMatch(render(provider, engine, 'awaiting'), /Loading model/)
locale = 'zh-Hans'
assert.doesNotMatch(render(provider, { ...engine, state: 'ready' }), /Thinking|Working/, 'The actual row reads the live profile language for its authored summary')
assert.equal(JSON.stringify({ provider, engine, turn, timeline }), before, 'display feedback never mutates run status, timeline timestamps or model metrics')
console.log('Actual TurnWorkRow loading→ready component rendering, matching built-in model guard and unchanged API/CLI/approval/remote states passed')
