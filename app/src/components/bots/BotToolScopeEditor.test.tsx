import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import * as policy from '../../../shared/bot-tool-policy'

const preset = { id: 'bot_damaged_fixture', name: 'Synthetic retained bot', role: 'Review synthetic material', instructions: 'Retained original instructions.', executionTarget: 'workspace', approvalScope: 'readonly' }
const Button = (props: any) => React.createElement('button', props, props.children)
function loadComponent(name: string): any {
  const source = readFileSync(path.join(process.cwd(), `src/components/bots/${name}.tsx`), 'utf8')
  const output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText
  const store: any = (selector: any) => selector({ settings: { agentPresets: [preset] } })
  store.getState = () => ({ setView() {} })
  const imports: Record<string, any> = {
    react: { useState: (value: any) => [value, () => {}], useEffect() {} },
    'react/jsx-runtime': { jsx: React.createElement, jsxs: React.createElement, Fragment: React.Fragment },
    '../../lib/store': { useStore: store }, '../../lib/icons': { Icon: () => null },
    '../../lib/ipc': { api: { getMcpSnapshot: async () => { throw new Error('Damaged selections must not read MCP metadata.') } } },
    '../ui': { Button, Input: (props: any) => React.createElement('input', props) },
    '../../../shared/bot-tool-policy': policy,
  }
  // Automatic JSX passes children on props; retain that production tree while
  // replacing only state/effect and external-service collaborators.
  imports['react/jsx-runtime'].jsx = imports['react/jsx-runtime'].jsxs = (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) })
  const module = { exports: {} as any }
  new Function('require', 'module', 'exports', output)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed component import ${id}`); return imports[id] }, module, module.exports)
  return module.exports[name]
}
const Editor = loadComponent('BotToolScopeEditor'), Chat = loadComponent('BotChatContext')
function findButton(node: any, label: string): any {
  if (!node || typeof node !== 'object') return undefined
  if (node.type === Button && node.props.children === label) return node
  for (const child of React.Children.toArray(node.props?.children)) { const found = findButton(child, label); if (found) return found }
}
const damaged = [null, { mode: 'unknown', names: [] }, { mode: 'selected', names: 'read_file' }, { mode: 'selected', names: [4] }, { mode: 'selected', names: Array(201).fill('read_file') }]
for (const toolScope of damaged) for (const quarantined of [false, true]) {
  const bot: any = { ...preset, toolScope: quarantined ? { mode: 'selected', names: [] } : toolScope, ...(quarantined ? { toolScopeProblem: { value: toolScope } } : {}) }
  let repaired: any
  const element = Editor({ bot, onChange: (value: any) => { repaired = value } })
  const markup = renderToStaticMarkup(element)
  assert.match(markup, /Tool access needs review/)
  assert.match(markup, /Original saved selection/)
  assert.match(markup, /Discard keeps the original selection/)
  const reset = findButton(element, 'Reset selection to no tools')
  assert.ok(reset); reset.props.onClick()
  assert.deepEqual(repaired.toolScope, { mode: 'selected', names: [] })
  assert.equal(repaired.toolScopeProblem, undefined)
  assert.equal(repaired.name, preset.name); assert.equal(repaired.instructions, preset.instructions)
  for (const timeline of [[], [{ kind: 'user', text: 'Synthetic retained history.' }]]) {
    const thread: any = { timeline, bot: { ...bot, presetId: bot.id } }
    const chat = renderToStaticMarkup(Chat({ thread }))
    assert.match(chat, /selection needs review/)
    assert.match(chat, /Your conversation is retained/)
    assert.match(chat, /Original captured selection/)
  }
}
for (const value of [null, {}, { value: null }]) {
  const bot = { ...preset, toolScope: { mode: 'selected', names: [] }, toolScopeProblem: value }
  assert.match(renderToStaticMarkup(Editor({ bot, onChange() {} })), /needs review/)
  assert.match(renderToStaticMarkup(Chat({ thread: { timeline: [], bot: { ...bot, presetId: bot.id } } })), /needs review/)
}
const normal = { ...preset, toolScope: { mode: 'selected', names: ['read_file'] } }
assert.match(renderToStaticMarkup(Chat({ thread: { timeline: [], bot: { ...normal, presetId: normal.id } } })), /Captured tool access: read_file/)
const vm = renderToStaticMarkup(Editor({ bot: { ...normal, executionTarget: 'vm', toolScope: { mode: 'selected', names: ['ask_user'] } }, onChange() {} }))
assert.match(vm, /Ask a structured question/)
assert.doesNotMatch(vm, /Unavailable in this VM/)
console.log('Actual bot components render damaged raw and quarantined selections without widening access or crashing; explicit reset retains instructions/history, and guest structured questions remain selectable.')
