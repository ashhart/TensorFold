import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import * as metadata from '../../lib/provider-model-metadata'
import type { Provider } from '../../lib/types'

const source = readFileSync(new URL('./BotReasoningEffort.tsx', import.meta.url), 'utf8')
const output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
const jsx = (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) })
const imports: Record<string, unknown> = { 'react/jsx-runtime': { jsx, jsxs: jsx }, '../../lib/provider-model-metadata': metadata }
const module = { exports: {} as any }
new Function('require', 'module', 'exports', output)((id: string) => {
  const reviewed = reviewedLanguageImport(id, imports)
  if (reviewed.matched) return reviewed.value
  assert.ok(id in imports, `Reviewed component import ${id}`)
  return imports[id]
}, module, module.exports)
const { BotReasoningEffort, effortForBotModel } = module.exports
const gemini: Provider = { id: 'gemini', name: 'Gemini API', kind: 'cloud', baseUrl: 'https://model.example.test/v1', apiKey: '', models: ['gemini-2.5-pro', 'gemini-3-pro'] }
const choices = (provider?: Provider, model?: string) => {
  const tree = BotReasoningEffort({ provider, model, value: 'low', onChange() {} })
  return renderToStaticMarkup(tree)
}
assert.match(choices(), /Effort/)
assert.match(renderToStaticMarkup(BotReasoningEffort({ defaultEffort: 'high', onChange() {} })), /Default \(High\)/, 'the inherited option displays the actual app setting')
for (const value of ['minimal', 'low', 'medium', 'high']) assert.match(choices(), new RegExp(`value="${value}"`))
for (const value of ['minimal', 'low', 'medium', 'high']) assert.match(choices(gemini), new RegExp(`value="${value}"`), 'a bot with no selected model offers the default effort levels')
assert.match(choices(gemini, 'gemini-2.5-pro'), /value="dynamic"/)
assert.match(choices(gemini, 'gemini-2.5-pro'), /value="512"/)
assert.doesNotMatch(choices(gemini, 'gemini-3-pro'), /value="512"/)
assert.equal(effortForBotModel('512', gemini, 'gemini-3-pro'), 'high', 'a model change replaces an unsupported saved level with the model default')
assert.equal(effortForBotModel('low', gemini, 'gemini-3-pro'), 'low')
assert.equal(effortForBotModel(undefined, gemini, 'gemini-3-pro'), undefined, 'a legacy inherited choice remains inherited')
let selected: unknown = null
const tree = BotReasoningEffort({ value: 'high', onChange(value: unknown) { selected = value } })
const select = React.Children.toArray(tree.props.children).find((node: any) => node.type === 'select') as any
assert.ok(select)
select.props.onChange({ target: { value: 'low' } })
assert.equal(selected, 'low', 'changing the rendered control updates the stored bot parameter')
select.props.onChange({ target: { value: '' } })
assert.equal(selected, undefined)
console.log('Bot effort editor checks passed: fallback levels, selected-model choices, model changes and control events.')
