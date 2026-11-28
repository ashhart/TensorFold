import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import * as metadata from '../../lib/provider-model-metadata'
import * as accounts from '../../lib/provider-accounts'
import type { BotThreadConfig, Provider, ReasoningEffortSetting } from '../../lib/types'

const jsx = (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) })
const jsxRuntime = { jsx, jsxs: jsx, Fragment: React.Fragment }
const compile = (source: string) => ts.transpileModule(source, { compilerOptions: {
  target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX,
} }).outputText
function load(file: string, imports: Record<string, unknown>) {
  const module = { exports: {} as any }
  new Function('require', 'module', 'exports', compile(readFileSync(path.join(__dirname, file), 'utf8')))((id: string) => {
    const reviewed = reviewedLanguageImport(id, imports)
    if (reviewed.matched) return reviewed.value
    assert.ok(id in imports, `Reviewed renderer import ${id}`)
    return imports[id]
  }, module, module.exports)
  return module.exports
}
const context = load('BotComposerSettings.tsx', { react: React, 'react/jsx-runtime': jsxRuntime })
const picker = load('../composer/ModelPickerMenu.tsx', {
  react: React, 'react/jsx-runtime': jsxRuntime, clsx: { default: clsx },
  '../../lib/icons': { Icon: () => null }, '../ui': { Menu: ({ trigger }: any) => trigger(false) },
  '../../../shared/provider-trust': { isExternalProvider: () => false }, '../../lib/ipc': { api: { platform: 'darwin' } },
  '../../lib/provider-accounts': accounts, '../../lib/provider-usage': {}, '../../lib/provider-model-metadata': metadata,
  './UsagePill': {}, './RateLimitBar': {}, '../models/EngineStatus': { EngineDot: () => null },
  '../../lib/store': {}, '../../lib/learning': { useLearning: () => ({ snapshot: null }) },
  '../../lib/engine': { useEngine: () => ({ snapshot: null }) }, '../../lib/turn-stats-line': { statsModelLabel: () => 'Fixture model' },
  '../../lib/model-picker': {}, '../../lib/favorite-models': {}, '../bots/BotComposerSettings': context,
})

function productionElement(file: string, tag: string, wrapper?: string) {
  const source = ts.createSourceFile(file, readFileSync(path.join(__dirname, file), 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
  let found: ts.JsxSelfClosingElement | undefined
  const visit = (node: ts.Node) => {
    if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(source) === tag) found = node
    ts.forEachChild(node, visit)
  }
  visit(source)
  assert.ok(found, `Production ${tag} element exists`)
  const parent = found.parent
  const element = wrapper && ts.isJsxElement(parent) && parent.openingElement.tagName.getText(source) === wrapper ? parent : found
  const code = compile(`const element = ${element.getText(source)}; return element`)
  return (environment: Record<string, unknown>) => new Function('require', 'exports', ...Object.keys(environment), code)(
    (id: string) => { assert.equal(id, 'react/jsx-runtime'); return jsxRuntime }, {}, ...Object.values(environment),
  )
}
const renderThreadComposer = productionElement('../Thread.tsx', 'Composer', 'BotComposerSettings')
const provider: Provider = { id: 'fixture', name: 'Fixture', kind: 'local', baseUrl: 'http://127.0.0.1', apiKey: '', models: ['fixture'] }
function composer(bot: BotThreadConfig | undefined, effort: ReasoningEffortSetting, roomWorkerEffort?: ReasoningEffortSetting) {
  const thread = { id: 'captured-bot-fixture', bot }
  function Composer() {
    return React.createElement(picker.ModelPickerMenu, {
      provider, model: 'fixture', providerAccountId: null, localOnly: false,
      policyProject: null, pickerProviders: [provider], recentModels: [],
      usageSnapshot: null, usageSummary: null, rateLimitStatus: null, onPickModel() {},
      detail: metadata.effortDisplayLabel(provider, 'fixture', effort), disabled: !!bot,
    })
  }
  return renderThreadComposer({ thread, roomWorkerEffort, Composer, BotComposerSettings: context.BotComposerSettings })
}
const bot = (reasoningEffort?: ReasoningEffortSetting) => ({ reasoningEffort } as BotThreadConfig)
const markup = (captured: BotThreadConfig | undefined, global: ReasoningEffortSetting, room?: ReasoningEffortSetting) => renderToStaticMarkup(composer(captured, global, room))
assert.match(markup(bot('low'), 'high'), />Low<\/span>/, 'the real thread/provider/picker wiring displays captured Low, not global High')
assert.match(markup(bot('high'), 'low'), />High<\/span>/, 'captured High remains independent of global Low')
assert.match(markup(bot(), 'high'), />High<\/span>/, 'a legacy bot without captured effort displays its global fallback')
assert.match(markup(undefined, 'low'), />Low<\/span>/, 'an ordinary conversation keeps the existing effort detail')
assert.doesNotMatch(markup(bot('dynamic'), 'high'), />High<\/span>/, 'an unsupported captured value cannot be misreported as global High')
const captured = bot('low')
assert.match(markup(captured, 'medium'), />Low<\/span>/, 'changing the app setting cannot rewrite a captured chat badge')
const siblings = renderToStaticMarkup(React.createElement(React.Fragment, null, composer(captured, 'high'), composer(undefined, 'high')))
assert.equal((siblings.match(/>Low<\/span>/g) ?? []).length, 1)
assert.equal((siblings.match(/>High<\/span>/g) ?? []).length, 1, 'a captured value cannot leak into an adjacent ordinary composer')
assert.match(markup(undefined, 'high', 'low'), />Low<\/span>/, 'the real worker composer displays captured Low instead of app High')
assert.match(markup(undefined, 'low', 'high'), />High<\/span>/, 'captured Room effort stays independent from app Low')
assert.match(markup(undefined, 'high'), />High<\/span>/, 'a completed worker returns to the ordinary app fallback')
console.log('Bot composer checks passed: captured/legacy/global badges, actual parent wiring and sibling isolation.')
