import { fixtureText, reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import { authoredText } from '../../lib/preferences/language-packs'
import * as providerAccounts from '../../lib/provider-accounts'
import * as turnStats from '../../lib/turn-stats-line'
let uiLocale = 'en'
import * as modelPicker from '../../lib/model-picker'
import * as favorites from '../../lib/favorite-models'
import { isExternalProvider } from '../../../shared/provider-trust'
import type { Provider, EngineSnapshot } from '../../lib/types'

const local = { id: 'tensorfold', name: 'TensorFold', kind: 'local', source: 'tensorfold', models: ['small', 'large'], baseUrl: '' } as Provider
const cloud = { id: 'cloud-fixture', name: 'External fixture', kind: 'cloud', models: ['default'], baseUrl: 'https://models.example.test/v1', accounts: [{ id: 'work', name: 'Work', models: ['work-model'] }] } as Provider
const engine = { status: { available: true, state: 'ready', model: 'small', contextTokens: 8192, memoryLimitBytes: 8 * 1024 ** 3 }, installed: [
  { repo: 'small', title: 'Small local', memoryFloorBytes: 3 * 1024 ** 3, fitsMemory: true },
  { repo: 'large', title: 'Large local', memoryFloorBytes: 12 * 1024 ** 3, fitsMemory: false },
], suggestions: [] } as unknown as EngineSnapshot
const data = new Map<string, string>()
const preference = new favorites.FavoriteModelsPreferences('synthetic-picker', () => ({ getItem: (key) => data.get(key) ?? null, setItem: (key, value) => data.set(key, value) }))
preference.toggle({ providerId: local.id, accountId: null, model: 'small' })
preference.toggle({ providerId: cloud.id, accountId: 'deleted-account', model: 'gone' })
preference.toggle({ providerId: cloud.id, accountId: 'work', model: 'work-model' })
let cursor = 0, slots: any[] = []
let preferenceOwner: string | null = null
let picked: any[] = [], closes = 0, preferenceError: string | null = null
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const imports: Record<string, any> = {
  '../../lib/preferences/useAuthoredText': { useAuthoredText: () => uiLocale === 'en' ? fixtureText : (source: string, params = {}) => authoredText(uiLocale, source, params) },
  react: { useEffect() {}, useId: () => 'picker-fixture', useState(initial: any) { const index = cursor++; if (!(index in slots)) slots[index] = typeof initial === 'function' ? initial() : initial; return [slots[index], (value: any) => { slots[index] = typeof value === 'function' ? value(slots[index]) : value }] }, useRef(initial: any) { const index = cursor++; if (!(index in slots)) slots[index] = { current: initial }; return slots[index] } },
  'react/jsx-runtime': jsx, clsx: { default: clsx }, '../../lib/icons': { Icon: () => null }, '../../../shared/provider-trust': { isExternalProvider }, '../ui': { Menu: 'menu-fixture' },
  '../../lib/provider-accounts': providerAccounts, '../../lib/provider-usage': { modelUsageLabel: () => null }, '../../lib/provider-model-metadata': { providerIconName: () => 'cpu' },
  './UsagePill': { UsagePill: () => null }, './RateLimitBar': { RateLimitBar: () => null }, '../models/EngineStatus': { EngineDot: () => null },
  '../../lib/store': { useStore: (selector: any) => selector({ setView() {} }) }, '../../lib/learning': { useLearning: () => ({ snapshot: null }), learningBusy: () => false },
  '../../lib/engine': { useEngine: () => ({ snapshot: engine }) }, '../../lib/ipc': { api: { platform: 'darwin' } }, '../../lib/turn-stats-line': turnStats, '../../lib/model-picker': modelPicker,
  '../../lib/favorite-models': { ...favorites, useFavoriteModels(owner: string | null) { preferenceOwner = owner; const current = preference.get(); return { favorites: current, ready: true, error: preferenceError,
    isFavorite: (route: favorites.FavoriteModel) => current.some((item) => favorites.favoriteModelKey(item) === favorites.favoriteModelKey(route)),
    toggle: (route: favorites.FavoriteModel) => preference.toggle(route) } } },
}
const module = { exports: {} as any }
const compiled = ts.transpileModule(readFileSync(path.join(__dirname, 'ModelPickerMenu.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
new Function('require', 'module', 'exports', `${compiled}\nexports.body = ModelPickerBody;`)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, id); return imports[id] }, module, module.exports)
function descendants(node: any, match: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(match(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => descendants(child, match))] }
function render(overrides: any = {}) { cursor = 0; return module.exports.body({ close: () => closes++, provider: local, model: 'small', providerAccountId: null, ownerId: 'owned-host', localOnly: false,
  policyProject: null, pickerProviders: [local, cloud], recentModels: [], usageSnapshot: null, usageSummary: null, rateLimitStatus: null,
  onPickModel: (...args: any[]) => { picked.push(args) }, ...overrides }) }
function labelled(node: any, label: string) { const button = descendants(node, (item) => item.type === 'button' && item.props['aria-label'] === label)[0]; assert.ok(button, label); return button }
let menu = render()
assert.equal(preferenceOwner, 'owned-host')
let markup = renderToStaticMarkup(menu)
assert.match(markup, /Favourites/); assert.match(markup, /Small local/); assert.match(markup, /work-model/)
assert.doesNotMatch(markup, /deleted-account|gone/)
assert.equal(descendants(menu, (item) => item.type === 'button' && item.props['data-model-choice'] === modelPicker.modelChoiceKey(local.id, 'small', null)).length, 1, 'favourites are not duplicated in the provider section')
labelled(menu, 'Remove Small local from favourites').props.onClick()
assert.deepEqual(picked, []); assert.equal(closes, 0, 'starring is independent of loading/selecting a model')
menu = render(); assert.equal(labelled(menu, 'Add Small local to favourites').props['aria-pressed'], false)
labelled(menu, 'Add Large local to favourites').props.onClick()
menu = render()
const large = descendants(menu, (item) => item.type === 'button' && item.props['data-model-choice'] === modelPicker.modelChoiceKey(local.id, 'large', null))[0]
assert.equal(large.props.disabled, true, 'a favourite cannot override engine compatibility restrictions')
assert.equal(labelled(menu, 'Remove Large local from favourites').props.disabled, false, 'an unavailable model can still be removed from favourites')
menu = render({ localOnly: true, pickerProviders: [local] })
markup = renderToStaticMarkup(menu)
assert.doesNotMatch(markup, /work-model|External fixture|gone/, 'saved cloud favourites cannot bypass local-only provider filtering')
assert.match(markup, /Large local/)
const search = descendants(menu, (item) => item.type === 'input' && item.props['aria-label'] === 'Search models')[0]
search.props.onChange({ target: { value: 'Small local' } })
menu = render({ localOnly: true, pickerProviders: [local] })
markup = renderToStaticMarkup(menu)
assert.match(markup, /Small local/); assert.doesNotMatch(markup, /Large local/)
assert.deepEqual(picked, [])
const small = descendants(menu, (item) => item.type === 'button' && item.props['data-model-choice'] === modelPicker.modelChoiceKey(local.id, 'small', null))[0]
await small.props.onClick(); for (let index = 0; index < 10; index++) await Promise.resolve()
assert.equal(picked.length, 1)
assert.equal(picked[0][0], local); assert.deepEqual((picked as any[][])[0].slice(1), ['small', null]); assert.equal(closes, 1)
preferenceError = 'Synthetic favourites storage failure'
menu = render(); assert.match(renderToStaticMarkup(menu), /Synthetic favourites storage failure/)
assert.equal(descendants(menu, (item) => item.props.role === 'alert').length, 1)
console.log('Actual model picker stars never select/load models, honour exact accounts and local policy, keep compatibility gates, avoid duplicate rows and show preference failures.')

uiLocale = 'zh-Hans'
const outer = (model: string | null) => { cursor = 0; slots = []; return module.exports.ModelPickerMenu({ provider: cloud, model, providerAccountId: null, localOnly: false, policyProject: null, pickerProviders: [cloud], recentModels: [], usageSnapshot: null, usageSummary: null, rateLimitStatus: null, onPickModel() { throw new Error('Language display cannot select a model') } }).props.trigger(false) }
const blankLabel = renderToStaticMarkup(outer(null))
assert.ok(blankLabel.includes(authoredText(uiLocale, 'Choose a model')), 'An unconfigured model selector needs a localized action, not the raw English fallback')
assert.doesNotMatch(blankLabel, /Select model/)
for (const model of ['Settings', 'Select model']) assert.ok(renderToStaticMarkup(outer(model)).includes(model), 'A real model whose name resembles UI text stays original')
assert.equal(picked.length, 1, 'Rendering translated model labels cannot start model work')
console.log('Actual model trigger localizes an empty selection while preserving real model names and avoiding selection or loading.')

uiLocale = 'en'; slots = []; preferenceError = null
const planningProps = { provider: cloud, model: 'work-model', providerAccountId: 'work', initialScope: 'local' }
menu = render(planningProps)
const choiceButton = (tree: any, providerId: string, model: string, accountId: string | null = null) => descendants(tree, item => item.type === 'button' && item.props['data-model-choice'] === modelPicker.modelChoiceKey(providerId, model, accountId))[0]
assert.ok(choiceButton(menu, local.id, 'small'), 'The planning shortcut offers a usable local replacement')
assert.equal(choiceButton(menu, local.id, 'large').props.disabled, true, 'The shortcut cannot bypass memory admission')
assert.equal(choiceButton(menu, cloud.id, 'work-model', 'work'), undefined, 'The shortcut initially filters external routes')
const beforePick = picked.length, beforeClose = closes
choiceButton(menu, local.id, 'small').props.onClick()
for (let index = 0; index < 10; index++) await Promise.resolve()
assert.equal(picked.length, beforePick + 1)
assert.deepEqual(picked.at(-1), [local, 'small', null])
assert.equal(closes, beforeClose + 1)
menu = render(planningProps)
descendants(menu, item => item.type === 'button' && item.props.children === 'All')[0].props.onClick()
menu = render(planningProps)
assert.ok(choiceButton(menu, cloud.id, 'work-model', 'work'), 'The optional local scope does not hide the All escape hatch')
const closeBeforeFailure = closes
menu = render({ ...planningProps, onPickModel: async () => { throw new Error('Synthetic route save failed') } })
choiceButton(menu, cloud.id, 'work-model', 'work').props.onClick()
for (let index = 0; index < 10; index++) await Promise.resolve()
menu = render(planningProps)
assert.match(renderToStaticMarkup(menu), /Could not switch models:.*Synthetic route save failed/)
assert.equal(closes, closeBeforeFailure, 'A failed route save keeps the picker open for another choice')
assert.equal(choiceButton(menu, local.id, 'small').props.disabled, false, 'A failed pick releases the selection lock')
let prevented = false, stopped = false
menu.props.onKeyDown({ key: 'Escape', nativeEvent: { isComposing: false }, preventDefault() { prevented = true }, stopPropagation() { stopped = true } })
assert.ok(prevented && stopped)
assert.equal(closes, closeBeforeFailure + 1, 'Escape dismisses the planning shortcut without selecting a route')
console.log('Actual planning shortcut filters local replacements, preserves hardware gates, allows all routes, reports failed saves and supports keyboard dismissal.')

slots = []
const lan = { ...cloud, id: 'lan-fixture', kind: 'local', source: 'custom', transport: 'chat-completions', accounts: [], baseUrl: 'http://192.0.2.42:8888/v1', models: ['lan-model'] } as Provider
const lanProps = { ...planningProps, pickerProviders: [local, cloud, lan] }
menu = render(lanProps)
assert.equal(choiceButton(menu, lan.id, 'lan-model'), undefined)
descendants(menu, item => item.type === 'button' && item.props.children === 'All')[0].props.onClick()
menu = render(lanProps)
const lanButton = choiceButton(menu, lan.id, 'lan-model')
assert.ok(lanButton && !lanButton.props.disabled, 'A LAN route remains selectable through All')
lanButton.props.onClick()
for (let index = 0; index < 10; index++) await Promise.resolve()
assert.deepEqual(picked.at(-1), [lan, 'lan-model', null])
assert.equal(isExternalProvider(lan), true, 'Selecting a LAN provider never promotes it to trusted-local')
