import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import * as profiles from '../../../shared/model-profiles'
import { defaultSettings } from '../../../electron/store/defaults'
import type { EngineInstalledModel, SaveEngineModelProfilesInput } from '../../lib/types'

const compilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true }
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = ({ children, ...props }: any) => React.createElement('button', props, children)
const Input = (props: any) => React.createElement('input', props)
let states: any[] = [], refs: any[] = [], cursor = 0, refCursor = 0
let settings = defaultSettings(), saves: SaveEngineModelProfilesInput[] = [], closed = 0, refreshes = 0
let mode: 'success' | 'reject' | 'pending' = 'success'
const pending = { resolve: null as (() => void) | null }
const useStore = Object.assign((selector: any) => selector({ settings }), { getState: () => ({ settings }), setState: (next: any) => { settings = next.settings } })
const imports: Record<string, unknown> = {
  react: { useState: (initial: any) => { const index = cursor++; if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial; return [states[index], (value: any) => { states[index] = typeof value === 'function' ? value(states[index]) : value }] }, useRef: (initial: any) => { const index = refCursor++; return refs[index] ??= { current: initial } }, useEffect() {} },
  'react/jsx-runtime': jsx, 'react-dom': { createPortal: (element: any) => element },
  '../../../shared/model-profiles': profiles,
  '../../lib/store': { useStore }, '../../lib/engine': { refreshEngine: async () => { refreshes++ } },
  '../../lib/ipc': { api: { engineSaveModelProfiles: async (input: SaveEngineModelProfilesInput) => {
    saves.push(structuredClone(input))
    if (mode === 'reject') throw new Error('Synthetic server busy; edits retained')
    if (mode === 'pending') await new Promise<void>((resolve) => { pending.resolve = resolve })
    return profiles.updateEngineModelProfiles(settings, input)
  } } }, '../ui': { Button, Input },
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'ModelProfileEditor.tsx'), 'utf8'), { compilerOptions }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed import ${id}`); return imports[id] }, loaded, loaded.exports)
const originalDocument = globalThis.document
Object.defineProperty(globalThis, 'document', { configurable: true, value: { body: {} } })
const model: EngineInstalledModel = { repo: 'example/model-a', modelType: 'qwen3_5', family: 'qwen', format: 'mlx', contextTokens: 32768, weightsBytes: 100, memoryFloorBytes: 100, fitsMemory: true, drafter: null, drafterInstalled: null }
function render(extra: any = {}) { cursor = 0; refCursor = 0; return loaded.exports.ModelProfileEditor({ model, onClose: () => { closed++ }, ...extra }) }
function all(node: any, predicate: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function byId(node: any, id: string) { const found = all(node, (entry) => entry.props?.id === id)[0]; assert.ok(found, id); return found }
function button(node: any, label: string) { const found = all(node, (entry) => entry.type === Button && entry.props.children === label)[0]; assert.ok(found, label); return found }
async function settle() { for (let i = 0; i < 5; i++) await Promise.resolve() }
function reset() { states = []; refs = []; saves = []; settings = defaultSettings(); mode = 'success'; closed = 0; refreshes = 0 }
try {
  reset()
  let node = render()
  assert.match(renderToStaticMarkup(node), /role="dialog"/)
  assert.match(renderToStaticMarkup(node), /aria-modal="true"/)
  assert.equal(byId(node, 'model-profile-contextTokens').props.value, 32768)
  byId(node, 'model-profile-temperature').props.onChange({ currentTarget: { value: '0.65' } })
  byId(node, 'model-profile-contextTokens').props.onChange({ currentTarget: { value: '8192' } })
  node = render()
  mode = 'reject'; button(node, 'Save & apply').props.onClick(); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /Synthetic server busy; edits retained/)
  assert.equal(byId(node, 'model-profile-temperature').props.value, 0.65)
  assert.equal(byId(node, 'model-profile-contextTokens').props.value, 8192)
  mode = 'success'; button(node, 'Save & apply').props.onClick(); await settle(); node = render()
  assert.equal(settings.engineModelProfiles?.[0].model, model.repo)
  assert.equal(settings.engineModelProfiles?.[0].profiles[0].temperature, 0.65)
  assert.equal(settings.engineModelProfiles?.[0].profiles[0].contextTokens, 8192)
  assert.equal(refreshes, 1)
  assert.match(renderToStaticMarkup(node), /Profile saved and applied/)
  button(node, 'Add large context').props.onClick(); node = render()
  assert.equal(byId(node, 'model-profile-contextTokens').props.value, 32768)
  button(node, 'Save & apply').props.onClick(); await settle(); node = render()
  assert.equal(settings.engineModelProfiles?.[0].profiles.length, 2)
  const active = settings.engineModelProfiles?.[0].activeProfileId
  button(node, 'Delete').props.onClick(); node = render()
  button(node, 'Save & apply').props.onClick(); await settle(); node = render()
  assert.equal(settings.engineModelProfiles?.[0].activeProfileId, null)
  assert.ok(!settings.engineModelProfiles?.[0].profiles.some((entry) => entry.id === active))
  assert.match(renderToStaticMarkup(node), /Engine defaults saved and applied/)

  reset(); node = render(); byId(node, 'model-profile-contextTokens').props.onChange({ currentTarget: { value: '' } }); node = render()
  assert.equal(byId(node, 'model-profile-contextTokens').props.value, '', 'clearing a number stays an editable blank draft')
  button(node, 'Save & apply').props.onClick(); await settle(); node = render()
  assert.equal(saves.length, 0); assert.match(renderToStaticMarkup(node), /Context window must/)
  button(node, 'Cancel').props.onClick(); assert.equal(closed, 1); assert.equal(settings.engineModelProfiles, undefined)

  reset(); node = render({ blocked: true }); assert.equal(button(node, 'Save & apply').props.disabled, true)
  button(node, 'Save & apply').props.onClick(); await settle(); assert.equal(saves.length, 0, 'busy-state direct handler invocation cannot save')
  node = render(); mode = 'pending'; button(node, 'Save & apply').props.onClick(); button(node, 'Save & apply').props.onClick(); await settle(); node = render()
  assert.equal(saves.length, 1, 'duplicate saves are synchronously rejected')
  assert.equal(button(node, 'Saving and applying…').props.disabled, true)
  button(node, 'Close').props.onClick(); assert.equal(closed, 0, 'pending host apply cannot be dismissed')
  pending.resolve?.(); await settle(); node = render(); assert.equal(button(node, 'Close').props.disabled, false)

  reset(); node = render();
  settings = profiles.updateEngineModelProfiles(settings, { model: model.repo, expectedRevision: 0, activeProfileId: null, profiles: [] }); node = render()
  assert.equal(button(node, 'Save & apply').props.disabled, true); button(node, 'Save & apply').props.onClick(); await settle(); assert.equal(saves.length, 0)
  assert.match(renderToStaticMarkup(node), /Profiles changed in another window/)

  for (const modelType of ['qwen4_exp', 'glm5_next']) {
    reset(); settings.temperature = 0.45
    const extra = { model: { ...model, modelType } }
    node = render(extra)
    assert.equal(byId(node, 'model-profile-temperature').props.value, 0.45, `${modelType} retains the default sampling temperature`)
    for (const field of ['temperature', 'topP', 'topK', 'minP']) assert.equal(byId(node, `model-profile-${field}`).props.disabled, false, `${modelType} ${field} is editable on the single-Mac engine`)
    assert.doesNotMatch(renderToStaticMarkup(node), /supports greedy decoding only/)
    for (const [field, value] of [['temperature', '0.65'], ['topP', '0.85'], ['topK', '30'], ['minP', '0.05']]) {
      byId(node, `model-profile-${field}`).props.onChange({ currentTarget: { value } }); node = render(extra)
    }
    button(node, 'Save & apply').props.onClick(); await settle(); node = render(extra)
    const saved = settings.engineModelProfiles?.[0].profiles[0]
    assert.deepEqual([saved?.temperature, saved?.topP, saved?.topK, saved?.minP], [0.65, 0.85, 30, 0.05], `${modelType} actual profile handlers persist sampling settings`)
    assert.match(renderToStaticMarkup(node), /Profile saved and applied/)
  }

  reset(); node = render({ model: { ...model, modelType: 'glm_moe_dsa' } })
  assert.equal(byId(node, 'model-profile-temperature').props.value, 0)
  for (const field of ['temperature', 'topP', 'topK', 'minP']) assert.equal(byId(node, `model-profile-${field}`).props.disabled, true)
  assert.match(renderToStaticMarkup(node), /supports greedy decoding only/)
  console.log('Actual profile editor handlers: edit/save/retry/presets/delete/cancel/invalid/stale/busy/duplicate and greedy model controls passed')
} finally { Object.defineProperty(globalThis, 'document', { configurable: true, value: originalDocument }) }
