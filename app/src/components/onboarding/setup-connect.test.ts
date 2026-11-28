import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import { flowTextFor } from '../../lib/preferences/flow-catalogue'
import * as firstModel from './first-answer/model'
import * as onboarding from '../onboarding-model'

// Full setup's connect step with installed models, rendered from the production components without an app boot.
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = ({ variant: _v, size: _s, icon: _i, iconRight: _r, children, ...props }: any) => React.createElement('button', props, children)
const react = { useState: (initial: any) => [typeof initial === 'function' ? initial() : initial, () => {}], useRef: (current: any) => ({ current }), useEffect() {}, useCallback: (fn: any) => fn }
const hook = { useFlowText: () => flowTextFor('en') }
function load(file: string, extra: Record<string, any> = {}, tail = '') {
  const module = { exports: {} as any }
  const imports: Record<string, any> = { react, clsx: { default: clsx }, 'react/jsx-runtime': jsx, ...extra }
  const compiled = ts.transpileModule(readFileSync(path.join(process.cwd(), file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
  new Function('require', 'module', 'exports', compiled + tail)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value; assert.ok(name in imports, `Reviewed setup component import ${name}`); return imports[name] }, module, module.exports)
  return module.exports
}
const installed = [
  { repo: 'synthetic/first-model', title: 'First synthetic model', fitsMemory: true },
  { repo: 'synthetic/second-model', fitsMemory: true },
]
const EngineChoice = load('src/components/models/EngineChoice.tsx', { '../../lib/preferences/useFlowText': hook, '../../lib/icons': { Icon: () => null }, '../../lib/engine': { useEngine: () => ({ snapshot: { status: { available: true, state: 'stopped', memoryLimitBytes: 100 }, installed }, error: '' }), formatBytes: () => '100 B' }, '../ui': { Button }, './EngineStatus': { engineDotTone: () => '' }, '../BrandMark': { BrandMark: () => null } }).EngineChoice
const Field = load('src/components/onboarding/Field.tsx').Field
const settings = { desktopMode: 'full', providers: [], uiPreferences: { locale: 'en' } }
const useStore = Object.assign((selector: any) => selector({ settings, saveSettings() {}, setView() {} }), { getState: () => ({ threads: [], settings }), setState() {} })
const componentImports = {
  '../lib/preferences/useFlowText': hook, '../lib/store': { useStore, requestSettingsSection() {} }, '../lib/ipc': { api: {} }, '../lib/icons': { Icon: () => null }, '../lib/brand': { BrandIcon: () => null, providerBrand: () => 'fixture' }, './ui': { Button, Input: (props: any) => React.createElement('input', props) },
  './settings/shared': { CLOUD_PROVIDER_PRESETS: [], LOCAL_RUNTIME_PRESETS: [], newProviderId: () => 'synthetic', providerPresetNeedsApiKey: () => true }, './onboarding-model': onboarding,
  './models/EngineChoice': { EngineChoice }, './onboarding/Field': { Field }, './BrandMark': { BrandMark: () => null }, '../lib/engine': { ENGINE_PROVIDER_ID: 'tensorfold', formatBytes: () => '100 B', useEngine: () => ({ snapshot: null }) },
  '../lib/full-workspace-layout': { savedFullWorkspaceLayout: () => 'chat', fullLayoutPreferenceReadable: () => true }, '../lib/onboarding-landing': { finishOnboardingLanding() {} },
  './onboarding/first-answer/FirstAnswerStep': { FirstAnswerStep: () => null }, './onboarding/first-answer/MinimalSetupGuide': { MinimalSetupGuide: () => null }, './onboarding/first-answer/model': firstModel,
  './onboarding/first-answer/transition': { openFirstAnswerDraft() {} }, '../lib/conversation-drafts': { conversationDraftsFor() {} }, '../lib/desktop-drafts': { desktopDraftPort: null },
}
const { ConnectStep } = load('src/components/Onboarding.tsx', componentImports, '\nexports.ConnectStep = ConnectStep;')

const used: string[] = []
const connect = (connectedLabel: string | null) => renderToStaticMarkup(React.createElement(ConnectStep, {
  providerStep: 'choice', setProviderStep() {}, mode: 'cloud', preset: null, apiKey: '', localPath: '', busy: false, error: '', foundModels: null, cliAuth: null,
  connectedLabel, desktopMode: 'full', onPickCloud() {}, onPickLocal() {}, onDetect() {}, onConnect() {}, onApiKeyChange() {}, onLocalPathChange() {},
  onOpenModels() {}, onOpenEngineModels() {}, onUseEngineModel(repo: string) { used.push(repo) }, onContinue() {},
}))
const connected = connect('TensorFold (built in)')

// Each model action names its model, so a screen reader lists distinct actions rather than several plain "Use" buttons.
assert.match(connected, /aria-label="Use First synthetic model"/)
assert.match(connected, /aria-label="Use synthetic\/second-model"/, 'a model without a title is named by its repository')
assert.equal((connected.match(/aria-label="Use /gu) ?? []).length, installed.length)

// Connected, the step's own Continue is gone: the footer's Continue is the one control that advances.
assert.equal((connected.match(/>Continue</gu) ?? []).length, 0, 'no second Continue inside the connected step')
assert.doesNotMatch(connected, /Continue without a model/)
assert.equal((connect(null).match(/Continue without a model/gu) ?? []).length, 1, 'without a connection the step still offers to go on without a model')

// The Connected status reads as ink on the positive tint (about 15:1), not positive green on it (about 2.9:1).
const status = connected.match(/<div class="([^"]*bg-positive\/10[^"]*)">/u)
assert.ok(status, 'the connected status block renders')
assert.doesNotMatch(status[1], /\btext-positive\b/)
assert.match(status[1], /\btext-ink\b/)
assert.doesNotMatch(connected, /class="mt-1 text-ink-3">TensorFold \(built in\)/)
assert.deepEqual(used, [], 'rendering runs no model action')
console.log('PASS setup connect step: named model actions, one Continue when connected, legible Connected status')

type TestProps = Record<string, unknown> & { children?: React.ReactNode }
type TestElement = React.ReactElement<TestProps>
type EffectSlot = { deps?: readonly unknown[]; cleanup?: void | (() => void) }
type DocumentState = { body: object; activeElement: unknown; querySelector?: (selector: string) => { focus(): void } }
function isElement(node: React.ReactNode): node is TestElement {
  return typeof node === 'object' && node !== null && 'type' in node && 'props' in node
}

// Slots are heterogeneous; each generic cast recovers the value stored by the same hook index.
class Hooks {
  slots: unknown[] = []; cursor = 0; pending: Array<() => void> = []; beforeEffects?: (tree: React.ReactNode) => void
  react = {
    useState: <T>(initial: T | (() => T)): [T, React.Dispatch<React.SetStateAction<T>>] => {
      const index = this.cursor++
      if (!(index in this.slots)) this.slots[index] = typeof initial === 'function' ? (initial as () => T)() : initial
      const set = (next: React.SetStateAction<T>) => { this.slots[index] = typeof next === 'function' ? (next as (previous: T) => T)(this.slots[index] as T) : next }
      return [this.slots[index] as T, set]
    },
    useRef: <T>(initial: T): React.MutableRefObject<T> => {
      const index = this.cursor++
      const ref = (this.slots[index] ??= { current: initial }) as React.MutableRefObject<T>
      return ref
    },
    useEffect: (fn: () => void | (() => void), deps?: readonly unknown[]) => {
      const index = this.cursor++
      const old = this.slots[index] as EffectSlot | undefined
      if (!old || !deps || deps.some((item, n) => item !== old.deps?.[n])) this.pending.push(() => { old?.cleanup?.(); this.slots[index] = { deps, cleanup: fn() } satisfies EffectSlot })
    },
  }
  render<P>(fn: (props: P) => React.ReactNode, props: P): React.ReactNode {
    this.cursor = 0
    const tree = fn(props)
    this.beforeEffects?.(tree)
    const effects = this.pending.splice(0)
    for (const effect of effects) effect()
    return tree
  }
}
function materialize(node: React.ReactNode): React.ReactNode {
  if (!isElement(node)) return node
  if (typeof node.type === 'function') {
    // This harness renders function components only; host elements retain their native type.
    const component = node.type as (props: TestProps) => React.ReactNode
    return materialize(component(node.props))
  }
  return { ...node, props: { ...node.props, children: React.Children.toArray(node.props.children).map(materialize) } }
}
function all(node: React.ReactNode, predicate: (entry: TestElement) => boolean): TestElement[] {
  if (!isElement(node)) return []
  return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props.children).flatMap((child) => all(child, predicate))]
}
function text(node: React.ReactNode): string {
  return React.Children.toArray(node).map((child) => isElement(child) ? text(child.props.children) : String(child)).join('')
}
async function settle() { for (let i = 0; i < 8; i++) await Promise.resolve() }
const inlineContinue = (node: TestElement) => node.type === 'button' && text(node.props.children) === 'Continue without a model'

async function readinessFocus(mode: 'focus' | 'blur' | 'navigation') {
  const hooks = new Hooks()
  const focusSettings = { desktopMode: 'full', desktopOnboardingComplete: false, activeProviderId: 'tensorfold', activeModel: 'synthetic/first-model', providers: [{ id: 'tensorfold', name: 'TensorFold', models: ['synthetic/first-model'] }], uiPreferences: { locale: 'en' } }
  const focusStore = Object.assign((selector: (state: { settings: typeof focusSettings; saveSettings: () => void; setView: () => void }) => unknown) => selector({ settings: focusSettings, saveSettings() {}, setView() {} }), { getState: () => ({ threads: [], settings: focusSettings, view: { name: 'home' } }), setState() {} })
  const { Onboarding } = load('src/components/Onboarding.tsx', {
    ...componentImports, react: hooks.react, '../lib/store': { useStore: focusStore, requestSettingsSection() {} },
    '../lib/ipc': { api: { getModelReadiness: async () => ({ ready: true }) } },
    './onboarding-model': { ...onboarding, readOnboardingDraft: () => ({ mode: 'full', step: 'connect' }) },
  })
  const body = {}, elsewhere = {}
  let footerFocuses = 0
  const documentState: DocumentState = { body, activeElement: body }
  const footer = { focus() { footerFocuses++; documentState.activeElement = footer } }
  documentState.querySelector = (selector: string) => { assert.equal(selector, '[data-testid="onboarding-next"]'); return footer }
  const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document')
  Object.defineProperty(globalThis, 'document', { configurable: true, value: documentState })
  try {
    const disconnected = materialize(hooks.render(Onboarding, { onDone() {} }))
    const inline = all(disconnected, inlineContinue)[0]
    assert.ok(inline, 'checking readiness retains the inline continuation')
    const onFocus = inline.props.onFocus
    assert.ok(typeof onFocus === 'function', 'the inline continuation records focus before readiness hides it')
    documentState.activeElement = inline
    onFocus()
    if (mode === 'blur') {
      const onBlur = inline.props.onBlur
      assert.ok(typeof onBlur === 'function')
      onBlur()
      documentState.activeElement = elsewhere
    }
    if (mode === 'navigation') {
      const onClick = inline.props.onClick
      assert.ok(typeof onClick === 'function')
      onClick()
      materialize(hooks.render(Onboarding, { onDone() {} }))
      await settle()
      const guide = materialize(hooks.render(Onboarding, { onDone() {} }))
      const back = all(guide, (node) => node.props['data-testid'] === 'onboarding-back')[0]
      assert.ok(back)
      const onBack = back.props.onClick
      assert.ok(typeof onBack === 'function')
      documentState.activeElement = elsewhere
      onBack()
      const returned = materialize(hooks.render(Onboarding, { onDone() {} }))
      assert.equal(all(returned, inlineContinue).length, 0)
      assert.equal(all(returned, (node) => node.props['data-testid'] === 'onboarding-next').length, 1)
      assert.equal(documentState.activeElement, elsewhere, 'a later return does not reuse stale inline focus')
      assert.equal(footerFocuses, 0)
      return
    }
    await settle()
    hooks.beforeEffects = (tree) => {
      assert.equal(all(materialize(tree), inlineContinue).length, 0, 'readiness removes the inline continuation')
      if (mode === 'focus') documentState.activeElement = body // Removing a focused DOM node resets focus to the body.
    }
    const connected = materialize(hooks.render(Onboarding, { onDone() {} }))
    assert.equal(all(connected, (node) => node.props['data-testid'] === 'onboarding-next').length, 1)
    assert.equal(documentState.activeElement, mode === 'blur' ? elsewhere : footer)
    assert.equal(footerFocuses, mode === 'blur' ? 0 : 1)
  } finally {
    if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument)
    else Reflect.deleteProperty(globalThis, 'document')
  }
}

// Readiness transfers current inline focus only; blur and navigation cannot leave a stale focus claim.
await readinessFocus('focus')
await readinessFocus('blur')
await readinessFocus('navigation')
console.log('PASS setup connect focus: readiness transfers current focus and preserves blurred or navigated focus')
