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
const settings = { desktopMode: 'full', providers: [], uiPreferences: { locale: 'en' } }
const useStore = Object.assign((selector: any) => selector({ settings, saveSettings() {}, setView() {} }), { getState: () => ({ threads: [], settings }), setState() {} })
const { ConnectStep } = load('src/components/Onboarding.tsx', {
  '../lib/preferences/useFlowText': hook, '../lib/store': { useStore, requestSettingsSection() {} }, '../lib/ipc': { api: {} }, '../lib/icons': { Icon: () => null }, '../lib/brand': { BrandIcon: () => null, providerBrand: () => 'fixture' }, './ui': { Button, Input: (props: any) => React.createElement('input', props) },
  './settings/shared': { CLOUD_PROVIDER_PRESETS: [], LOCAL_RUNTIME_PRESETS: [], newProviderId: () => 'synthetic', providerPresetNeedsApiKey: () => true }, './onboarding-model': onboarding,
  './models/EngineChoice': { EngineChoice }, './BrandMark': { BrandMark: () => null }, '../lib/engine': { ENGINE_PROVIDER_ID: 'tensorfold', formatBytes: () => '100 B', useEngine: () => ({ snapshot: null }) },
  '../lib/full-workspace-layout': { savedFullWorkspaceLayout: () => 'chat', fullLayoutPreferenceReadable: () => true }, '../lib/onboarding-landing': { finishOnboardingLanding() {} },
  './onboarding/first-answer/FirstAnswerStep': { FirstAnswerStep: () => null }, './onboarding/first-answer/MinimalSetupGuide': { MinimalSetupGuide: () => null }, './onboarding/first-answer/model': firstModel,
  './onboarding/first-answer/transition': { openFirstAnswerDraft() {} }, '../lib/conversation-drafts': { conversationDraftsFor() {} }, '../lib/desktop-drafts': { desktopDraftPort: null },
}, '\nexports.ConnectStep = ConnectStep;')

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
