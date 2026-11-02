import { reviewedLanguageImport } from '../../components/preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import clsx from 'clsx'
import { flowEnglish, flowSpanish, flowTextFor, flowTranslate } from './flow-catalogue'
import * as firstModel from '../../components/onboarding/first-answer/model'
import * as onboarding from '../../components/onboarding-model'
import * as modelFit from '../../../shared/model-fit'
import type { RunRecoverySnapshot } from '../../../shared/run-recovery'
import { engineReadiness } from '../engine-readiness'
import { repoDisplayName } from '../turn-stats-line'

assert.deepEqual(Object.keys(flowEnglish).sort(), Object.keys(flowSpanish).sort())
for (const [key, en] of Object.entries(flowEnglish)) {
  assert.ok(en.trim() && flowSpanish[key as keyof typeof flowSpanish].trim(), key)
  assert.deepEqual([...en.matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort(), [...flowSpanish[key as keyof typeof flowSpanish].matchAll(/\{(\w+)\}/g)].map(match => match[1]).sort(), `Placeholder parity ${key}`)
}
assert.equal(flowTranslate('fr', 'approval.allow'), 'Allow')
assert.equal(flowTranslate('es', 'missing.internal.key'), 'Interface text unavailable.')
assert.doesNotMatch(flowTranslate('es', 'first.selected', { model: '\u202E<script>fixture</script>' }), /\u202e/)
assert.equal(flowTextFor('es').authored('Unmodified provider diagnostic'), 'Unmodified provider diagnostic')

let locale = 'en', effects = 0
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = ({ variant: _v, size: _s, icon: _i, iconRight: _r, children, ...props }: any) => React.createElement('button', props, children)
const react = { useState: (initial: any) => [typeof initial === 'function' ? initial() : initial, () => {}], useRef: (current: any) => ({ current }), useEffect() {}, useCallback: (fn: any) => fn }
const hook = { useFlowText: () => flowTextFor(locale) }
function load(file: string, extra: Record<string, any> = {}, tail = '') {
  const module = { exports: {} as any }
  const imports: Record<string, any> = { react, clsx: { default: clsx }, 'react/jsx-runtime': jsx, ...extra }
  const compiled = ts.transpileModule(readFileSync(path.join(process.cwd(), file), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
  new Function('require', 'module', 'exports', compiled + tail)((name: string) => { const reviewed = reviewedLanguageImport(name, imports); if (reviewed.matched) return reviewed.value;  assert.ok(name in imports, `Reviewed locale component import ${name}`); return imports[name] }, module, module.exports)
  return module.exports
}
const first = load('src/components/onboarding/first-answer/FirstAnswerStep.tsx', { './model': firstModel, '../../../lib/preferences/useFlowText': hook }).FirstAnswerStep
const approval = load('src/components/ApprovalCard.tsx', { '../lib/preferences/useFlowText': hook, '../lib/icons': { Icon: () => null } }).ApprovalCard
const recovery = load('src/components/run-recovery/RunRecoveryCard.tsx', { '../../lib/preferences/useFlowText': hook, '../ui': { Button } }).RunRecoveryCard
const engine = load('src/components/models/EngineChoice.tsx', { '../../lib/preferences/useFlowText': hook, '../../lib/icons': { Icon: () => null }, '../../lib/engine': { useEngine: () => ({ snapshot: { status: { available: true, state: 'stopped', memoryLimitBytes: 100 }, installed: [] }, error: 'HTTP 503 original provider diagnostic' }), formatBytes: () => '100 B' }, '../../lib/engine-readiness': { engineReadiness }, '../../lib/turn-stats-line': { repoDisplayName }, '../ui': { Button }, './EngineStatus': { engineDotTone: () => '' }, '../BrandMark': { BrandMark: () => null } }).EngineChoice
const settings = { desktopMode: 'full', providers: [], uiPreferences: { locale: 'en' } }
const useStore = Object.assign((selector: any) => selector({ settings, saveSettings() {}, setView() {} }), { getState: () => ({ threads: [], settings }), setState() {} })
const onboardingComponents = load('src/components/Onboarding.tsx', {
  '../lib/preferences/useFlowText': hook, '../lib/store': { useStore, requestSettingsSection() {} }, '../lib/ipc': { api: {} }, '../lib/icons': { Icon: () => null }, '../lib/brand': { BrandIcon: () => null, providerBrand: () => 'fixture' }, './ui': { Button, Input: (props: any) => React.createElement('input', props) },
  './settings/shared': { CLOUD_PROVIDER_PRESETS: [], LOCAL_RUNTIME_PRESETS: [], newProviderId: () => 'synthetic', providerPresetNeedsApiKey: () => true }, './onboarding-model': onboarding,
  './models/EngineChoice': { EngineChoice: engine }, './BrandMark': { BrandMark: () => null }, '../lib/engine': { ENGINE_PROVIDER_ID: 'tensorfold', formatBytes: () => '100 B', useEngine: () => ({ snapshot: null }) },
  '../lib/full-workspace-layout': { savedFullWorkspaceLayout: () => 'chat', fullLayoutPreferenceReadable: () => true }, '../lib/onboarding-landing': { finishOnboardingLanding() {} },
  './onboarding/first-answer/FirstAnswerStep': { FirstAnswerStep: first }, './onboarding/first-answer/MinimalSetupGuide': { MinimalSetupGuide: () => null }, './onboarding/first-answer/model': firstModel,
  './onboarding/first-answer/transition': { openFirstAnswerDraft() {} }, '../lib/conversation-drafts': { conversationDraftsFor() {} }, '../lib/desktop-drafts': { desktopDraftPort: null },
  '../lib/model-chat': { openModelChat() {} },
}, '\nexports.WorkspaceChoice = WorkspaceChoice; exports.ConnectStep = ConnectStep;')
const ownership = {}
const qualificationReport = { providerId: 'original-provider', accountId: 'original-account', model: 'Original / 模型', key: modelFit.modelQualificationKey({ providerId: 'original-provider', accountId: 'original-account', model: 'Original / 模型' }), checks: { ...modelFit.emptyModelChecks(), tools: { state: 'failed', message: 'The synthetic tool call contained malformed or schema-invalid arguments; no tools were executed.' } }, checkedAt: 1, probeRevision: 1, revision: 1 }
const qualification = load('src/components/models/qualification/ModelQualificationPanel.tsx', { react: { ...react, useState: (initial: any) => [initial === null ? qualificationReport : initial, () => {}] }, '../../../lib/preferences/useFlowText': hook, '../../../../shared/model-fit': modelFit, '../../../lib/ipc': { api: {} }, '../../ui': { Button }, './useModelQualifications': { useModelQualifications: () => ({ reports: [qualificationReport], error: '', refresh() {}, ownership }) } }).ModelQualificationPanel
const state: RunRecoverySnapshot = { runId: 'synthetic', state: 'stopping', reason: 'user_stop', message: 'Stop requested; waiting for already-started work to settle.', completedTools: 2, completedMutations: 1, pendingOperations: 1, pendingMutations: 1, delivery: 'complete', retry: 'unavailable', retryReason: '', canStartNewRun: false, updatedAt: 1 }
const collect = (node: any, predicate: (n: any) => boolean): any[] => !node || typeof node !== 'object' ? [] : [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap(child => collect(child, predicate))]
for (const language of ['en', 'es']) {
  locale = language
  const setupHtml = renderToStaticMarkup(React.createElement(onboardingComponents.Onboarding, { onDone() {} }))
  assert.match(setupHtml, language === 'es' ? /¿Cómo vas a usar TensorFold/ : /How will you use TensorFold/)
  assert.match(setupHtml, language === 'es' ? /conversaciones existentes.*código.*diagnósticos originales/ : /Existing conversations, code and original diagnostics stay unchanged/)
  const choiceTree = onboardingComponents.WorkspaceChoice({ value: 'full', onChange: (value: string) => { assert.equal(value, 'minimal'); effects++ } })
  collect(choiceTree, node => node.type === 'input' && node.props.value === 'minimal')[0].props.onChange()
  const connectTree = onboardingComponents.ConnectStep({ providerStep: 'configure', mode: 'cloud', preset: { label: 'Original provider', auth: 'api-key', provider: {}, desc: 'Frontier API with GPT and reasoning models.' }, apiKey: 'synthetic', localPath: '', busy: false, error: 'HTTP 401 Original provider diagnostic', foundModels: null, cliAuth: null, desktopMode: 'full', onConnect: () => { effects++ }, onContinue() {}, setProviderStep() {}, onApiKeyChange() {} })
  const connectHtml = renderToStaticMarkup(connectTree)
  assert.match(connectHtml, language === 'es' ? /Conectar modelo/ : /Connect model/)
  assert.match(connectHtml, language === 'es' ? /La configuración del modelo necesita atención/ : /Model setup needs attention/)
  assert.match(connectHtml, /HTTP 401 Original provider diagnostic/)
  collect(connectTree, node => node.type === Button && node.props.icon === 'check')[0].props.onClick()
  const qualificationTree = qualification({ route: { providerId: qualificationReport.providerId, accountId: qualificationReport.accountId, model: qualificationReport.model }, disabled: true, unavailableReason: 'Load this model explicitly before checking its capabilities; a check never starts a model automatically.' })
  const qualificationHtml = renderToStaticMarkup(qualificationTree)
  assert.match(qualificationHtml, language === 'es' ? /Comprobaciones de capacidad/ : /Capability checks/)
  assert.match(qualificationHtml, language === 'es' ? /argumentos mal formados/ : /malformed or schema-invalid arguments/)
  assert.match(qualificationHtml, /original-provider/)
  assert.ok(qualificationHtml.includes('Original / 模型'))
  assert.equal(collect(qualificationTree, node => node.type === Button)[0].props.disabled, true)
  const firstTree = first({ starter: 'plan', onStarter: () => { effects++ }, model: 'Original / 模型', readiness: { providerKind: 'local', builtIn: true, selected: true, cached: false, configured: false, loading: false, loaded: false, fitsMemory: null } })
  const firstHtml = renderToStaticMarkup(firstTree)
  assert.match(firstHtml, language === 'es' ? /El camino a tu primera respuesta/ : /Your path to a first answer/)
  assert.match(firstHtml, language === 'es' ? /Vuelve a Modelos para completar/ : /Return to Models to finish/)
  assert.ok(firstHtml.includes(flowTextFor(language).authored(firstModel.FIRST_ANSWER_STARTERS[0].prompt)), 'Only predefined new starter prompts are localized; existing conversation content stays original')
  collect(firstTree, node => node.type === 'input' && node.props.value === 'plan')[0].props.onChange()
  const approvalTree = approval({ title: 'Run', subject: 'Choose a model\nOriginal 用户文字', mono: true, reason: 'sending email requires approval', canAllowScoped: true, onAllow: () => { effects++ }, onDeny: () => { effects++ }, onAllowScoped: () => { effects++ } })
  const approvalHtml = renderToStaticMarkup(approvalTree)
  assert.match(approvalHtml, language === 'es' ? /¿Ejecutar este comando/ : /Run this command/)
  assert.match(approvalHtml, language === 'es' ? /enviar correo requiere aprobación/ : /sending email requires approval/)
  assert.match(approvalHtml, /Choose a model\nOriginal 用户文字/, 'Tool arguments are never translated')
  for (const button of collect(approvalTree, node => node.type === 'button')) button.props.onClick()
  const recoveryTree = recovery({ recovery: state, onEditRequest: () => { effects++ }, onReviewEffects: () => { effects++ } })
  const recoveryHtml = renderToStaticMarkup(recoveryTree)
  assert.match(recoveryHtml, language === 'es' ? /Deteniendo/ : /Stopping/)
  assert.match(recoveryHtml, language === 'es' ? /1 operación todavía activa/ : /1 operation still active/)
  assert.match(recoveryHtml, language === 'es' ? /Se solicitó la detención/ : /Stop requested/)
  assert.ok(collect(recoveryTree, node => node.type === Button).every(button => button.props.disabled === true), 'Localization does not bypass the drain gate')
  const engineHtml = renderToStaticMarkup(React.createElement(engine, { onOpenModels() { effects++ }, onUse() { effects++ } }))
  assert.match(engineHtml, language === 'es' ? /Elegir un modelo/ : /Choose a model/)
  assert.match(engineHtml, /HTTP 503 original provider diagnostic/)
  assert.match(engineHtml, language === 'es' ? /Diagnóstico, texto original/ : /Diagnostic, original text/)
  assert.doesNotMatch(firstHtml + approvalHtml + recoveryHtml + engineHtml, /setup\.[a-z]|recovery\.[a-z]|qualification\.[a-z]/)
}
assert.equal(effects, 12, 'Only the explicit starter and approval callbacks ran; translated labels add no model or host action')
async function verifyOutcomeReviewDialog() {
  const previousDocument = globalThis.document
  ;(globalThis as any).document = { body: {} }
  try {
    for (const language of ['en', 'es']) {
      locale = language
      let cursor = 0, closes = 0, apiCalls = 0
      const slots: any[] = []
      let modal: any = null, resolveReview!: (value: { reviewedRunId: string }) => void, rejectReview!: (error: Error) => void
      const thread = { id: 'synthetic-review-chat', runRecovery: { ...state, state: 'needs_review', runId: 'exact-synthetic-run', message: 'HTTP 503 original run diagnostic' } }
      let saved = { threads: [{ ...thread }, { id: 'other-chat', runRecovery: { runId: 'exact-synthetic-run' } }] as any[] }
      const calls: any[] = []
      const review = load('src/components/run-recovery/RunOutcomeReviewDialog.tsx', {
        react: { useState(initial: any) { const id = cursor++; if (!(id in slots)) slots[id] = initial; return [slots[id], (next: any) => { slots[id] = next }] }, useRef(initial: any) { const id = cursor++; return slots[id] ??= { current: initial } } },
        'react-dom': { createPortal: (node: any) => node }, '../../lib/preferences/useFlowText': hook,
        '../../lib/ipc': { api: { reviewRunOutcome: (input: any) => { apiCalls++; calls.push(input); return new Promise((resolve, reject) => { resolveReview = resolve; rejectReview = reject }) } } },
        '../../lib/store': { useStore: { setState: (update: any) => { saved = { ...saved, ...update(saved) } } } },
        '../navigation/useAccessibleDialog': { useAccessibleDialog: (options: any) => { modal = options } },
      }).RunOutcomeReviewDialog
      const render = () => { cursor = 0; return review({ thread, onClose: () => { closes++ } }) }
      const text = (node: any): string => typeof node === 'string' ? node : React.Children.toArray(node?.props?.children).map(text).join('')
      const recordButton = (tree: any) => collect(tree, node => node.type === 'button' && text(node) === flowTranslate(language, 'review.record'))[0]
      let tree = render()
      const markup = renderToStaticMarkup(tree)
      assert.match(markup, language === 'es' ? /Revisa el resultado anterior/ : /Review the previous outcome/)
      assert.match(markup, language === 'es' ? /He comprobado el resultado anterior/ : /I have checked the previous outcome/)
      assert.match(markup, language === 'es' ? /nunca vuelve a enviar un mensaje ni repite una acción/ : /never resends a message or repeats an action/)
      assert.match(markup, language === 'es' ? /Diagnóstico, texto original/ : /Diagnostic, original text/)
      assert.match(markup, /HTTP 503 original run diagnostic/)
      assert.equal(modal.open, true); assert.ok(modal.dialogRef)
      const dialog = collect(tree, node => node.props.role === 'dialog')[0]
      assert.equal(dialog.props['aria-modal'], 'true'); assert.equal(dialog.props['aria-labelledby'], 'run-outcome-title'); assert.equal(dialog.props.tabIndex, -1)
      assert.equal(recordButton(tree).props.disabled, true)
      recordButton(tree).props.onClick(); assert.equal(apiCalls, 0, 'Unchecked confirmation cannot acknowledge a run')
      collect(tree, node => node.type === 'input' && node.props.type === 'checkbox')[0].props.onChange({ target: { checked: true } })
      tree = render(); assert.equal(recordButton(tree).props.disabled, false)
      recordButton(tree).props.onClick(); tree = render()
      assert.deepEqual(calls[0], { threadId: thread.id, runId: 'exact-synthetic-run' })
      assert.equal(collect(tree, node => node.type === 'input')[0].props.disabled, true)
      assert.ok(collect(tree, node => node.type === 'button').every(button => button.props.disabled))
      assert.match(renderToStaticMarkup(tree), language === 'es' ? /Guardando la revisión/ : /Saving review/)
      modal.onClose(); assert.equal(closes, 0, 'Accessible-dialog Escape remains blocked while saving')
      rejectReview(new Error('Original native refusal'))
      for (let i = 0; i < 8; i++) await Promise.resolve()
      tree = render(); assert.match(renderToStaticMarkup(tree), language === 'es' ? /La ejecución cambió o no se pudo guardar tu revisión/ : /The run changed or could not save your review/)
      assert.equal(closes, 0); assert.equal(saved.threads[0].runRecoveryReviewedId, undefined)
      recordButton(tree).props.onClick(); resolveReview({ reviewedRunId: 'exact-synthetic-run' })
      for (let i = 0; i < 8; i++) await Promise.resolve()
      assert.equal(apiCalls, 2); assert.equal(closes, 1)
      assert.equal(saved.threads[0].runRecoveryReviewedId, 'exact-synthetic-run')
      assert.equal(saved.threads[1].runRecoveryReviewedId, undefined, 'Acknowledgement never marks another chat reviewed')
    }
  } finally { (globalThis as any).document = previousDocument }
}
await verifyOutcomeReviewDialog()
console.log('PASS stable flow catalogue parity/fallback and real English/Spanish first-answer, approval, recovery/outcome review and model-error interfaces with unchanged draft/argument content and action semantics')
