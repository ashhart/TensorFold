import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import React from 'react'
import ts from 'typescript'
import clsx from 'clsx'
import * as fit from '../../../shared/model-fit'
import * as trust from '../../../shared/provider-trust'
import * as memoryPolicy from '../../../shared/thread-memory-policy'
import * as policy from '../../lib/policy'
import * as modelSelection from '../../lib/assistant-model'
import * as providerAccounts from '../../lib/provider-accounts'
import * as providerMetadata from '../../lib/provider-model-metadata'
import * as composerBudgetModule from '../../lib/composer-budget'
import * as composerKeyboard from '../../lib/composer-keyboard'
import * as workspace from '../../lib/thread-workspace'
import * as execution from '../../lib/execution-target'
import * as attachmentsModule from '../../lib/attachments'
import * as format from '../../lib/format'
import { modelPickerChoices } from '../../lib/model-picker'
import { find, harness, jsx, settle } from '../reading/reading-test-harness'

function load(file: string, overrides: Record<string, any>) {
  const source = readFileSync(file, 'utf8'), ast = ts.createSourceFile(file, source, ts.ScriptTarget.ES2022, true), imports: Record<string, any> = {}
  for (const node of ast.statements) if (ts.isImportDeclaration(node) && ts.isStringLiteral(node.moduleSpecifier)) {
    const name = node.moduleSpecifier.text
    const reviewed = reviewedLanguageImport(name)
    if (reviewed.matched) { imports[name] = reviewed.value; continue }
    imports[name] = new Proxy({ __esModule: true }, { get(object, key) { if (key in object) return (object as any)[key]; return () => { throw new Error(`Unexpected Composer adapter: ${name}.${String(key)}`) } } })
  }
  Object.assign(imports, overrides)
  const module = { exports: {} as any }, output = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', output)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports); return imports[id] }, module, module.exports)
  return module.exports
}
const local: any = { id: 'fit-local', name: 'Synthetic local model', kind: 'local', source: 'custom', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:12345/v1', apiKey: '', models: ['selected'] }
const image: any = { id: 'synthetic-image', kind: 'image', name: 'Synthetic.png', mime: 'image/png', status: 'ready', dataUrl: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=' }
async function scenario(provider: any, attached: any[], visionState = 'failed', task = false) {
  const hooks = harness(), stateHook = hooks.hooks.useState
  let stateCalls = 0, sends = 0, cleared = 0
  hooks.hooks.useLayoutEffect = hooks.hooks.useEffect
  hooks.hooks.useState = (initial: any) => stateHook(stateCalls++ === 0 ? { ready: true, reason: null, message: 'Synthetic model ready.' } : initial)
  const thread: any = { id: 'selected-thread', kind: task ? 'task' : 'chat', title: 'Synthetic selected model', cwd: task ? '/synthetic/workspace' : '', providerId: provider.id, providerAccountId: null, model: 'selected', timeline: [], apiMessages: [], status: 'idle' }
  const settings: any = { providers: [provider], activeProviderId: provider.id, activeProviderAccountId: null, activeModel: 'selected', approvalMode: 'suggest', reasoningEffort: 'low', contextLimit: 8192, localContextLimit: 8192, defaultCwd: '', voice: { enabled: false }, agentPresets: [], remoteOrchestrators: [], defaultExecutionTarget: 'local', capabilities: {} }
  const notices: any[] = []
  const state: any = { settings, status: {}, projects: [], terminals: {}, activeTerminalId: {}, sshHosts: [], sshStatus: {}, pendingMention: null, composerSeed: null,
    send: async () => { sends++; return true }, launch: async () => { sends++; return { ok: true } }, setNotice: (notice: any) => notices.push(notice), setView() {},
  }
  const useStore = Object.assign((select: any) => select(state), { getState: () => state, setState: (patch: any) => { if (patch.notice) notices.push(patch.notice); Object.assign(state, patch) } })
  const reports: any[] = [{ providerId: provider.id, accountId: null, model: 'selected', key: fit.modelQualificationKey({ providerId: provider.id, accountId: null, model: 'selected' }), probeRevision: fit.MODEL_PROBE_REVISION, revision: 1, checkedAt: 1, checks: { ...fit.emptyModelChecks(), text: { state: 'observed', message: 'Synthetic text qualified.' }, vision: { state: visionState, message: 'Synthetic image evidence.' } } }]
  const captured = { text: 'Answer the supplied question.', attachments: attached }
  const { Composer } = load('src/components/Composer.tsx', {
    react: hooks.hooks, 'react/jsx-runtime': jsx, clsx, '../lib/store': { useStore }, '../lib/ipc': { api: { platform: 'browser' } },
    './models/qualification/useModelQualifications': { useModelQualifications: () => ({ reports }) }, '../../shared/model-fit': fit, '../../shared/provider-trust': trust, '../../shared/thread-memory-policy': memoryPolicy,
    '../lib/policy': policy, '../lib/assistant-model': modelSelection, '../lib/provider-accounts': providerAccounts, '../lib/provider-model-metadata': providerMetadata, '../lib/thread-workspace': workspace, '../lib/execution-target': execution, '../lib/attachments': attachmentsModule, '../lib/format': format,
    '../lib/use-conversation-draft': { useConversationDraft: () => ({ ...captured, ready: true, error: '', scopeKey: 'synthetic-scope', setText() {}, setAttachments() {}, capture: () => captured, clearAccepted: async () => { cleared++ } }) },
    '../lib/desktop-drafts': { desktopDraftPort: null }, '../lib/composer-estimate': { boundedComposerEstimate: () => ({ tokens: 10, partial: false }) }, '../lib/engine': { useEngineContextTokens: () => null }, '../lib/provider-usage': { providerUsageSummary: () => null }, '../lib/provider-status': { providerStatusViewFor: () => null },
    '../lib/composer-budget': composerBudgetModule, '../lib/composer-keyboard': composerKeyboard,
    '../lib/recent-models': { loadRecentModels: () => [] }, '../lib/capabilities': { BUILTIN_CAPABILITIES: [] }, './composer/ModelPickerMenu': { ModelPickerMenu: 'model-picker-fixture' }, './composer/TaskStarters': { TASK_STARTERS: [], TaskStarters: 'task-starters-fixture' },
  })
  const tree = hooks.render(() => Composer({ mode: 'thread', thread }))
  const picker = find(tree, (node) => node.type === 'model-picker-fixture')
  assert.equal(picker.props.model, 'selected'); assert.equal(picker.props.provider.id, provider.id)
  const choices = modelPickerChoices([provider], null, { requirements: picker.props.requirements, qualifications: reports })
  find(tree, (node) => node.type === 'button' && node.props['aria-label'] === 'Send').props.onClick()
  await settle()
  return { sends, cleared, notices, picker: picker.props, choice: choices[0] }
}
const failedImage = await scenario(local, [image])
assert.match(failedImage.choice.disabledReason!, /vision check/, 'The real picker already rejects this exact active provider/account/model for the attached image')
assert.equal(failedImage.sends, 0, 'The already-selected model must not dispatch from actual Composer.submit after its required image check failed')
assert.equal(failedImage.cleared, 0, 'A capability refusal preserves the accepted draft token')
const textOnly = await scenario(local, [])
assert.equal(textOnly.sends, 1, 'A failed image check cannot block text-only work')
const unknownImage = await scenario(local, [image], 'unknown')
assert.equal(unknownImage.sends, 1, 'Unknown capability remains explicitly unqualified rather than an invented failure')
for (const transport of ['chat-completions', 'codex-cli']) {
  const external = { ...local, id: `planner-${transport}`, kind: 'cloud', transport, baseUrl: 'https://example.invalid/v1' }
  const planning = await scenario(external, [], 'unknown', true)
  assert.equal(planning.sends, 1, 'A workspace path alone must not forbid the existing external planning-only route')
}
console.log('Actual Composer active selection uses the same required-evidence refusal as its picker while preserving drafts, text-only work, unknown evidence and external planning routes.')
