import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import ts from 'typescript'
import { createSerialTaskQueue } from '../../agent/serial-task-queue'
import { redactSettings } from '../../agent/security'
import { importReviewedRecipe, reviewedRecipeExport, exportRecipePreview } from './bot-recipes'
import { WorkspaceMutationGuard } from './workspace-mutation-guard'
import type { Settings } from '../../agent/types'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-recipe-transaction-'))
process.env.VONTRA_HOME = profile
const { defaultSettings, saveSettings, loadSettings, closeSessionSearchIndexes } = await import('../../store')
const provider = { id: 'local-fixture', name: 'Local fixture', kind: 'local' as const, baseUrl: 'http://127.0.0.1:9090/v1', apiKey: 'synthetic-recipe-provider-key', models: ['fixture'] }
const original = { id: 'existing_bot', name: 'Evidence helper', role: 'Review evidence', lane: 'review' as const, instructions: 'Explain what supports each conclusion.', providerId: provider.id, model: 'fixture', approvalScope: 'review' as const, executionTarget: 'vm' as const, createdAt: 1, updatedAt: 1 }
const store = { settings: { ...defaultSettings(), providers: [provider], agentPresets: [original], approvalRules: [] } as Settings }
const serial = createSerialTaskQueue();let fail = false;let effects = 0
const source = await fs.readFile(new URL('./core-ipc.ts', import.meta.url), 'utf8')
const ast = ts.createSourceFile('core-ipc.ts', source, ts.ScriptTarget.ES2022, true);let declaration: ts.FunctionDeclaration | undefined
function visit(node: ts.Node) { if (ts.isFunctionDeclaration(node) && node.name?.text === 'commitSettingsChange') declaration = node; ts.forEachChild(node, visit) }
visit(ast);assert.ok(declaration)
const compiled = ts.transpileModule(ts.createPrinter().printNode(ts.EmitHint.Unspecified, declaration, ast), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const commit = new Function('store', 'workspaceMutationGuard', 'settingsChanges', 'saveSettings', 'redactSettings', 'auditPreview', `${compiled};return commitSettingsChange;`)(store, new WorkspaceMutationGuard(), serial, async (next: Settings) => { if (fail) throw new Error('Synthetic disk unavailable'); await saveSettings(next) }, redactSettings, (e: unknown) => String(e))
const transaction = (prepare: (current: Settings) => Promise<Settings>) => commit(prepare, async () => { effects++ })
try {
  await saveSettings(store.settings)
  const preview = exportRecipePreview(store.settings, original.id)
  const input = { text: preview.text, target: { providerId: provider.id, providerAccountId: null, model: 'fixture', workspace: 'unassigned' as const }, reviewed: true as const }
  assert.equal(effects, 0);assert.equal(store.settings.agentPresets.length, 1, 'preview has no persistence or run effects')
  const before = store.settings
  fail = true
  await assert.rejects(() => importReviewedRecipe(input, transaction, () => 'failed_bot', () => 2), /disk unavailable/)
  assert.equal(store.settings, before);assert.deepEqual((await loadSettings()).agentPresets.map(bot => bot.id), [original.id]);assert.equal(effects, 0, 'failed persistence does not publish a saved/imported bot')
  fail = false
  await Promise.all([importReviewedRecipe(input, transaction, () => 'new_bot_a', () => 3), importReviewedRecipe(input, transaction, () => 'new_bot_b', () => 4)])
  assert.deepEqual(store.settings.agentPresets.map(bot => bot.id), ['new_bot_b', 'new_bot_a', original.id], 'real settings queue appends against latest durable state without dropping concurrent imports')
  assert.deepEqual((await loadSettings()).agentPresets.map(bot => bot.id), ['new_bot_b', 'new_bot_a', original.id])
  assert.equal(effects, 2)
  for (const bot of store.settings.agentPresets.slice(0,2)) { assert.equal(bot.paused, true);assert.equal(bot.approvalScope, 'readonly');assert.equal(bot.executionTarget, 'workspace');assert.deepEqual((bot as any).toolScope, { mode: 'selected', names: [] }) }
  assert.equal(store.settings.agentPresets[2], original);assert.deepEqual(store.settings.approvalRules, [])
  const count = store.settings.agentPresets.length
  await assert.rejects(() => importReviewedRecipe(input, transaction, () => original.id), /unique/)
  await assert.rejects(() => importReviewedRecipe({ ...input, reviewed: false }, transaction), /Review/)
  await assert.rejects(() => importReviewedRecipe({ ...input, automaticRun: true }, transaction), /Review/)
  assert.equal(store.settings.agentPresets.length, count)
  assert.equal(reviewedRecipeExport(store.settings, { presetId: original.id, reviewedText: preview.text, reviewed: true }).text, preview.text)
  assert.throws(() => reviewedRecipeExport(store.settings, { presetId: original.id, reviewedText: preview.text, reviewed: false }), /Review/)
  store.settings = { ...store.settings, agentPresets: store.settings.agentPresets.map(bot => bot.id === original.id ? { ...bot, instructions: 'New unseen instructions.' } : bot) }
  assert.throws(() => reviewedRecipeExport(store.settings, { presetId: original.id, reviewedText: preview.text, reviewed: true }), /changed after your review/)
  const settingsText = await fs.readFile(path.join(profile, 'settings.json'), 'utf8')
  assert.ok(!settingsText.includes(provider.apiKey), 'real protected settings never write the fictional provider credential in plaintext')
  console.log('Actual settings transaction/queue plus protected save/reload preserves paused imports, prior bots/rules, failed-disk truth and export-review freshness')
} finally { closeSessionSearchIndexes(); await fs.rm(profile, { recursive: true, force: true }) }
