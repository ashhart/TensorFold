import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import ts from 'typescript'
import { previewBotRecipe, recipeObject } from '../../../../shared/bot-recipe'
import { exportRecipePreview, importReviewedRecipe, reviewedRecipeExport } from '../bot-recipes'
import type { Settings } from '../../../agent/types'

const temp = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-recipe-ipc-'))
const provider = { id: 'fixture-local', name: 'Local', kind: 'local' as const, baseUrl: 'http://127.0.0.1:9090/v1', apiKey: '', models: ['fixture'] }
const original = { id: 'original_bot', name: 'Evidence helper', role: 'Review evidence', instructions: 'Describe uncertainty.', lane: 'research' as const, providerId: provider.id, model: 'fixture', approvalScope: 'review' as const, executionTarget: 'vm' as const, createdAt: 1, updatedAt: 1 }
let settings = { providers: [provider], agentPresets: [original] } as Settings
let denied = false;let dialogs = 0;let commits = 0;let mode: 'cancel' | 'save' | 'stale' = 'save';let destination = path.join(temp, 'reviewed.json')
const handlers = new Map<string, (event: any, input: any) => Promise<any>>()
const dependencies = {
  BrowserWindow: { fromWebContents: () => null },
  dialog: { showSaveDialog: async () => { dialogs++; if (mode === 'stale') settings = { ...settings, agentPresets: settings.agentPresets.map(bot => ({ ...bot, instructions: 'Unseen change.' })) }; return mode === 'cancel' ? { canceled: true } : { canceled: false, filePath: destination } } },
  fs, previewBotRecipe, recipeObject, exportRecipePreview, importReviewedRecipe, reviewedRecipeExport,
  handle: (channel: string, callback: any) => handlers.set(channel, callback),
  requireSecurityRole: (role: string) => { assert.equal(role, 'owner'); if (denied) throw new Error('Synthetic role denied') },
}
const text = await fs.readFile(new URL('./bot-recipe-ipc.ts', import.meta.url), 'utf8')
const ast = ts.createSourceFile('bot-recipe-ipc.ts', text, ts.ScriptTarget.ES2022, true)
const node = ast.statements.find((item): item is ts.FunctionDeclaration => ts.isFunctionDeclaration(item) && item.name?.text === 'registerBotRecipeIpc');assert.ok(node)
const source = ts.createPrinter().printNode(ts.EmitHint.Unspecified, node, ast).replace(/^export\s+/, '')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const register = new Function(...Object.keys(dependencies), `${compiled};return registerBotRecipeIpc;`)(...Object.values(dependencies))
register({ currentSettings: () => settings, commitImport: async (prepare: any) => { commits++;settings = await prepare(settings);return settings } })
const event = { sender: {} };const call = (channel: string, value: unknown) => handlers.get('botRecipe:'+channel)!(event,value)
try {
  const preview = await call('preview', { kind: 'export', value: original.id });assert.equal(dialogs, 0);assert.equal(commits, 0)
  const reviewed = { presetId: original.id, reviewedText: preview.text, reviewed: true }
  await assert.rejects(() => call('export', { ...reviewed, reviewed: false }), /Review/);assert.equal(dialogs, 0)
  denied = true;await assert.rejects(() => call('preview', { kind: 'import', value: preview.text }), /role denied/);denied = false
  mode = 'cancel';assert.deepEqual(await call('export', reviewed), { saved: false });assert.equal(await fs.stat(destination).then(() => true, () => false), false)
  mode = 'stale';await assert.rejects(() => call('export', reviewed), /changed after your review/);assert.equal(await fs.stat(destination).then(() => true, () => false), false)
  settings = { ...settings, agentPresets: [original] };mode = 'save'
  assert.deepEqual(await call('export', reviewed), { saved: true });assert.equal(await fs.readFile(destination, 'utf8'), preview.text)
  assert.equal((await fs.stat(destination)).mode & 0o777, 0o600)
  await assert.rejects(() => call('export', reviewed), /existing file was not changed/);assert.equal(await fs.readFile(destination, 'utf8'), preview.text)
  destination = path.join(temp, 'missing-directory', 'recipe.json')
  await assert.rejects(() => call('export', reviewed), (error: any) => { assert.ok(!error.message.includes(temp));return /writable location/.test(error.message) })
  const target = { providerId: provider.id, providerAccountId: null, model: 'fixture', workspace: 'unassigned' }
  await assert.rejects(() => call('import', { text: preview.text, target, reviewed: false }), /Review/);assert.equal(commits, 0)
  const saved = await call('import', { text: preview.text, target, reviewed: true });assert.equal(commits, 1)
  assert.equal(saved.agentPresets.length, 2);assert.notEqual(saved.agentPresets[0].id, original.id);assert.equal(saved.agentPresets[0].paused, true)
  assert.equal(saved.agentPresets[0].approvalScope, 'readonly');assert.equal(saved.agentPresets[0].executionTarget, 'workspace')
  assert.equal(saved.agentPresets[1], original)
  console.log('Actual recipe IPC handler AST enforces owner/review, cancellation/freshness, exclusive private-file export, generic I/O errors and paused additive import')
} finally { await fs.rm(temp, { recursive: true, force: true }) }
