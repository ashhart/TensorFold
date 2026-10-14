import assert from 'node:assert/strict'
import { BOT_RECIPE_MAX_BYTES, importedRecipeBot, parseBotRecipe, previewBotRecipe, recipeForBot, reviewedRecipeImport, serializeBotRecipe } from '../../shared/bot-recipe'
import { captureBotConfig } from '../../shared/bot-policy'
import type { AgentPreset, Provider } from './types'

const provider: Provider = { id: 'fixture-local', name: 'Local fixture', kind: 'local', baseUrl: 'http://127.0.0.1:9090/v1', apiKey: 'synthetic-provider-secret', models: ['fixture-model'], accounts: [{ id: 'fixture-account', name: 'Fixture account', kind: 'api-key', enabled: true, models: ['account-model'], apiKey: 'synthetic-account-secret', createdAt: 1, updatedAt: 1 }] }
const preset: AgentPreset = { id: 'private-preset-id', name: 'Research partner', role: 'Check evidence', lane: 'research', instructions: 'Cite sources and describe uncertainty.', providerId: provider.id, providerAccountId: 'fixture-account', model: 'account-model', approvalScope: 'review', executionTarget: 'vm', paused: false, createdAt: 12, updatedAt: 34 }
const preview = recipeForBot({ ...preset, vmId: 'private-vm-id', cwd: 'private-workspace', toolScope: { mode: 'selected', names: ['write_file'] }, token: provider.apiKey, transcript: 'private-history' } as any)
assert.deepEqual(Object.keys(preview.recipe), ['format', 'version', 'bot'])
assert.deepEqual(Object.keys(preview.recipe.bot), ['name', 'role', 'instructions', 'lane'])
for (const privateValue of [preset.id, provider.id, 'fixture-account', 'account-model', 'private-vm-id', 'private-workspace', provider.apiKey, 'private-history', 'write_file']) assert.ok(!preview.text.includes(privateValue))
assert.deepEqual(parseBotRecipe(preview.text), preview.recipe)
assert.equal(serializeBotRecipe(parseBotRecipe(preview.text)), preview.text)
const mutate = (patch: Record<string, unknown>) => JSON.stringify({ ...preview.recipe, ...patch })
for (const version of [0, 2, '1', null]) assert.throws(() => parseBotRecipe(mutate({ version })), /format or version/)
for (const patch of [{ token: 'extra' }, { paths: ['..'] }, { vmId: 'extra' }, { __proto__: null, authority: 'extra' }]) assert.throws(() => parseBotRecipe(mutate(patch)), /fields/)
for (const patch of [{ name: '../escape' }, { name: 'folder\\escape' }, { name: '..' }, { name: '' }, { name: 'x'.repeat(81) }, { instructions: 'x'.repeat(16_001) }, { role: 'x'.repeat(241) }, { lane: 'unknown' }, { permissions: 'auto' }, { toolScope: { mode: 'selected', names: ['write_file'] } }]) assert.throws(() => parseBotRecipe(mutate({ bot: { ...preview.recipe.bot, ...patch } })))
assert.throws(() => parseBotRecipe('{"format":"tensorfold.bot-recipe","version":1,"bot":{"name":"Example","role":"","instructions":"","lane":"review","__proto__":{}}}'), /fields/)
assert.throws(() => parseBotRecipe('x'.repeat(BOT_RECIPE_MAX_BYTES + 1)), /64 KiB/)
assert.throws(() => parseBotRecipe('💡'.repeat(20_000)), /64 KiB/, 'bound bytes before parsing, not just JS characters')
assert.throws(() => parseBotRecipe('{broken'), /not valid JSON/)
assert.throws(() => parseBotRecipe({} as any), /JSON text/)
const homePath = ['','Users','fictional','Private','context.txt'].join('/')
const credential = ['sk-', 'test-', 'q'.repeat(40)].join('')
for (const instructions of [homePath, ['','Users','fictional'].join('/'), credential, ['api','Key'].join('') + '=fictional-value-for-recipe-tests', 'Authorization: Bearer fictional-bearer-example-427', ['-----BEGIN ','PRIVATE KEY-----'].join('')]) {
  assert.throws(() => recipeForBot({ ...preset, instructions }), (error: any) => { assert.ok(!error.message.includes(instructions)); return /common credential|home-folder/.test(error.message) })
}
assert.equal(previewBotRecipe(mutate({ bot: { ...preview.recipe.bot, instructions: 'Contact fictional@example.test about the example.' } })).recipe.bot.instructions, 'Contact fictional@example.test about the example.', 'explicit human review remains necessary; no universal PII detector is claimed')
const target = { providerId: provider.id, providerAccountId: 'fixture-account', model: 'account-model', workspace: 'unassigned' as const }
const imported = importedRecipeBot(preview.recipe, target, [provider], 'new_bot', 100)
assert.equal(imported.paused, true);assert.equal(imported.approvalScope, 'readonly');assert.equal(imported.executionTarget, 'workspace');assert.equal(imported.archivedAt, null)
assert.deepEqual(imported.toolScope, { mode: 'selected', names: [] })
assert.notEqual(imported.id, preset.id);assert.equal(imported.createdAt, 100)
assert.throws(() => captureBotConfig(imported, [provider]), /Resume/, 'imported bot cannot start until a separate user action')
assert.throws(() => reviewedRecipeImport({ text: preview.text, target, reviewed: false }), /Review/)
for (const bad of [{ ...target, cwd: homePath }, { ...target, workspace: '../escape' }, { ...target, providerAccountId: 'missing' }, { ...target, model: 'missing' }, { ...target, providerId: 'missing' }]) assert.throws(() => importedRecipeBot(preview.recipe, bad as any, [provider], 'new_bot', 100))
assert.throws(() => importedRecipeBot(preview.recipe, target, [{ ...provider, baseUrl: 'https://model.example.test/v1' }], 'new_bot', 100), /on this Mac/)
assert.throws(() => importedRecipeBot(preview.recipe, target, [{ ...provider, accounts: [{ ...provider.accounts![0], enabled: false }] }], 'new_bot', 100), /account/)
assert.equal(preset.approvalScope, 'review');assert.equal(preset.executionTarget, 'vm');assert.equal(preset.paused, false)
Object.defineProperty(globalThis, 'window', { value: { dispatchEvent: () => true }, configurable: true })
Object.defineProperty(globalThis, 'localStorage', { value: { getItem: () => null, setItem: () => {}, removeItem: () => {} }, configurable: true })
const { createRemoteBrowserApi } = await import('./remote-transport')
let requests = 0
const remote = createRemoteBrowserApi({ token: 'fictional-device-token', fetchImpl: (async () => { requests++;throw new Error('No request expected') }) as typeof fetch, openEventStream: () => ({ onEvent: () => {}, close: () => {} }) })
for (const action of [() => remote.botRecipePreview({ kind: 'import', value: preview.text }), () => remote.botRecipeImport({ text: preview.text, target, reviewed: true }), () => remote.botRecipeExport({ presetId: preset.id, reviewedText: preview.text, reviewed: true })]) await assert.rejects(action, /paired session/)
assert.equal(requests, 0, 'paired clients neither retain a browser-demo import nor transmit recipe text to the host')
console.log('Portable recipe whitelist, byte/version/traversal limits, explicit review, common-sensitive-text guard, local bindings and paused no-authority import checks passed')
