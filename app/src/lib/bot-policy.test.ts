import assert from 'node:assert/strict'
import { assertBotModel, botInstructions, botProviderSupported, botRunApprovalMode, captureBotConfig, normalizeAgentPresets } from '../../shared/bot-policy'
import type { AgentPreset, Provider } from './types'

const provider: Provider = { id: 'local', name: 'Local test model', transport: 'chat-completions', kind: 'local', models: ['model-a', 'model-b'], baseUrl: 'http://127.0.0.1:9090/v1', apiKey: '' }
const preset: AgentPreset = { id: 'bot_fixture', name: 'Review helper', role: 'Review supplied code', lane: 'review', providerId: provider.id, model: 'model-a', instructions: 'Describe reproducible defects.', createdAt: 1, updatedAt: 1 }
const bot = captureBotConfig(preset, [provider], 123)
assert.equal(bot.approvalScope, 'readonly')
assert.equal(bot.executionTarget, 'workspace')
assert.equal(bot.capturedAt, 123)
assert.equal(bot.providerAccountId, null)
assert.equal(botRunApprovalMode({ bot }, [preset], [provider], 'auto'), 'readonly', 'global auto mode cannot elevate a read-only bot')
assert.equal(botRunApprovalMode({ bot: { ...bot, approvalScope: 'review' } }, [preset], [provider], 'auto'), 'suggest')
assert.equal(botRunApprovalMode({}, [preset], [provider], 'auto'), 'auto', 'ordinary chats keep their existing policy')
assert.equal(botRunApprovalMode({ bot }, [{ ...preset, instructions: 'Different instructions.', model: 'model-b', approvalScope: 'review' }], [provider], 'auto'), 'readonly')
assert.match(botInstructions(bot), /Describe reproducible defects/)
assert.match(botInstructions(bot), /Review supplied code/)
assert.equal(bot.model, 'model-a', 'changing a preset does not rewrite captured conversations')
for (const current of [[], [{ ...preset, paused: true }], [{ ...preset, archivedAt: 456 }]]) assert.throws(() => botRunApprovalMode({ bot }, current, [provider], 'auto'), /archived|paused/)
assert.throws(() => captureBotConfig({ ...preset, paused: true }, [provider]), /Resume/)
assert.throws(() => captureBotConfig({ ...preset, archivedAt: 100 }, [provider]), /Restore/)
assert.throws(() => captureBotConfig({ ...preset, name: ' ' }, [provider]), /name/)
assert.throws(() => assertBotModel(bot, []), /missing/)
assert.throws(() => assertBotModel(bot, [{ ...provider, models: ['model-b'] }]), /unavailable/)
assert.throws(() => captureBotConfig(preset, [{ ...provider, transport: 'codex-cli' }]), /API model/)
assert.throws(() => captureBotConfig({ ...preset, executionTarget: 'vm' }, [{ ...provider, kind: 'cloud', baseUrl: 'https://model.example.test/v1' }]), /model running on this Mac/)
assert.throws(() => captureBotConfig({ ...preset, executionTarget: 'vm' }, [{ ...provider, kind: 'local', baseUrl: 'http://192.0.2.20:9000/v1' }]), /model running on this Mac/)
assert.equal(botProviderSupported(provider), true)
assert.equal(botProviderSupported({ ...provider, transport: undefined }), true)
for (const transport of ['claude-cli', 'codex-cli', 'opencode', 'kilo', 'pi-cli', 'cursor-cli', 'gemini-cli'] as const) assert.equal(botProviderSupported({ ...provider, transport }), false)
assert.throws(() => captureBotConfig({ ...preset, providerAccountId: 'missing' }, [provider]), /account is unavailable/)
const account = { id: 'account-fixture', name: 'Fixture', kind: 'api-key' as const, models: ['account-model'], createdAt: 1, updatedAt: 1 }
const accountProvider = { ...provider, accounts: [account] } as Provider
assert.equal(captureBotConfig({ ...preset, providerAccountId: account.id, model: 'account-model' }, [accountProvider]).providerAccountId, account.id)
assert.throws(() => captureBotConfig({ ...preset, providerAccountId: account.id, model: 'model-a' }, [accountProvider]), /saved model is unavailable/)
assert.throws(() => captureBotConfig({ ...preset, providerAccountId: account.id, model: 'account-model' }, [{ ...accountProvider, accounts: [{ ...account, enabled: false }] } as Provider]), /account is unavailable/)
const legacy = normalizeAgentPresets([preset])[0]
assert.equal(legacy.approvalScope, 'readonly')
assert.equal(legacy.executionTarget, 'workspace')
assert.equal(legacy.instructions, preset.instructions)
assert.deepEqual(normalizeAgentPresets(null), [])
assert.throws(() => normalizeAgentPresets(null, true), /list/)
assert.equal(normalizeAgentPresets([preset, preset]).length, 1)
assert.throws(() => normalizeAgentPresets([preset, preset], true), /Invalid/)
for (const patch of [{ id: '../escape' }, { executionTarget: 'host-fallback' }, { approvalScope: 'auto' }, { paused: 'false' }, { instructions: {} }, { archivedAt: -1 }, { lane: 'invalid' }]) {
  assert.equal(normalizeAgentPresets([{ ...preset, ...patch }]).length, 0)
  assert.throws(() => normalizeAgentPresets([{ ...preset, ...patch }], true), /Invalid/)
}
assert.equal(normalizeAgentPresets([{ ...preset, name: 'x'.repeat(100), instructions: 'x'.repeat(20_000) }], true)[0].instructions?.length, 16_000)
for (const patch of [{ executionTarget: 'invalid' }, { approvalScope: 'auto' }, { instructions: {} }, { providerAccountId: undefined }, { presetId: '../escape' }]) assert.throws(() => botRunApprovalMode({ bot: { ...bot, ...patch } as any }, [preset], [provider], 'auto'), /invalid/)
console.log('Bot policy checks passed: legacy migration, captured scope, model/account pins, pause/archive, malformed configuration and CLI isolation.')
