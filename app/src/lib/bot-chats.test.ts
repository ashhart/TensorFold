import assert from 'node:assert/strict'
import { captureBotConfig } from '../../shared/bot-policy'
import { mostRecentCompatibleBotChat } from './bot-chats'
import type { AgentPreset, Provider, Thread } from './types'

const provider: Provider = { id: 'fixture', name: 'Local fixture', kind: 'local', transport: 'chat-completions', baseUrl: 'http://127.0.0.1:9090/v1', apiKey: '', models: ['model-a', 'model-b'] }
const preset: AgentPreset = { id: 'bot_fixture', name: 'Reviewer', role: 'Review code', lane: 'review', instructions: 'Explain reproducible defects.', providerId: provider.id, model: 'model-a', createdAt: 1, updatedAt: 1 }
const bot = captureBotConfig(preset, [provider], 10)
const thread: Thread = { id: 'chat-a', title: 'Reviewer chat', kind: 'chat', cwd: '', providerId: provider.id, providerAccountId: null, model: 'model-a', createdAt: 1, updatedAt: 10, status: 'idle', bot, apiMessages: [{ role: 'user', content: 'Synthetic previous request.' }], timeline: [] }
assert.equal(mostRecentCompatibleBotChat(preset, [provider], [thread]), thread)
const newer = { ...thread, id: 'chat-b', updatedAt: 20, bot: { ...bot, capturedAt: 15 } }
assert.equal(mostRecentCompatibleBotChat(preset, [provider], [thread, newer]), newer)
assert.equal(mostRecentCompatibleBotChat({ ...preset, paused: true }, [provider], [thread]), thread, 'a paused chat can still be opened and read')
assert.equal(mostRecentCompatibleBotChat({ ...preset, archivedAt: 123 }, [provider], [thread]), thread, 'archiving a bot retains access to its conversation')
assert.equal(mostRecentCompatibleBotChat(preset, [provider], [{ ...thread, archivedAt: 123 }]), null)
for (const patch of [{ name: 'Different identity' }, { role: 'Different responsibility' }, { instructions: 'Different instructions' }, { model: 'model-b' }, { approvalScope: 'review' as const }, { executionTarget: 'vm' as const }]) assert.equal(mostRecentCompatibleBotChat({ ...preset, ...patch }, [provider], [thread]), null, 'changed bot behavior starts a new conversation instead of retargeting existing history')
for (const patch of [{ model: 'model-b' }, { providerId: 'other' }, { providerAccountId: 'other' }, { cwd: '/workspace' }, { projectId: 'project' }, { workspace: { kind: 'local', path: '/tmp' } }, { orchestratorHostId: 'remote' }, { engine: 'omp' }]) assert.equal(mostRecentCompatibleBotChat(preset, [provider], [{ ...thread, ...patch } as Thread]), null)
assert.equal(mostRecentCompatibleBotChat(preset, [], [thread]), null)
assert.equal(mostRecentCompatibleBotChat(preset, [{ ...provider, models: ['model-b'] }], [thread]), null)
assert.equal(thread.bot, bot)
assert.equal(thread.model, 'model-a', 'matching never rewrites a saved chat')
console.log('Bot conversation reuse checks passed: newest compatible history, paused/archive readability, exact scope/model/configuration and no retargeting.')
