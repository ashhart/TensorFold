import assert from 'node:assert/strict'
import { captureBotConfig, normalizeAgentPresets } from '../../shared/bot-policy'
import { botRunReasoningEffort } from '../../shared/bot-reasoning'
import { mostRecentCompatibleBotChat } from './bot-chats'
import type { AgentPreset, Provider, Thread } from './types'

const provider: Provider = { id: 'fixture', name: 'Local fixture', kind: 'local', baseUrl: 'http://127.0.0.1:1/v1', apiKey: '', models: ['fixture'] }
const preset: AgentPreset = { id: 'bot_effort', name: 'Careful reviewer', role: 'Review supplied text', lane: 'review', providerId: provider.id, model: 'fixture', createdAt: 1, updatedAt: 1 }

for (const reasoningEffort of ['none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max', 'ultrathink', 'ultracode', 'dynamic', '512'] as const) {
  const configured: AgentPreset = { ...preset, reasoningEffort }
  const saved = normalizeAgentPresets([configured], true)[0]
  assert.equal(saved.reasoningEffort, reasoningEffort, 'saving a bot preserves its selected effort')
  const captured = captureBotConfig(saved, [provider])
  assert.equal(captured.reasoningEffort, reasoningEffort, 'new bot chats capture the saved effort')
  configured.reasoningEffort = 'medium'
  assert.equal(captured.reasoningEffort, reasoningEffort, 'later preset edits leave the captured chat unchanged')
}
assert.equal((normalizeAgentPresets([preset], true)[0] as any).reasoningEffort, undefined, 'legacy bots retain their global-setting fallback')
for (const reasoningEffort of [null, false, 3, {}, '', ' high ', 'x'.repeat(100)]) {
  assert.throws(() => normalizeAgentPresets([{ ...preset, reasoningEffort }], true), /reasoning effort/i)
}
const captured = captureBotConfig({ ...preset, reasoningEffort: 'low' }, [provider])
const thread: Thread = { id: 'bot_chat', title: 'Review', kind: 'chat', cwd: '', providerId: provider.id, providerAccountId: null, model: 'fixture', bot: captured, createdAt: 1, updatedAt: 1, status: 'idle', timeline: [], apiMessages: [] }
assert.equal(botRunReasoningEffort(thread, 'high'), 'low')
assert.equal(botRunReasoningEffort({}, 'high'), 'high')
assert.equal(botRunReasoningEffort({ bot: captureBotConfig(preset, [provider]) }, 'high'), 'high')
assert.equal(mostRecentCompatibleBotChat({ ...preset, reasoningEffort: 'low' }, [provider], [thread])?.id, thread.id)
assert.equal(mostRecentCompatibleBotChat({ ...preset, reasoningEffort: 'high' }, [provider], [thread]), null, 'a new effort setting must not reopen a chat with a different captured effort')
console.log('Bot reasoning persistence and snapshot checks passed.')
