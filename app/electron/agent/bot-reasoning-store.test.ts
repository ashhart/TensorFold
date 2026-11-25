import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { captureBotConfig } from '../../shared/bot-policy'
import type { AgentPreset, Provider, Thread } from './types'

const profile = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-bot-effort-store-'))
const previousHome = process.env.VONTRA_HOME
process.env.VONTRA_HOME = profile
try {
  const { loadSettings, saveSettings, loadThreads, saveThreads } = await import('../store')
  const provider: Provider = { id: 'fixture', name: 'Local fixture', kind: 'local', baseUrl: 'http://127.0.0.1:1/v1', apiKey: '', models: ['fixture'] }
  const preset: AgentPreset = { id: 'bot_effort', name: 'Careful reviewer', role: 'Review supplied text', lane: 'review', providerId: provider.id, model: 'fixture', reasoningEffort: 'high', createdAt: 1, updatedAt: 1 }
  await saveSettings({ ...await loadSettings(), providers: [provider], agentPresets: [preset] })
  const restarted = await loadSettings()
  assert.equal(restarted.agentPresets[0].reasoningEffort, 'high', 'protected settings retain per-bot effort across restart')
  const bot = captureBotConfig(restarted.agentPresets[0], restarted.providers)
  const thread: Thread = { id: 'effort-chat', title: 'Review', kind: 'chat', cwd: '', providerId: provider.id, providerAccountId: null, model: 'fixture', bot, status: 'idle', createdAt: 1, updatedAt: 1, timeline: [], apiMessages: [] }
  await saveThreads([thread])
  await saveSettings({ ...restarted, agentPresets: [{ ...preset, reasoningEffort: 'low' }] })
  assert.equal((await loadSettings()).agentPresets[0].reasoningEffort, 'low')
  assert.equal((await loadThreads())[0].bot?.reasoningEffort, 'high', 'persisted conversations keep their captured effort after a bot edit')
  console.log('Protected bot effort settings and chat restart checks passed.')
} finally {
  if (previousHome === undefined) delete process.env.VONTRA_HOME
  else process.env.VONTRA_HOME = previousHome
  await fs.rm(profile, { recursive: true, force: true })
}
