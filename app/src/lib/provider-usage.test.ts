import assert from 'node:assert/strict'
import { formatUsageTokens, modelUsageLabel, providerUsageSummary, usageAccountIdForProvider } from './provider-usage.ts'
import type { ModelRouterSnapshot, Provider, TokenUsageSnapshot } from './types.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('provider-usage tests:')

const codex: Provider = { id: 'codex-cli', name: 'Codex CLI OAuth', transport: 'codex-cli', baseUrl: '', apiKey: '', models: ['gpt-5.5'], spendBudget: { enabled: true, period: 'day', spentUsd: 4, limitUsd: 10 } }
const usage: TokenUsageSnapshot = {
  generatedAt: Date.now(),
  latestAt: Date.now(),
  active: true,
  windows: {
    five_hour: null as any,
    week: null as any,
    month: null as any,
    all: null as any,
    today: {
      key: 'today',
      label: 'Today',
      startAt: Date.now(),
      total: 150_000,
      accounts: [{ id: 'codex', name: 'Codex', latestAt: Date.now(), total: 150_000, input: 80_000, output: 20_000, cacheCreation: 0, cacheRead: 0, cached: 40_000, reasoning: 10_000, models: [] }],
      models: [{ model: 'gpt-5.5', total: 150_000, input: 80_000, output: 20_000, cacheCreation: 0, cacheRead: 0, cached: 40_000, reasoning: 10_000 }],
    },
  },
  daily: [],
  hourly: [],
}
usage.windows.five_hour = usage.windows.today
usage.windows.week = usage.windows.today
usage.windows.month = usage.windows.today
usage.windows.all = usage.windows.today

{
  assert.equal(formatUsageTokens(150_000), '150K')
  assert.equal(formatUsageTokens(1_500_000), '1.5M')
  ok('formatUsageTokens: compacts token counts')
}

{
  assert.equal(usageAccountIdForProvider(codex), 'codex')
  assert.equal(usageAccountIdForProvider({ id: 'opencode', name: 'OpenCode', baseUrl: '', apiKey: '', models: [] }), 'opencode')
  ok('usageAccountIdForProvider: maps known local harness providers')
}

{
  assert.equal(modelUsageLabel('GPT-5.5', usage), '150K today')
  ok('modelUsageLabel: matches models case-insensitively')
}

{
  const route: ModelRouterSnapshot = { generatedAt: Date.now(), activeCooldowns: [{ providerId: 'codex-cli', providerName: 'Codex', providerKind: 'cloud', providerTransport: 'codex-cli', accountId: null, accountName: 'default', accountKind: 'provider-default', model: 'gpt-5.5', kind: 'rate_limit', observedAt: Date.now(), resetAt: Date.now() + 60_000, source: 'test' }], recentlyReset: [], history: [] }
  const summary = providerUsageSummary(codex, null, 'gpt-5.5', usage, route)
  assert.equal(summary.usage, '150K today')
  assert.equal(summary.budget, '$4.00 / $10')
  assert.equal(summary.route, 'Cooling')
  ok('providerUsageSummary: combines usage, budget, and route health')
}

console.log(`\n${passed} passed`)
