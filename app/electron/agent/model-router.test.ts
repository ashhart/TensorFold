// Run: npx tsx electron/agent/model-router.test.ts
import assert from 'node:assert'
import {
  classifyModelExhaustion,
  ModelRouter,
  recordModelExhaustion,
  routeAfterModelError,
  selectModelRoute,
  type ModelRouteState,
} from './model-router.ts'
import type { Provider } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ok ${name}`) }

const now = Date.UTC(2026, 5, 19, 12, 0, 0)

const provider: Provider = {
  id: 'codex-cli',
  name: 'Codex CLI OAuth',
  transport: 'codex-cli',
  auth: 'codex-oauth',
  baseUrl: '',
  apiKey: 'provider-secret',
  models: ['gpt-5.5', 'o4-mini'],
  defaultAccountId: 'personal',
  accounts: [
    { id: 'personal', name: 'Personal', kind: 'subscription', enabled: true, autoFailover: true, models: ['gpt-5.5', 'o4-mini'], apiKey: 'secret-a', createdAt: 1, updatedAt: 1 },
    { id: 'work', name: 'Work', kind: 'subscription', enabled: true, models: ['gpt-5.5', 'o4-mini'], apiKey: 'secret-b', createdAt: 1, updatedAt: 1 },
    { id: 'review', name: 'Review', kind: 'subscription', enabled: true, models: ['o4-mini'], apiKey: 'secret-c', createdAt: 1, updatedAt: 1 },
    { id: 'disabled', name: 'Disabled', kind: 'subscription', enabled: false, models: ['gpt-5.5'], apiKey: 'secret-d', createdAt: 1, updatedAt: 1 },
  ],
}

const mixedProvider: Provider = {
  ...provider,
  id: 'mixed',
  accounts: [
    { id: 'personal', name: 'Personal', kind: 'subscription', enabled: true, autoFailover: true, models: ['gpt-5.5'], apiKey: 'secret-a', createdAt: 1, updatedAt: 1 },
    { id: 'team-api', name: 'Team API', kind: 'api-key', enabled: true, models: ['gpt-5.5'], apiKey: 'secret-api', createdAt: 1, updatedAt: 1 },
    { id: 'work', name: 'Work', kind: 'subscription', enabled: true, models: ['gpt-5.5'], apiKey: 'secret-b', createdAt: 1, updatedAt: 1 },
  ],
}

console.log('model router tests:')

{
  const signal = classifyModelExhaustion({
    status: 429,
    message: 'Rate limit reached for this subscription.',
    headers: { 'retry-after': '30' },
  }, now)
  assert.strictEqual(signal?.kind, 'rate_limit')
  assert.strictEqual(signal?.resetAt, now + 30_000)
  assert.strictEqual(signal?.source, 'retry-after')
  ok('classifies rate-limit exhaustion with retry-after reset window')
}

{
  const signal = classifyModelExhaustion({
    status: 429,
    message: 'Rate limit reached.',
    headers: { 'RateLimit-Reset': '30' },
  }, now)
  assert.strictEqual(signal?.resetAt, now + 30_000)
  assert.strictEqual(signal?.retryAfterMs, 30_000)
  assert.strictEqual(signal?.source, 'ratelimit-reset')
  ok('interprets standard RateLimit-Reset as delta seconds')
}

{
  const signal = classifyModelExhaustion(new Error('This model maximum context length is 128000 tokens'), now)
  assert.strictEqual(signal, null)
  ok('does not treat context-size failures as account exhaustion')
}

{
  const result = recordModelExhaustion(undefined, {
    provider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: new Error('HTTP 429 Too Many Requests - quota exhausted'),
    now,
    defaultCooldownMs: 90_000,
  })
  assert.strictEqual(result.signal?.kind, 'quota_exhausted')
  assert.strictEqual(result.state.exhausted.length, 1)
  assert.strictEqual(result.state.exhausted[0].accountId, 'personal')
  assert.strictEqual(result.state.exhausted[0].resetAt, now + 90_000)
  assert.deepStrictEqual(result.audit.exhaustedAccountIds, ['personal'])
  ok('records exhaustion against provider account and model')
}

{
  const state: ModelRouteState = {
    exhausted: [{
      providerId: provider.id,
      accountId: 'personal',
      model: 'gpt-5.5',
      kind: 'rate_limit',
      observedAt: now,
      resetAt: now + 60_000,
      message: 'rate limit',
      source: 'test',
    }],
  }
  const route = selectModelRoute({ provider, model: 'gpt-5.5', requestedAccountId: 'personal', state, now })
  assert.strictEqual(route.audit.action, 'failover')
  assert.strictEqual(route.selectedAccountId, 'work')
  assert.deepStrictEqual(route.audit.exhaustedAccountIds, ['personal'])
  assert.deepStrictEqual(route.audit.eligibleAccountIds, ['work'])
  assert.deepStrictEqual(route.failoverChain.map((candidate) => candidate.accountId), ['work'])
  assert.deepStrictEqual(route.audit.failoverChainAccountIds, ['work'])
  assert.ok(!route.candidates.some((candidate) => candidate.accountId === 'review'))
  assert.ok(!route.candidates.some((candidate) => candidate.accountId === 'disabled'))
  ok('fails over to an equivalent enabled account for the same model')
}

{
  const route = selectModelRoute({ provider, model: 'gpt-5.5', requestedAccountId: 'personal', state: { exhausted: [] }, now })
  assert.strictEqual(route.audit.action, 'select_route')
  assert.strictEqual(route.selectedAccountId, 'personal')
  assert.deepStrictEqual(route.failoverChain.map((candidate) => candidate.accountId), ['personal', 'work'])
  assert.ok(route.failoverChain.every((candidate) => candidate.model === 'gpt-5.5' && candidate.accountKind === 'subscription'))
  ok('builds an ordered failover chain inside the same model and account capability class')
}

{
  const capabilityProvider: Provider = {
    ...provider,
    id: 'capability-boundary',
    accounts: [
      { id: 'personal', name: 'Personal', kind: 'subscription', enabled: true, autoFailover: true, models: ['gpt-5.5'], apiKey: 'secret-a', createdAt: 1, updatedAt: 1 },
      { id: 'same-model', name: 'Same Model', kind: 'subscription', enabled: true, models: ['gpt-5.5'], apiKey: 'secret-b', createdAt: 1, updatedAt: 1 },
      { id: 'different-model', name: 'Different Model', kind: 'subscription', enabled: true, models: ['o4-mini'], apiKey: 'secret-c', createdAt: 1, updatedAt: 1 },
      { id: 'different-kind', name: 'Different Kind', kind: 'api-key', enabled: true, models: ['gpt-5.5'], apiKey: 'secret-d', createdAt: 1, updatedAt: 1 },
    ],
  }
  const route = selectModelRoute({ provider: capabilityProvider, model: 'gpt-5.5', requestedAccountId: 'personal', state: { exhausted: [] }, now })
  assert.deepStrictEqual(route.failoverChain.map((candidate) => candidate.accountId), ['personal', 'same-model'])
  assert.ok(route.failoverChain.every((candidate) =>
    candidate.providerId === capabilityProvider.id &&
    candidate.model === 'gpt-5.5' &&
    candidate.accountKind === 'subscription'
  ))
  ok('keeps failover inside one provider, exact model, and account capability class')
}

{
  const state: ModelRouteState = {
    exhausted: [{
      providerId: provider.id,
      accountId: 'personal',
      model: 'gpt-5.5',
      kind: 'rate_limit',
      observedAt: now,
      resetAt: now + 60_000,
      message: 'rate limit',
      source: 'test',
    }],
  }
  const route = selectModelRoute({ provider, model: 'gpt-5.5', requestedAccountId: 'personal', state, allowFailover: false, now })
  assert.strictEqual(route.audit.action, 'no_route')
  assert.strictEqual(route.selected, null)
  assert.match(route.audit.reason, /Automatic failover is disabled/)
  ok('blocks automatic failover when the account setting is off')
}

{
  const state: ModelRouteState = {
    exhausted: [{
      providerId: mixedProvider.id,
      accountId: 'personal',
      model: 'gpt-5.5',
      kind: 'rate_limit',
      observedAt: now,
      resetAt: now + 60_000,
      message: 'rate limit',
      source: 'test',
    }],
  }
  const route = selectModelRoute({ provider: mixedProvider, model: 'gpt-5.5', requestedAccountId: 'personal', state, now })
  assert.strictEqual(route.audit.action, 'failover')
  assert.strictEqual(route.selectedAccountId, 'work')
  assert.deepStrictEqual(route.failoverChain.map((candidate) => candidate.accountId), ['work'])
  assert.ok(!route.candidates.some((candidate) => candidate.accountId === 'team-api'))
  ok('does not fail over from a subscription into an API-key account')
}

{
  const onlyApiAlternative: Provider = {
    ...mixedProvider,
    accounts: mixedProvider.accounts?.filter((account) => account.id !== 'work'),
  }
  const state: ModelRouteState = {
    exhausted: [{
      providerId: onlyApiAlternative.id,
      accountId: 'personal',
      model: 'gpt-5.5',
      kind: 'rate_limit',
      observedAt: now,
      resetAt: now + 60_000,
      message: 'rate limit',
      source: 'test',
    }],
  }
  const route = selectModelRoute({ provider: onlyApiAlternative, model: 'gpt-5.5', requestedAccountId: 'personal', state, now })
  assert.strictEqual(route.audit.action, 'no_route')
  assert.strictEqual(route.selected, null)
  assert.deepStrictEqual(route.audit.eligibleAccountIds, [])
  ok('blocks failover when the only matching-model account has a different kind')
}

{
  const state: ModelRouteState = {
    exhausted: [{
      providerId: mixedProvider.id,
      accountId: null,
      model: 'gpt-5.5',
      kind: 'rate_limit',
      observedAt: now,
      resetAt: now + 60_000,
      message: 'rate limit',
      source: 'test',
    }],
  }
  const route = selectModelRoute({ provider: mixedProvider, model: 'gpt-5.5', requestedAccountId: null, state, now })
  assert.strictEqual(route.audit.action, 'no_route')
  assert.ok(!route.candidates.some((candidate) => candidate.accountId === 'personal'))
  ok('does not fail over provider-default runs into named accounts')
}

{
  const route = selectModelRoute({
    provider: { ...provider, kind: 'cloud' },
    model: 'gpt-5.5',
    requestedAccountId: 'personal',
    state: { exhausted: [] },
    localOnly: true,
    now,
  })
  assert.strictEqual(route.audit.action, 'no_route')
  assert.strictEqual(route.selected, null)
  assert.deepStrictEqual(route.failoverChain, [])
  assert.match(route.audit.reason, /local-only/)
  ok('blocks cloud provider routes for local-only projects')
}

{
  const state: ModelRouteState = {
    exhausted: [{
      providerId: provider.id,
      accountId: 'personal',
      model: 'gpt-5.5',
      kind: 'rate_limit',
      observedAt: now,
      resetAt: now + 60_000,
      message: 'rate limit',
      source: 'test',
    }],
  }
  const route = selectModelRoute({ provider, model: 'gpt-5.5', requestedAccountId: 'personal', state, now: now + 60_001 })
  assert.strictEqual(route.audit.action, 'select_route')
  assert.strictEqual(route.selectedAccountId, 'personal')
  assert.deepStrictEqual(route.audit.resetEligibleAccountIds, ['personal'])
  ok('returns preferred account after the reset window passes')
}

{
  const routed = routeAfterModelError(undefined, {
    provider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: { status: 429, message: 'too many requests', headers: { 'retry-after': '15' } },
    now,
  })
  assert.strictEqual(routed.signal?.kind, 'rate_limit')
  assert.strictEqual(routed.route.selectedAccountId, 'work')
  assert.strictEqual(routed.audit[0].action, 'record_exhaustion')
  assert.strictEqual(routed.audit[1].action, 'failover')
  ok('records exhaustion and returns the next route in one pure step')
}

{
  let state: ModelRouteState = { exhausted: [] }
  const router = new ModelRouter({
    getState: () => state,
    setState: (next) => { state = next },
  })
  const firstOutput = router.route({
    role: 'chat',
    provider,
    model: 'gpt-5.5',
    requestedAccountId: 'personal',
    now,
    contextLimit: 196_000,
    reasoningEffort: 'high',
    cacheStrategy: 'stable_prefix',
    toolNeeds: ['read_file', 'apply_patch'],
    contextPressure: 0.72,
    budget: 'mission',
    providerHealth: 'healthy',
    accountQuota: 'available',
    scorecard: 0.91,
    latencyMs: 120,
    costEstimate: 0.42,
    userPreference: 'prefer personal account',
    source: 'unit-test',
  })
  const first = firstOutput.decision
  assert.strictEqual(first.selectedAccountId, 'personal')
  assert.strictEqual(firstOutput.contextLimit, 196_000)
  assert.strictEqual(firstOutput.reasoningEffort, 'high')
  assert.strictEqual(firstOutput.cacheStrategy, 'stable_prefix')
  assert.deepStrictEqual(firstOutput.fallbackOrder, ['personal', 'work'])
  assert.deepStrictEqual(firstOutput.inputs.toolNeeds, ['read_file', 'apply_patch'])
  assert.strictEqual(firstOutput.inputs.contextPressure, 0.72)
  assert.strictEqual(firstOutput.inputs.budget, 'mission')
  assert.strictEqual(firstOutput.inputs.providerHealth, 'healthy')
  assert.strictEqual(firstOutput.inputs.accountQuota, 'available')
  assert.strictEqual(firstOutput.inputs.scorecard, 0.91)
  assert.strictEqual(firstOutput.inputs.latencyMs, 120)
  assert.strictEqual(firstOutput.inputs.costEstimate, 0.42)
  assert.strictEqual(firstOutput.inputs.userPreference, 'prefer personal account')
  assert.strictEqual(firstOutput.inputs.contextLimit, 196_000)
  assert.strictEqual(firstOutput.inputs.source, 'unit-test')
  const recovered = router.routeAfterError({
    role: 'chat',
    provider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: { status: 429, message: 'too many requests', headers: { 'retry-after': '15' } },
    contextLimit: 196_000,
    reasoningEffort: 'high',
    now,
  })
  assert.strictEqual(state.exhausted.length, 1)
  assert.strictEqual(recovered.route.selectedAccountId, 'work')
  assert.strictEqual(recovered.output.role, 'chat')
  assert.deepStrictEqual(recovered.output.fallbackOrder, ['work'])
  assert.strictEqual(recovered.output.contextLimit, 196_000)
  assert.strictEqual(recovered.output.reasoningEffort, 'high')
  ok('ModelRouter service stores exhaustion and returns structured route output')
}

{
  const router = new ModelRouter({
    getState: () => ({ exhausted: [] }),
    setState: () => {},
  })
  const localOutput = router.route({
    role: 'ticket_review',
    provider: { ...provider, id: 'local-router', name: 'Local Router', kind: 'local', baseUrl: 'http://127.0.0.1:1234/v1', models: ['qwen-32k'] },
    model: 'qwen-32k',
    requestedAccountId: null,
    contextPressure: 0.91,
    now,
  })
  assert.strictEqual(localOutput.contextLimit, 24_000)
  assert.strictEqual(localOutput.reasoningEffort, 'medium')
  assert.strictEqual(localOutput.cacheStrategy, 'summarize_next')
  ok('ModelRouter output infers local context, role effort, and cache strategy')
}

{
  const router = new ModelRouter({
    getState: () => ({ exhausted: [] }),
    setState: () => {},
  })
  const budgetProvider: Provider = {
    ...provider,
    accounts: provider.accounts?.map((account) => account.id === 'personal'
      ? { ...account, spendBudget: { enabled: true, period: 'month', limitUsd: 10, spentUsd: 9.8, hardStop: true } }
      : account.id === 'work'
        ? { ...account, spendBudget: { enabled: true, period: 'month', limitUsd: 10, spentUsd: 2, hardStop: true } }
        : account
    ),
  }
  const route = router.route({ role: 'chat', provider: budgetProvider, model: 'gpt-5.5', requestedAccountId: 'personal', costEstimate: 0.5, now })
  assert.strictEqual(route.decision.selectedAccountId, 'work')
  assert.match(route.decision.audit.reason, /over budget/)
  assert.strictEqual(route.spend.account?.blocked, false)
  assert.strictEqual(route.spend.account?.remainingUsd, 7.5)
  ok('ModelRouter skips an over-budget account and selects the next eligible account')
}

{
  const router = new ModelRouter({
    getState: () => ({ exhausted: [] }),
    setState: () => {},
  })
  const budgetProvider: Provider = {
    ...provider,
    spendBudget: { enabled: true, period: 'month', limitUsd: 10, spentUsd: 10, hardStop: true },
  }
  const route = router.route({ role: 'chat', provider: budgetProvider, model: 'gpt-5.5', requestedAccountId: 'personal', costEstimate: 0.1, now })
  assert.strictEqual(route.decision.selected, null)
  assert.strictEqual(route.spend.provider?.blocked, true)
  assert.match(route.decision.audit.reason, /hard spend budget/)
  ok('ModelRouter blocks provider routes that exceed a hard spend budget')

  // select() is the entry point solo thread runs use for their initial route
  // (thread-runtime startThreadRun); the hard stop must refuse THERE, with the reason the
  // renderer will surface, or an over-budget solo run would silently proceed.
  const decision = router.select({ role: 'task', provider: budgetProvider, model: 'gpt-5.5', requestedAccountId: 'personal', costEstimate: 0.1, now })
  assert.strictEqual(decision.selected, null)
  assert.match(decision.audit.reason, /hard spend budget/)
  ok('ModelRouter.select — the solo-run entry point — refuses hard-stopped routes with a visible reason')
}

{
  const router = new ModelRouter({
    getState: () => ({ exhausted: [] }),
    setState: () => {},
  })
  const budgetProvider: Provider = {
    ...provider,
    spendBudget: { enabled: true, period: 'month', limitUsd: 10, spentUsd: 8, warnAtPercent: 80, hardStop: false },
  }
  const route = router.route({ role: 'chat', provider: budgetProvider, model: 'gpt-5.5', requestedAccountId: 'personal', costEstimate: 0.2, now })
  assert.strictEqual(route.decision.selectedAccountId, 'personal')
  assert.strictEqual(route.spend.blocked, false)
  assert.match(route.spend.warning ?? '', /near the 10.00 USD month budget/)
  ok('ModelRouter treats whole-number spend warning thresholds as percentages')
}

{
  let state: ModelRouteState = { exhausted: [] }
  const router = new ModelRouter({
    getState: () => state,
    setState: (next) => { state = next },
  })
  const budgetProvider: Provider = {
    ...provider,
    accounts: provider.accounts?.map((account) => account.id === 'work'
      ? { ...account, spendBudget: { enabled: true, period: 'month', limitUsd: 10, spentUsd: 9.9, hardStop: true } }
      : account
    ),
  }
  const routed = router.routeAfterError({
    role: 'chat',
    provider: budgetProvider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: { status: 429, message: 'too many requests' },
    costEstimate: 0.25,
    now,
  })
  assert.strictEqual(routed.signal?.kind, 'rate_limit')
  assert.strictEqual(routed.route.selected, null)
  assert.match(routed.route.audit.reason, /hard spend budget/)
  ok('ModelRouter applies spend controls to recovery failover routes')
}

{
  const router = new ModelRouter({
    getState: () => ({ exhausted: [] }),
    setState: () => {},
  })
  const cloudSuite: Provider = {
    id: 'frontier-suite',
    name: 'Frontier Suite',
    baseUrl: 'https://api.example.com/v1',
    apiKey: '',
    models: ['opus-4.8', 'sonnet-fast'],
    kind: 'cloud',
  }
  const localSuite: Provider = {
    id: 'local-suite',
    name: 'Local Suite',
    baseUrl: 'http://127.0.0.1:9000/v1',
    apiKey: '',
    models: ['qwen-coder-32k', 'tiny-summary-8k'],
    kind: 'local',
  }
  const scorecards = [
    { providerId: cloudSuite.id, model: 'opus-4.8', scores: { reviewer: 0.98, builder: 0.42, planner: 0.93 }, successRate: 0.94, latencyMs: 2_500, costPerMillionTokens: 24 },
    { providerId: localSuite.id, model: 'qwen-coder-32k', scores: { builder: 0.97, reviewer: 0.45, scout: 0.88 }, successRate: 0.9, latencyMs: 900, costPerMillionTokens: 0 },
    { providerId: localSuite.id, model: 'tiny-summary-8k', scores: { summarizer: 0.92, connector: 0.86 }, successRate: 0.87, latencyMs: 500, costPerMillionTokens: 0 },
  ]
  const builder = router.recommendRoleRoute({ role: 'builder', providers: [cloudSuite, localSuite], scorecards, source: 'unit-test' })
  assert.strictEqual(builder.selected?.providerId, localSuite.id)
  assert.strictEqual(builder.selected?.model, 'qwen-coder-32k')
  assert.ok((builder.selected?.score ?? 0) > 0.8)
  const reviewer = router.recommendRoleRoute({ role: 'reviewer', providers: [cloudSuite, localSuite], scorecards, source: 'unit-test' })
  assert.strictEqual(reviewer.selected?.providerId, cloudSuite.id)
  assert.strictEqual(reviewer.selected?.model, 'opus-4.8')
  const localReviewer = router.recommendRoleRoute({ role: 'reviewer', providers: [cloudSuite, localSuite], scorecards, localOnly: true, source: 'unit-test' })
  assert.strictEqual(localReviewer.selected?.providerId, localSuite.id)
  assert.ok(localReviewer.rankings.every((ranking) => ranking.providerId === localSuite.id))
  ok('ModelRouter recommends scorecard-backed role routes while honoring local-only')
}

{
  const noAutoProvider: Provider = {
    ...provider,
    accounts: provider.accounts?.map((account) => account.id === 'personal' ? { ...account, autoFailover: false } : account),
  }
  const routed = routeAfterModelError(undefined, {
    provider: noAutoProvider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: { status: 429, message: 'too many requests', headers: { 'retry-after': '15' } },
    now,
  })
  assert.strictEqual(routed.signal?.kind, 'rate_limit')
  assert.strictEqual(routed.route.selected, null)
  assert.strictEqual(routed.audit[1].action, 'no_route')
  ok('routeAfterModelError obeys account-level automatic failover opt-in')
}

{
  const noAutoProvider: Provider = {
    ...provider,
    accounts: provider.accounts?.map((account) => ({ ...account, autoFailover: false })),
  }
  const routed = routeAfterModelError(undefined, {
    provider: noAutoProvider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: { status: 429, message: 'too many requests', headers: { 'retry-after': '15' } },
    allowFailover: true,
    now,
  })
  assert.strictEqual(routed.signal?.kind, 'rate_limit')
  assert.strictEqual(routed.route.selectedAccountId, 'work')
  assert.strictEqual(routed.audit[1].action, 'failover')
  ok('routeAfterModelError can honor a caller-level rotate-on-limit failover policy')
}

{
  const routed = routeAfterModelError(undefined, {
    provider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: new Error('HTTP 500 upstream exploded'),
    now,
  })
  assert.strictEqual(routed.signal, null)
  assert.strictEqual(routed.route.selectedAccountId, 'personal')
  assert.strictEqual(routed.audit[0].reason, 'Error is not provider account exhaustion.')
  ok('leaves non-exhaustion errors as normal transport failures')
}

{
  const routed = routeAfterModelError(undefined, {
    provider,
    accountId: 'personal',
    model: 'gpt-5.5',
    error: new Error('HTTP 429 Too Many Requests'),
    now,
  })
  const serialized = JSON.stringify(routed.audit)
  assert.ok(!serialized.includes('secret-a'))
  assert.ok(!serialized.includes('secret-b'))
  assert.ok(!serialized.includes('provider-secret'))
  ok('keeps audit metadata free of provider secrets')
}

console.log(`model router tests: ${passed} passed`)
