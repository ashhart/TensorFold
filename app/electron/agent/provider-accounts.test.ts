// Run: node electron/agent/provider-accounts.test.ts
import assert from 'node:assert'
import { normalizeLegacyCliModelList } from './cli-models.ts'
import { modelChoicesForProvider, providerWithAccount, resolveProviderModel, selectProviderAccount } from './provider-accounts.ts'
import type { Provider } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const codex: Provider = {
  id: 'codex-cli',
  name: 'Codex CLI',
  transport: 'codex-cli',
  auth: 'codex-oauth',
  baseUrl: '',
  apiKey: '',
  models: ['codex-cli-default'],
  cliPath: 'codex',
  cliProfile: 'base',
  accounts: [
    { id: 'personal', name: 'Personal', kind: 'subscription', enabled: true, cliProfile: 'personal', cliHome: '~/.codex-personal', models: ['codex-cli-default'], createdAt: 1, updatedAt: 1 },
    { id: 'disabled', name: 'Disabled', kind: 'subscription', enabled: false, cliProfile: 'disabled', createdAt: 1, updatedAt: 1 },
  ],
}

const apiProvider: Provider = {
  id: 'openrouter',
  name: 'OpenRouter',
  baseUrl: 'https://openrouter.ai/api/v1',
  apiKey: 'provider-key',
  models: ['openrouter/auto'],
  accounts: [
    { id: 'team', name: 'Team', kind: 'api-key', enabled: true, apiKey: 'team-key', models: ['model-a'], env: { EXTRA_TOKEN: 'secret' }, createdAt: 1, updatedAt: 1 },
  ],
}

const claude: Provider = {
  id: 'claude-cli',
  name: 'Claude CLI',
  transport: 'claude-cli',
  auth: 'claude-oauth',
  baseUrl: '',
  apiKey: '',
  models: ['claude-code-default', 'sonnet'],
  cliPath: 'claude',
  accounts: [
    {
      id: 'personal',
      name: 'Claude personal',
      kind: 'subscription',
      enabled: true,
      cliProfile: '~/.claude-personal/settings.json',
      cliHome: '~/.claude-personal',
      env: { CLAUDE_CODE_OAUTH_TOKEN: 'oauth-token' },
      models: ['sonnet'],
      createdAt: 1,
      updatedAt: 1,
    },
  ],
}

console.log('provider account tests:')

{
  assert.strictEqual(selectProviderAccount(codex, 'personal')?.name, 'Personal')
  assert.strictEqual(selectProviderAccount(codex, 'disabled'), null)
  assert.strictEqual(selectProviderAccount(codex, null), null)
  ok('selects enabled accounts and preserves explicit default-login null')
}

{
  const resolved = providerWithAccount(codex, 'personal')
  assert.strictEqual(resolved.cliProfile, 'personal')
  assert.ok(resolved.cliEnv?.CODEX_HOME?.endsWith('/.codex-personal'))
  ok('merges Codex account profile and CODEX_HOME')
}

{
  const resolved = providerWithAccount(apiProvider, 'team')
  assert.strictEqual(resolved.apiKey, 'team-key')
  assert.deepStrictEqual(resolved.models, ['model-a'])
  assert.strictEqual(resolved.cliEnv?.EXTRA_TOKEN, 'secret')
  ok('merges API account key, models, and env')
}

{
  assert.deepStrictEqual(modelChoicesForProvider(apiProvider, 'team'), ['model-a'])
  assert.strictEqual(resolveProviderModel(apiProvider, 'team', 'missing-model', 'model-a'), 'model-a')
  assert.strictEqual(resolveProviderModel(apiProvider, 'team', 'missing-model'), 'model-a')
  assert.strictEqual(resolveProviderModel({ ...apiProvider, models: [] }, null, 'custom-model'), 'custom-model')
  ok('resolves stale saved models to available provider/account models')
}

{
  const resolved = providerWithAccount(claude, 'personal')
  assert.ok(resolved.cliProfile?.endsWith('/.claude-personal/settings.json'))
  assert.ok(resolved.cliEnv?.CLAUDE_CONFIG_DIR?.endsWith('/.claude-personal'))
  assert.ok(resolved.cliEnv?.CLAUDE_HOME?.endsWith('/.claude-personal'))
  assert.strictEqual(resolved.cliEnv?.CLAUDE_CODE_OAUTH_TOKEN, 'oauth-token')
  assert.deepStrictEqual(resolved.models, ['sonnet'])
  ok('merges Claude account config home, settings file, token, and models')
}

{
  assert.ok(normalizeLegacyCliModelList('codex-cli', ['codex-cli-default']).includes('gpt-5.5'))
  assert.deepStrictEqual(normalizeLegacyCliModelList('claude-cli', ['claude-code-default', 'ultracode', 'opus-4.8', 'sonnet', 'opus']), ['claude-code-default', 'opus', 'sonnet'])
  assert.deepStrictEqual(normalizeLegacyCliModelList('codex-cli', ['custom-codex']), ['custom-codex'])
  ok('expands legacy CLI model lists while preserving custom overrides')
}

{
  assert.strictEqual(resolveProviderModel({ ...claude, models: ['claude-code-default', 'opus', 'sonnet'] }, null, 'opus-4.8'), 'opus')
  assert.strictEqual(resolveProviderModel({ ...claude, models: ['claude-code-default', 'opus', 'sonnet'] }, null, 'ultracode'), 'opus')
  ok('maps stale Claude display names to supported CLI model aliases')
}

console.log(`provider account tests: ${passed} passed`)
