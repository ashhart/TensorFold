import assert from 'node:assert/strict'
import { existsSync, mkdtempSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

const home = mkdtempSync(path.join(os.tmpdir(), 'vontra-acp-main-secrets-'))
const externalKey = Buffer.alloc(32, 41)
process.env.VONTRA_HOME = home
process.env.VONTRA_SECRET_KEY_BASE64 = externalKey.toString('base64')
delete process.env.VONTRA_REQUIRE_EXTERNAL_SECRET_KEY

// Import only after the managed deployment environment is installed: local-crypto
// captures the external key once and removes it from process.env at module load.
const { encryptWithKey } = await import('../local-crypto.ts')
const protect = (value: string) => `safe:v1:${encryptWithKey(value, externalKey)}`

writeFileSync(path.join(home, 'settings.json'), JSON.stringify({
  activeProviderId: 'acp-provider',
  activeModel: 'acp-model',
  providers: [{
    id: 'acp-provider',
    name: 'ACP provider',
    transport: 'opencode',
    baseUrl: 'https://provider.example.test/v1',
    apiKey: protect('provider-api-key'),
    models: ['acp-model'],
    serverUrl: 'http://127.0.0.1:4096',
    serverPassword: protect('server-password'),
    cliEnv: {
      PROVIDER_TOKEN: protect('provider-cli-token'),
      NON_SECRET_SETTING: 'plain-setting',
      WRONG_KEY_TOKEN: `safe:v1:${encryptWithKey('must-not-pass-through', Buffer.alloc(32, 88))}`,
    },
    accounts: [{
      id: 'account-one',
      name: 'Account one',
      kind: 'api-key',
      enabled: true,
      apiKey: protect('account-api-key'),
      env: { ACCOUNT_TOKEN: protect('account-env-token') },
      createdAt: 1,
      updatedAt: 1,
    }],
  }],
}, null, 2), { mode: 0o600 })

const { resolveDefaultProviderModel } = await import('./acp-server-main.ts')
const resolved = resolveDefaultProviderModel()

assert.equal(resolved.model, 'acp-model')
assert.equal(resolved.provider.apiKey, 'provider-api-key', 'ACP reveals the provider API key')
assert.equal(resolved.provider.serverPassword, 'server-password', 'ACP reveals the local-server password')
assert.equal(resolved.provider.cliEnv?.PROVIDER_TOKEN, 'provider-cli-token', 'ACP reveals provider CLI environment secrets')
assert.equal(resolved.provider.cliEnv?.NON_SECRET_SETTING, 'plain-setting', 'ACP preserves non-secret CLI environment values')
assert.equal(resolved.provider.cliEnv?.WRONG_KEY_TOKEN, '', 'ACP sanitizes an unreadable protected value instead of forwarding safe:v1 ciphertext')
assert.equal(resolved.provider.accounts?.[0]?.apiKey, 'account-api-key', 'ACP reveals account API keys')
assert.equal(resolved.provider.accounts?.[0]?.env?.ACCOUNT_TOKEN, 'account-env-token', 'ACP reveals account environment secrets')
assert.equal(process.env.VONTRA_SECRET_KEY_BASE64, undefined, 'the external key is removed before ACP can spawn provider children')
assert.equal(existsSync(path.join(home, '.secretkey')), false, 'external-key ACP mode does not create a profile key')

console.log('ACP standalone provider-secret resolution tests passed')
