import assert from 'node:assert/strict'
import { mergeSettingsSecrets, redactSettings, SECRET_MASK } from './security'
import { providerWithAccount } from './provider-accounts'
import type { Settings } from './types'

function fixture(): Settings {
  return {
    providers: [{
      id: 'fixture-provider', name: 'Fixture provider', kind: 'cloud', transport: 'chat-completions',
      baseUrl: 'https://api.example.com/v1', apiKey: 'fixture-default-key', models: ['fixture-model'],
      serverUrl: 'https://opencode.example.com', serverUsername: 'fixture', serverPassword: 'fixture-server-password',
      accounts: [{ id: 'fixture-account', name: 'Fixture account', kind: 'api-key', enabled: true,
        apiKey: 'fixture-account-key', models: [], env: {}, createdAt: 0, updatedAt: 0 }],
    }],
    defaultCwd: null,
    capabilities: { disabledCapabilityIds: [], health: {}, localScripts: [], packagedPlugins: [] },
    email: { sources: [{ id: 'mail', kind: 'imap', path: '', label: 'Mail', enabled: true,
      host: 'imap.example.com', port: 993, tls: true, user: 'fixture', password: 'fixture-mail-password',
      createdAt: 0, updatedAt: 0, lastSyncAt: null }], maxMessages: 10, lastSyncAt: null },
    slack: { sources: [], maxMessages: 10, lastSyncAt: null },
    discord: { sources: [], lastSyncAt: null },
    mattermost: { sources: [{ id: 'mm', label: 'Mattermost', baseUrl: 'https://mm.example.com',
      token: 'fixture-mm-token', enabled: true, allowedChannels: '', createdAt: 0, updatedAt: 0, lastSyncAt: null }], lastSyncAt: null },
    matrix: { sources: [{ id: 'mx', label: 'Matrix', baseUrl: 'https://matrix.example.com',
      token: 'fixture-mx-token', enabled: true, allowedRooms: '', createdAt: 0, updatedAt: 0, lastSyncAt: null }], lastSyncAt: null },
    googlechat: { sources: [], lastSyncAt: null },
    botControl: {
      telegram: { enabled: false, botToken: '', allowedChatIds: '', allowedUserIds: '', polling: false,
        lastUpdateId: null, lastPollAt: null, lastError: null },
      whatsapp: { enabled: false, accessToken: '', phoneNumberId: '', verifyToken: '', appSecret: '',
        allowedChatIds: '', lastEventAt: null, lastError: null },
    },
    voice: { apiBaseUrl: 'https://voice.example.com/v1', apiKey: 'fixture-voice-key', elevenLabsApiKey: '' },
    remoteControl: { authToken: 'fixture-local-owner-token' },
    remoteClient: { serverUrl: 'https://remote.example.com', authToken: 'fixture-remote-client-token' },
    remoteOrchestrators: [{ id: 'remote', serverUrl: 'https://remote.example.com', authToken: 'fixture-remote-host-token' }],
  } as unknown as Settings
}

const checks: Array<{ name: string; edit: (s: Settings) => void; secret: (s: Settings) => unknown }> = [
  { name: 'Provider account credential follows the protected base URL',
    edit: (s) => { s.providers[0].baseUrl = 'https://changed.example.com/v1' },
    secret: (s) => providerWithAccount(s.providers[0], 'fixture-account').apiKey },
  { name: 'OpenCode server password follows its server URL',
    edit: (s) => { s.providers[0].serverUrl = 'https://changed.example.com' },
    secret: (s) => s.providers[0].serverPassword },
  { name: 'Mattermost token follows its server URL',
    edit: (s) => { s.mattermost.sources[0].baseUrl = 'https://changed.example.com' },
    secret: (s) => s.mattermost.sources[0].token },
  { name: 'Matrix token follows its homeserver URL',
    edit: (s) => { s.matrix.sources[0].baseUrl = 'https://changed.example.com' },
    secret: (s) => s.matrix.sources[0].token },
  { name: 'IMAP password follows its hostname',
    edit: (s) => { s.email.sources[0].host = 'changed.example.com' },
    secret: (s) => s.email.sources[0].password },
  { name: 'IMAP password follows its selected transport mode',
    edit: (s) => { s.email.sources[0].tls = false },
    secret: (s) => s.email.sources[0].password },
  { name: 'IMAP password follows its port',
    edit: (s) => { s.email.sources[0].port = 1993 },
    secret: (s) => s.email.sources[0].password },
  { name: 'Voice credential follows its API endpoint',
    edit: (s) => { s.voice.apiBaseUrl = 'https://changed.example.com/v1' },
    secret: (s) => s.voice.apiKey },
  { name: 'Remote client token follows its server URL',
    edit: (s) => { s.remoteClient.serverUrl = 'https://changed.example.com' },
    secret: (s) => s.remoteClient.authToken },
  { name: 'Remote host token follows its server URL',
    edit: (s) => { s.remoteOrchestrators[0].serverUrl = 'https://changed.example.com' },
    secret: (s) => s.remoteOrchestrators[0].authToken },
  { name: 'A new remote profile cannot borrow the legacy client token',
    edit: (s) => { s.remoteOrchestrators[0].id = 'unknown-profile'; s.remoteOrchestrators[0].serverUrl = 'https://changed.example.com' },
    secret: (s) => s.remoteOrchestrators[0].authToken },
]

let failures = 0
for (const check of checks) {
  try {
    const current = fixture()
    const next = redactSettings(current)
    check.edit(next)
    assert.equal(check.secret(mergeSettingsSecrets(next, current)), '', check.name)
    console.log(`PASS ${check.name}`)
  } catch (error) { failures++; console.error(error) }
}

const current = fixture()
const unchanged = mergeSettingsSecrets(redactSettings(current), current)
for (const check of checks.slice(0, -1)) assert.equal(check.secret(unchanged), check.secret(current), `${check.name}: unchanged retained`)
const explicitlyEntered = redactSettings(current)
explicitlyEntered.mattermost.sources[0].baseUrl = 'https://changed.example.com'
explicitlyEntered.mattermost.sources[0].token = 'fixture-new-token'
assert.equal(mergeSettingsSecrets(explicitlyEntered, current).mattermost.sources[0].token, 'fixture-new-token')
const slashOnly = redactSettings(current)
slashOnly.mattermost.sources[0].baseUrl += '/'
assert.equal(mergeSettingsSecrets(slashOnly, current).mattermost.sources[0].token, current.mattermost.sources[0].token)
const clearedThenRetargeted = redactSettings(current)
clearedThenRetargeted.remoteClient.serverUrl = ''
assert.equal(clearedThenRetargeted.remoteClient.authToken, SECRET_MASK)
assert.equal(mergeSettingsSecrets(clearedThenRetargeted, current).remoteClient.authToken, '',
  'An empty intermediate endpoint must not retain a secret for a later endpoint edit')
assert.equal(failures, 0, `${failures} masked-secret destination binding regressions failed`)
