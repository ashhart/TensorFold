import { capturedExternalSecretKeyValue, decryptWithKey, encryptWithKey, externalSecretKeyRequired, resolveLocalSecretKeySync } from '../local-crypto'
import { createHmac } from 'node:crypto'
import { vontraPath, isRecord } from './paths'
import { quarantinedProtectedFiles } from './recovery-registry'
import { normalizeLegacyCliModelList } from '../agent/cli-models'
import type { Settings, McpServerConfig, WebhookEndpoint, Provider, EmailSourceConfig, SlackSourceConfig, DiscordSource, MattermostSource, MatrixSource, GoogleChatSource } from '../agent/types'

const SECRET_PREFIX = 'safe:v1:'
const PROTECTED_JSON_PREFIX = 'safe-json:v1:'

let cachedAppKeys: {
  discriminator: string
  key: Buffer
  mode: 'profile-key' | 'external-env-key'
  profileFallbackKey: Buffer | null
} | null = null

function appKeys() {
  const keyPath = vontraPath('.secretkey')
  const external = capturedExternalSecretKeyValue() ?? ''
  const discriminator = keyPath
  if (!cachedAppKeys || cachedAppKeys.discriminator !== discriminator) {
    cachedAppKeys = { discriminator, ...resolveLocalSecretKeySync(keyPath, external) }
  }
  return cachedAppKeys
}

function appKey(): Buffer { return appKeys().key }

export function blindSearchToken(token: string): string {
  return createHmac('sha256', appKey()).update(`vontra-search-v1\0${token}`).digest('hex')
}

function decryptWithAppKeys(payload: string): { plaintext: string; source: 'primary' | 'profile-fallback' } | null {
  const keys = appKeys()
  try { return { plaintext: decryptWithKey(payload, keys.key), source: 'primary' } } catch {  }
  if (keys.profileFallbackKey) {
    try { return { plaintext: decryptWithKey(payload, keys.profileFallbackKey), source: 'profile-fallback' } } catch {  }
  }
  return null
}

function revealLegacySafeStorage(_payload: string): string | undefined {
  return undefined
}

function isLegacySecretBlob(value: unknown): boolean {
  if (typeof value !== 'string' || !value.startsWith(SECRET_PREFIX)) return false
  return !decryptWithAppKeys(value.slice(SECRET_PREFIX.length))
}

function isProfileFallbackSecretBlob(value: unknown): boolean {
  if (typeof value !== 'string' || !value.startsWith(SECRET_PREFIX)) return false
  return decryptWithAppKeys(value.slice(SECRET_PREFIX.length))?.source === 'profile-fallback'
}

function settingsSecretValues(loaded: Partial<Settings>): unknown[] {
  const values: unknown[] = []
  for (const p of loaded.providers ?? []) {
    values.push(p.apiKey, p.serverPassword, ...Object.values(p.cliEnv ?? {}))
    for (const a of p.accounts ?? []) {
      values.push(a.apiKey, ...Object.values(a.env ?? {}))
    }
  }
  values.push(loaded.imessage?.blueBubblesPassword)
  values.push(...(loaded.email?.sources ?? []).map((s) => s.password))
  values.push(...(loaded.slack?.sources ?? []).map((s) => s.token))
  values.push(...(loaded.discord?.sources ?? []).map((s) => s.token))
  values.push(...(loaded.mattermost?.sources ?? []).map((s) => s.token))
  values.push(...(loaded.matrix?.sources ?? []).map((s) => s.token))
  for (const s of loaded.googlechat?.sources ?? []) values.push(s.webhookUrl, s.accessToken)
  const bot = loaded.botControl
  values.push(bot?.telegram?.botToken, bot?.whatsapp?.accessToken, bot?.whatsapp?.verifyToken, bot?.whatsapp?.appSecret)
  values.push(loaded.voice?.apiKey, loaded.voice?.elevenLabsApiKey)
  values.push(loaded.remoteControl?.authToken, loaded.remoteClient?.authToken)
  values.push(...(loaded.remoteOrchestrators ?? []).map((h) => h.authToken))
  return values
}

function hasLegacySecret(loaded: Partial<Settings>): boolean { return settingsSecretValues(loaded).some(isLegacySecretBlob) }
function countLegacySecrets(loaded: Partial<Settings>): number { return settingsSecretValues(loaded).filter(isLegacySecretBlob).length }
function hasProfileFallbackSecret(loaded: Partial<Settings>): boolean { return settingsSecretValues(loaded).some(isProfileFallbackSecretBlob) }

let legacySecretTemplate: Partial<Settings> | null = null
let legacySettingsBackupPath: string | null = null
let legacyMcpTemplate: McpServerConfig[] | null = null
let legacyMcpBackupPath: string | null = null
let legacyWebhookTemplate: WebhookEndpoint[] | null = null
let legacyWebhookBackupPath: string | null = null

export function getLegacySecretTemplate(): Partial<Settings> | null { return legacySecretTemplate }
export function setLegacySecretTemplate(value: Partial<Settings> | null): void { legacySecretTemplate = value }
export function getLegacySettingsBackupPath(): string | null { return legacySettingsBackupPath }
export function setLegacySettingsBackupPath(value: string | null): void { legacySettingsBackupPath = value }
export function getLegacyMcpTemplate(): McpServerConfig[] | null { return legacyMcpTemplate }
export function setLegacyMcpTemplate(value: McpServerConfig[] | null): void { legacyMcpTemplate = value }
export function getLegacyMcpBackupPath(): string | null { return legacyMcpBackupPath }
export function setLegacyMcpBackupPath(value: string | null): void { legacyMcpBackupPath = value }
export function getLegacyWebhookTemplate(): WebhookEndpoint[] | null { return legacyWebhookTemplate }
export function setLegacyWebhookTemplate(value: WebhookEndpoint[] | null): void { legacyWebhookTemplate = value }
export function getLegacyWebhookBackupPath(): string | null { return legacyWebhookBackupPath }
export function setLegacyWebhookBackupPath(value: string | null): void { legacyWebhookBackupPath = value }

function countMcpLegacySecrets(items: McpServerConfig[] | null): number {
  return (items ?? []).flatMap((server) => Object.values(server.env ?? {})).filter(isLegacySecretBlob).length
}

function countWebhookLegacySecrets(items: WebhookEndpoint[] | null): number {
  return (items ?? []).map((endpoint) => endpoint.secretRef).filter(isLegacySecretBlob).length
}

export interface LocalSecretStorageStatus {
  keyMode: 'profile-key' | 'external-env-key'
  externalKeyRequired: boolean
  copiedProfileDecryptable: boolean
  legacySafeStorageSecrets: number
  legacySafeStorageBackupPath: string | null
  quarantinedProtectedFiles: number
  quarantinedProtectedPaths: string[]
  quarantineBackupPaths: string[]
}

export function localSecretStorageStatus(): LocalSecretStorageStatus {
  const keys = appKeys()
  const keyMode = keys.mode
  return {
    keyMode,
    externalKeyRequired: externalSecretKeyRequired(),

    copiedProfileDecryptable: keyMode === 'profile-key' || keys.profileFallbackKey !== null,
    legacySafeStorageSecrets:
      (legacySecretTemplate ? countLegacySecrets(legacySecretTemplate) : 0) +
      countMcpLegacySecrets(legacyMcpTemplate) +
      countWebhookLegacySecrets(legacyWebhookTemplate),
    legacySafeStorageBackupPath: legacySettingsBackupPath ?? legacyMcpBackupPath ?? legacyWebhookBackupPath,
    quarantinedProtectedFiles: quarantinedProtectedFiles.size,
    quarantinedProtectedPaths: [...quarantinedProtectedFiles.keys()],
    quarantineBackupPaths: [...quarantinedProtectedFiles.values()].map((entry) => entry.backupPath),
  }
}

function containsLegacySecret(value: unknown): boolean {
  if (isLegacySecretBlob(value)) return true
  if (Array.isArray(value)) return value.some(containsLegacySecret)
  return isRecord(value) && Object.values(value).some(containsLegacySecret)
}

function mergeUnresolvedLegacySecrets(template: unknown, next: unknown): unknown {
  if (isLegacySecretBlob(template)) {
    return typeof next === 'string' && next.length > 0 ? next : template
  }
  if (Array.isArray(template) && Array.isArray(next)) {
    return next.map((item, index) => {
      const itemId = isRecord(item) && typeof item.id === 'string' ? item.id : null
      const prior = itemId
        ? template.find((candidate) => isRecord(candidate) && candidate.id === itemId)
        : template[index]
      return prior === undefined ? item : mergeUnresolvedLegacySecrets(prior, item)
    })
  }
  if (isRecord(template) && isRecord(next) && containsLegacySecret(template)) {
    const merged: Record<string, unknown> = { ...next }
    for (const [key, prior] of Object.entries(template)) {
      if (Object.prototype.hasOwnProperty.call(next, key) && containsLegacySecret(prior)) {
        merged[key] = mergeUnresolvedLegacySecrets(prior, next[key])
      }
    }
    return merged
  }
  return next
}

function protectSecret(value: string): string {
  if (!value || value.startsWith(SECRET_PREFIX)) return value
  return SECRET_PREFIX + encryptWithKey(value, appKey())
}

function revealSecret(value: string): string {
  if (!value?.startsWith(SECRET_PREFIX)) return value || ''
  const payload = value.slice(SECRET_PREFIX.length)
  const decrypted = decryptWithAppKeys(payload)
  if (decrypted) return decrypted.plaintext
  {

    return revealLegacySafeStorage(payload) ?? ''
  }
}

function protectJsonPayload(data: unknown): unknown {
  return { encrypted: PROTECTED_JSON_PREFIX + encryptWithKey(JSON.stringify(data), appKey()) }
}

function revealJsonPayload<T>(data: unknown, fallback: T): T {
  if (!data || typeof data !== 'object' || Array.isArray(data)) return data as T
  const encrypted = (data as { encrypted?: unknown }).encrypted
  if (typeof encrypted !== 'string' || !encrypted.startsWith(PROTECTED_JSON_PREFIX)) return data as T
  const payload = encrypted.slice(PROTECTED_JSON_PREFIX.length)
  const decrypted = decryptWithAppKeys(payload)
  if (decrypted) {
    try { return JSON.parse(decrypted.plaintext) as T } catch { return fallback }
  }
  {

    const legacy = revealLegacySafeStorage(payload)
    if (legacy != null) {
      try { return JSON.parse(legacy) as T } catch {  }
    }
    return fallback
  }
}

function isProtectedJsonEnvelope(data: unknown): boolean {
  return !!data &&
    typeof data === 'object' &&
    !Array.isArray(data) &&
    typeof (data as { encrypted?: unknown }).encrypted === 'string' &&
    (data as { encrypted: string }).encrypted.startsWith(PROTECTED_JSON_PREFIX)
}

function isLegacyProtectedEnvelope(data: unknown): boolean {
  const enc = (data as { encrypted?: unknown } | null | undefined)?.encrypted
  if (typeof enc !== 'string' || !enc.startsWith(PROTECTED_JSON_PREFIX)) return false
  return !decryptWithAppKeys(enc.slice(PROTECTED_JSON_PREFIX.length))
}

function isProfileFallbackProtectedEnvelope(data: unknown): boolean {
  const enc = (data as { encrypted?: unknown } | null | undefined)?.encrypted
  if (typeof enc !== 'string' || !enc.startsWith(PROTECTED_JSON_PREFIX)) return false
  return decryptWithAppKeys(enc.slice(PROTECTED_JSON_PREFIX.length))?.source === 'profile-fallback'
}

function hasMalformedDecryptedProtectedPayload(data: unknown): boolean {
  const enc = (data as { encrypted?: unknown } | null | undefined)?.encrypted
  if (typeof enc !== 'string' || !enc.startsWith(PROTECTED_JSON_PREFIX)) return false
  const decrypted = decryptWithAppKeys(enc.slice(PROTECTED_JSON_PREFIX.length))
  if (!decrypted) return false // wrong/missing keys are classified separately below
  try {
    JSON.parse(decrypted.plaintext)
    return false
  } catch {
    return true
  }
}

function isUndecryptableEnvelope(data: unknown): boolean {
  if (!isLegacyProtectedEnvelope(data)) return false // either ours, or not an envelope
  const enc = (data as { encrypted: string }).encrypted.slice(PROTECTED_JSON_PREFIX.length)
  return revealLegacySafeStorage(enc) == null // safeStorage can't recover it either -> truly lost
}

function matchesProtectedFallbackShape(value: unknown, fallback: unknown): boolean {
  if (Array.isArray(fallback)) return Array.isArray(value)
  if (fallback !== null && typeof fallback === 'object') {
    return value !== null && typeof value === 'object' && !Array.isArray(value)
  }
  if (fallback !== null && fallback !== undefined) return typeof value === typeof fallback

  return true
}

function protectSettings(s: Settings): Settings {
  const protectProvider = (p: Provider): Provider => ({
    ...p,
    apiKey: protectSecret(p.apiKey),
    serverPassword: p.serverPassword ? protectSecret(p.serverPassword) : p.serverPassword,
    cliEnv: p.cliEnv ? Object.fromEntries(Object.entries(p.cliEnv).map(([key, value]) => [key, protectSecret(value)])) : p.cliEnv,
    accounts: p.accounts?.map((account) => ({
      ...account,
      apiKey: protectSecret(account.apiKey ?? ''),
      env: account.env ? Object.fromEntries(Object.entries(account.env).map(([key, value]) => [key, protectSecret(value)])) : account.env,
    })),
  })
  return {
    ...s,
    providers: s.providers.map(protectProvider),
    imessage: { ...s.imessage, blueBubblesPassword: s.imessage.blueBubblesPassword ? protectSecret(s.imessage.blueBubblesPassword) : s.imessage.blueBubblesPassword },
    email: { ...s.email, sources: s.email.sources.map(protectEmailSource) },
    slack: { ...s.slack, sources: s.slack.sources.map(protectSlackSource) },
    discord: { ...s.discord, sources: s.discord.sources.map(protectDiscordSource) },
    mattermost: { ...s.mattermost, sources: s.mattermost.sources.map(protectMattermostSource) },
    matrix: { ...s.matrix, sources: s.matrix.sources.map(protectMatrixSource) },
    googlechat: { ...s.googlechat, sources: s.googlechat.sources.map(protectGoogleChatSource) },
    botControl: {
      ...s.botControl,
      telegram: { ...s.botControl.telegram, botToken: protectSecret(s.botControl.telegram.botToken) },
      whatsapp: {
        ...s.botControl.whatsapp,
        accessToken: protectSecret(s.botControl.whatsapp.accessToken),
        verifyToken: protectSecret(s.botControl.whatsapp.verifyToken),
        appSecret: protectSecret(s.botControl.whatsapp.appSecret),
      },
    },
    voice: {
      ...s.voice,
      apiKey: protectSecret(s.voice.apiKey),
      elevenLabsApiKey: protectSecret(s.voice.elevenLabsApiKey ?? ''),
    },
    remoteControl: {
      ...s.remoteControl,
      authToken: protectSecret(s.remoteControl.authToken),
    },
    remoteClient: {
      ...s.remoteClient,
      authToken: protectSecret(s.remoteClient.authToken),
    },
    remoteOrchestrators: s.remoteOrchestrators.map((host) => ({
      ...host,
      authToken: protectSecret(host.authToken),
    })),
  }
}

// IMAP account passwords live on each email source. Encrypt at rest; reveal on load.
function protectEmailSource(src: EmailSourceConfig): EmailSourceConfig {
  return { ...src, password: src.password ? protectSecret(src.password) : src.password }
}
function revealEmailSource(src: EmailSourceConfig): EmailSourceConfig {
  return { ...src, password: src.password ? revealSecret(src.password) : src.password }
}
// Live Slack workspace tokens (kind: 'api_token') are protected secrets too.
function protectSlackSource(src: SlackSourceConfig): SlackSourceConfig {
  return { ...src, token: src.token ? protectSecret(src.token) : src.token }
}
function revealSlackSource(src: SlackSourceConfig): SlackSourceConfig {
  return { ...src, token: src.token ? revealSecret(src.token) : src.token }
}
// Discord bot tokens live on each discord source.
function protectDiscordSource(src: DiscordSource): DiscordSource {
  return { ...src, token: src.token ? protectSecret(src.token) : src.token }
}
function revealDiscordSource(src: DiscordSource): DiscordSource {
  return { ...src, token: src.token ? revealSecret(src.token) : src.token }
}
// Mattermost Personal Access / bot tokens live on each mattermost source.
function protectMattermostSource(src: MattermostSource): MattermostSource {
  return { ...src, token: src.token ? protectSecret(src.token) : src.token }
}
function revealMattermostSource(src: MattermostSource): MattermostSource {
  return { ...src, token: src.token ? revealSecret(src.token) : src.token }
}
// Matrix access tokens live on each matrix source.
function protectMatrixSource(src: MatrixSource): MatrixSource {
  return { ...src, token: src.token ? protectSecret(src.token) : src.token }
}
function revealMatrixSource(src: MatrixSource): MatrixSource {
  return { ...src, token: src.token ? revealSecret(src.token) : src.token }
}

function protectGoogleChatSource(src: GoogleChatSource): GoogleChatSource {
  return {
    ...src,
    webhookUrl: src.webhookUrl ? protectSecret(src.webhookUrl) : src.webhookUrl,
    accessToken: src.accessToken ? protectSecret(src.accessToken) : src.accessToken,
  }
}
function revealGoogleChatSource(src: GoogleChatSource): GoogleChatSource {
  return {
    ...src,
    webhookUrl: src.webhookUrl ? revealSecret(src.webhookUrl) : src.webhookUrl,
    accessToken: src.accessToken ? revealSecret(src.accessToken) : src.accessToken,
  }
}

function revealSettings(s: Settings): Settings {
  const revealProvider = (p: Provider): Provider => ({
    ...p,
    apiKey: revealSecret(p.apiKey),
    serverPassword: p.serverPassword ? revealSecret(p.serverPassword) : p.serverPassword,
    cliEnv: p.cliEnv ? Object.fromEntries(Object.entries(p.cliEnv).map(([key, value]) => [key, revealSecret(value)])) : p.cliEnv,
    accounts: p.accounts?.map((account) => ({
      ...account,
      apiKey: revealSecret(account.apiKey ?? ''),
      env: account.env ? Object.fromEntries(Object.entries(account.env).map(([key, value]) => [key, revealSecret(value)])) : account.env,
    })),
  })
  return {
    ...s,
    providers: s.providers.map(revealProvider),
    imessage: { ...s.imessage, blueBubblesPassword: s.imessage.blueBubblesPassword ? revealSecret(s.imessage.blueBubblesPassword) : s.imessage.blueBubblesPassword },
    email: { ...s.email, sources: s.email.sources.map(revealEmailSource) },
    slack: { ...s.slack, sources: s.slack.sources.map(revealSlackSource) },
    discord: { ...s.discord, sources: s.discord.sources.map(revealDiscordSource) },
    mattermost: { ...s.mattermost, sources: s.mattermost.sources.map(revealMattermostSource) },
    matrix: { ...s.matrix, sources: s.matrix.sources.map(revealMatrixSource) },
    googlechat: { ...s.googlechat, sources: s.googlechat.sources.map(revealGoogleChatSource) },
    botControl: {
      ...s.botControl,
      telegram: { ...s.botControl.telegram, botToken: revealSecret(s.botControl.telegram.botToken) },
      whatsapp: {
        ...s.botControl.whatsapp,
        accessToken: revealSecret(s.botControl.whatsapp.accessToken),
        verifyToken: revealSecret(s.botControl.whatsapp.verifyToken),
        appSecret: revealSecret(s.botControl.whatsapp.appSecret),
      },
    },
    voice: {
      ...s.voice,
      apiKey: revealSecret(s.voice.apiKey),
      elevenLabsApiKey: revealSecret(s.voice.elevenLabsApiKey ?? ''),
    },
    remoteControl: {
      ...s.remoteControl,
      authToken: revealSecret(s.remoteControl.authToken),
    },
    remoteClient: {
      ...s.remoteClient,
      authToken: revealSecret(s.remoteClient.authToken),
    },
    remoteOrchestrators: s.remoteOrchestrators.map((host) => ({
      ...host,
      authToken: revealSecret(host.authToken),
    })),
  }
}

function normalizeCliSubscriptionModels(settings: Settings): { settings: Settings; changed: boolean } {
  let changed = false
  const providers = settings.providers.map((provider) => {
    if (provider.transport !== 'codex-cli' && provider.transport !== 'claude-cli') return provider
    const models = normalizeLegacyCliModelList(provider.transport, provider.models)
    if (models.join('\n') !== (provider.models ?? []).join('\n')) changed = true
    const accounts = provider.accounts?.map((account) => {
      if (!account.models?.length) return account
      const accountModels = normalizeLegacyCliModelList(provider.transport, account.models)
      if (accountModels.join('\n') !== account.models.join('\n')) changed = true
      return { ...account, models: accountModels }
    })
    return { ...provider, models, accounts }
  })
  return { settings: { ...settings, providers }, changed }
}

function protectMcpServers(servers: McpServerConfig[]): McpServerConfig[] {
  return servers.map((server) => ({
    ...server,
    env: Object.fromEntries(Object.entries(server.env ?? {}).map(([key, value]) => [key, protectSecret(value)])),
  }))
}

function revealMcpServers(servers: McpServerConfig[]): McpServerConfig[] {
  return servers.map((server) => ({
    ...server,
    env: Object.fromEntries(Object.entries(server.env ?? {}).map(([key, value]) => [key, revealSecret(value)])),
  }))
}

export {
  SECRET_PREFIX,
  PROTECTED_JSON_PREFIX,
  appKeys,
  decryptWithAppKeys,
  revealLegacySafeStorage,
  isLegacySecretBlob,
  isProfileFallbackSecretBlob,
  settingsSecretValues,
  hasLegacySecret,
  countLegacySecrets,
  hasProfileFallbackSecret,
  countMcpLegacySecrets,
  countWebhookLegacySecrets,
  containsLegacySecret,
  mergeUnresolvedLegacySecrets,
  protectSecret,
  revealSecret,
  protectJsonPayload,
  revealJsonPayload,
  isProtectedJsonEnvelope,
  isLegacyProtectedEnvelope,
  isProfileFallbackProtectedEnvelope,
  hasMalformedDecryptedProtectedPayload,
  isUndecryptableEnvelope,
  matchesProtectedFallbackShape,
  protectSettings,
  revealSettings,
  normalizeCliSubscriptionModels,
  protectMcpServers,
  revealMcpServers,
  protectWebhooks,
  revealWebhooks,
}

function protectWebhooks(w: WebhookEndpoint[]): WebhookEndpoint[] { return w.map((x) => ({ ...x, secretRef: protectSecret(x.secretRef) })) }
function revealWebhooks(w: WebhookEndpoint[]): WebhookEndpoint[] { return w.map((x) => ({ ...x, secretRef: revealSecret(x.secretRef) })) }
