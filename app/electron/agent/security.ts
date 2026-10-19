import { isManagedComputeProvider, isLoopbackUrl } from '../../shared/provider-trust'
import type { Settings, Provider, McpServerConfig, SecurityRole } from './types'
import { resolvePinnedHost, type PinnedHost, type PinnedHostResolver } from './pinned-fetch'
import { ipv4MappedToDotted, isLoopbackHost, isPrivateHost, isPrivateIp, isKnownMetadataIp, isGloballyRoutableIp, isLoopbackIp, isAlwaysForbiddenConnectorIp, isLinkLocalOrMetadataHost } from './security-addresses'
export { ipv4MappedToDotted, isLoopbackHost, isPrivateHost, isPrivateIp, isKnownMetadataIp, isGloballyRoutableIp } from './security-addresses'
import { redactOpaqueSecrets } from './secret-detection'

export const SECRET_MASK = '••••••••'
const SECURITY_ROLES = new Set<SecurityRole>(['viewer', 'operator', 'owner'])

export function currentSecurityRoleFromEnv(env: Partial<Record<string, string | undefined>> = process.env): SecurityRole {
  const raw = String(env.VONTRA_SECURITY_ROLE || env.VONTRA_ROLE || 'operator').toLowerCase()
  return SECURITY_ROLES.has(raw as SecurityRole) ? raw as SecurityRole : 'operator'
}

// The local owner may change settings; headless servers default to operator unless the launch role is explicit.
export function securityRoleForLaunch(
  env: Partial<Record<string, string | undefined>> = process.env,
  opts: { headless: boolean } = { headless: false },
): SecurityRole {
  const explicit = String(env.VONTRA_SECURITY_ROLE || env.VONTRA_ROLE || '').toLowerCase()
  if (SECURITY_ROLES.has(explicit as SecurityRole)) return explicit as SecurityRole
  return opts.headless ? 'operator' : 'owner'
}

const PREVIEW_FRAME_SOURCES = [3000, 4173, 5173].flatMap((port) => [
  `http://localhost:${port}`, `http://127.0.0.1:${port}`, `http://[::1]:${port}`,
  `https://localhost:${port}`, `https://127.0.0.1:${port}`, `https://[::1]:${port}`,
])

export function buildSessionCsp(opts: { trustedTypes?: boolean; connectSources?: string[]; frameSources?: string[]; dev?: boolean } = {}): string {
  // Renderer network access is same-origin only in production. Main-process provider and
  // connector transports do not need renderer CSP exceptions. Development passes the exact
  // Vite HTTP/WebSocket origins from app-window instead of granting every loopback port.
  const connectSources = uniqueCspSources(["'self'", ...(opts.connectSources ?? [])])
  const frameSources = uniqueCspSources(opts.frameSources ?? PREVIEW_FRAME_SOURCES)
  const directives = [
    "default-src 'self'",
    // Dev loads the Vite dev server, whose React-refresh preamble is an inline script
    // (and HMR can use eval). Relax script-src in dev only; production stays strict.
    opts.dev ? "script-src 'self' 'unsafe-inline'" : "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "media-src 'self' blob:",
    "font-src 'self' data:",
    `connect-src ${connectSources.join(' ')}`,
    `frame-src ${frameSources.length ? frameSources.join(' ') : "'none'"}`,
    "frame-ancestors 'none'",
    "object-src 'none'",
    "base-uri 'none'",
  ]
  const trustedTypes = opts.trustedTypes ?? !opts.dev
  if (trustedTypes) directives.push("trusted-types default vontra", "require-trusted-types-for 'script'")
  return directives.join('; ')
}

function uniqueCspSources(sources: string[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const source of sources) {
    const clean = source.trim()
    if (!clean || seen.has(clean)) continue
    seen.add(clean)
    out.push(clean)
  }
  return out
}

export function isSecretMask(value: unknown): boolean {
  return value === SECRET_MASK || value === '[set]'
}

/**
 * A stored (masked) secret must never be sent to a renderer-chosen URL. Returns the
 * user-facing error when a masked secret is paired with a different URL than the one
 * persisted next to it; callers must refuse the request in that case.
 */
export function maskedSecretUrlError(masked: boolean, incomingUrl: unknown, storedUrl: unknown): string | null {
  const normalize = (value: unknown): string => {
    if (typeof value !== 'string') return ''
    const trimmed = value.trim()
    return trimmed.endsWith('/') && trimmed.length > 1 ? trimmed.slice(0, -1) : trimmed
  }
  const incoming = normalize(incomingUrl)
  const stored = normalize(storedUrl)
  if (!masked || !incoming || !stored || incoming === stored) return null
  return 'A saved secret can only be used with its saved URL. Re-enter the secret (API key or token) to use a different URL.'
}

export function redactSettings(input: Settings): Settings {
  return {
    ...input,
    providers: input.providers.map((provider) => ({
      ...provider,
      apiKey: provider.apiKey ? SECRET_MASK : '',
      serverPassword: provider.serverPassword ? SECRET_MASK : provider.serverPassword,
      cliEnv: provider.cliEnv ? Object.fromEntries(Object.entries(provider.cliEnv).map(([key, value]) => [key, value ? SECRET_MASK : ''])) : provider.cliEnv,
      accounts: provider.accounts?.map((account) => ({
        ...account,
        apiKey: account.apiKey ? SECRET_MASK : '',
        env: account.env ? Object.fromEntries(Object.entries(account.env).map(([key, value]) => [key, value ? SECRET_MASK : ''])) : account.env,
      })),
    })),
    email: {
      ...input.email,
      // IMAP account passwords are masked per source (never reach the renderer).
      sources: input.email.sources.map((src) => ({ ...src, password: src.password ? SECRET_MASK : src.password })),
    },
    slack: {
      ...input.slack,
      // Live Slack workspace tokens are masked per source.
      sources: input.slack.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    discord: {
      ...input.discord,
      // Discord bot tokens are masked per source.
      sources: input.discord.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    mattermost: {
      ...input.mattermost,
      // Mattermost Personal Access / bot tokens are masked per source.
      sources: input.mattermost.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    matrix: {
      ...input.matrix,
      // Matrix homeserver access tokens are masked per source.
      sources: input.matrix.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    googlechat: {
      ...input.googlechat,
      // Google Chat incoming-webhook URLs (carry an auth token in their query string) and
      // optional service-account access tokens are masked per source.
      sources: input.googlechat.sources.map((src) => ({
        ...src,
        webhookUrl: src.webhookUrl ? SECRET_MASK : src.webhookUrl,
        accessToken: src.accessToken ? SECRET_MASK : src.accessToken,
      })),
    },
    botControl: {
      ...input.botControl,
      telegram: {
        ...input.botControl.telegram,
        botToken: input.botControl.telegram.botToken ? SECRET_MASK : '',
        ...(input.botControl.telegram.remote ? { remote: { ...input.botControl.telegram.remote, sessions: [] } } : {}),
      },
      whatsapp: {
        ...input.botControl.whatsapp,
        accessToken: input.botControl.whatsapp.accessToken ? SECRET_MASK : '',
        verifyToken: input.botControl.whatsapp.verifyToken ? SECRET_MASK : '',
        appSecret: input.botControl.whatsapp.appSecret ? SECRET_MASK : '',
      },
    },
    voice: {
      ...input.voice,
      apiKey: input.voice.apiKey ? SECRET_MASK : '',
      elevenLabsApiKey: input.voice.elevenLabsApiKey ? SECRET_MASK : '',
    },
    remoteControl: {
      ...input.remoteControl,
      authToken: input.remoteControl.authToken ? SECRET_MASK : '',
    },
    remoteClient: {
      ...input.remoteClient,
      authToken: input.remoteClient.authToken ? SECRET_MASK : '',
    },
    remoteOrchestrators: input.remoteOrchestrators.map((host) => ({
      ...host,
      authToken: host.authToken ? SECRET_MASK : '',
    })),
  }
}

export function mergeSettingsSecrets(next: Settings, current: Settings): Settings {
  const byId = new Map(current.providers.map((provider) => [provider.id, provider]))
  const currentEmailSources = new Map(current.email.sources.map((src) => [src.id, src]))
  const currentSlackSources = new Map(current.slack.sources.map((src) => [src.id, src]))
  const currentDiscordSources = new Map(current.discord.sources.map((src) => [src.id, src]))
  const currentMattermostSources = new Map(current.mattermost.sources.map((src) => [src.id, src]))
  const currentMatrixSources = new Map(current.matrix.sources.map((src) => [src.id, src]))
  const currentGoogleChatSources = new Map(current.googlechat.sources.map((src) => [src.id, src]))
  return {
    ...next,
    defaultCwd: next.defaultCwd || null,
    // Packaged-plugin trust is a main-process-owned decision: a renderer-supplied
    // `trusted: true` must never load code (plugins execute via require() in main).
    // Trust carries over from the persisted settings only when the plugin id AND
    // manifest path are unchanged; any new or changed entry arrives untrusted.
    capabilities: {
      ...next.capabilities,
      packagedPlugins: (next.capabilities?.packagedPlugins ?? []).map((plugin) => {
        const storedPlugin = (current.capabilities?.packagedPlugins ?? []).find((candidate) => candidate.id === plugin.id)
        return { ...plugin, trusted: storedPlugin && storedPlugin.manifestPath === plugin.manifestPath ? (storedPlugin.trusted ?? false) : false }
      }),
    },
    providers: next.providers.map((provider) => {
      const existing = byId.get(provider.id)
      const accountsById = new Map((existing?.accounts ?? []).map((account) => [account.id, account]))
      // A masked (stored) API key must never be re-pointed at a different base URL by
      // the renderer: changing where the key is SENT requires re-entering the key, so a
      // masked key paired with a changed baseUrl is cleared instead of restored.
      const stripTrailingSlash = (value: string): string => value.trim().replace(/\/+$/, '')
      const urlChangedWithMaskedKey = !!existing
        && isSecretMask(provider.apiKey)
        && typeof provider.baseUrl === 'string' && typeof existing.baseUrl === 'string'
        && stripTrailingSlash(provider.baseUrl) !== stripTrailingSlash(existing.baseUrl)
      return {
        ...provider,
        apiKey: urlChangedWithMaskedKey ? '' : isSecretMask(provider.apiKey) ? (existing?.apiKey ?? '') : provider.apiKey,
        serverPassword: isSecretMask(provider.serverPassword) ? existing?.serverPassword : provider.serverPassword,
        cliEnv: Object.fromEntries(Object.entries(provider.cliEnv ?? {}).map(([key, value]) => [key, isSecretMask(value) ? (existing?.cliEnv?.[key] ?? '') : value])),
        accounts: provider.accounts?.map((account) => {
          const existingAccount = accountsById.get(account.id)
          return {
            ...account,
            apiKey: isSecretMask(account.apiKey) ? (existingAccount?.apiKey ?? '') : account.apiKey,
            env: Object.fromEntries(Object.entries(account.env ?? {}).map(([key, value]) => [key, isSecretMask(value) ? (existingAccount?.env?.[key] ?? '') : value])),
          }
        }),
      }
    }),
    email: {
      ...next.email,
      // Restore masked IMAP passwords from the current persisted source; keep any
      // newly-typed cleartext password the renderer sent for a new/edited account.
      sources: next.email.sources.map((src) => {
        if (!src.password || !isSecretMask(src.password)) return src
        return { ...src, password: currentEmailSources.get(src.id)?.password ?? '' }
      }),
    },
    slack: {
      ...next.slack,
      // Restore masked Slack tokens from the persisted source.
      sources: next.slack.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        return { ...src, token: currentSlackSources.get(src.id)?.token ?? '' }
      }),
    },
    discord: {
      ...next.discord,
      // Restore masked Discord tokens from the persisted source.
      sources: next.discord.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        return { ...src, token: currentDiscordSources.get(src.id)?.token ?? '' }
      }),
    },
    mattermost: {
      ...next.mattermost,
      // Restore masked Mattermost tokens from the persisted source.
      sources: next.mattermost.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        return { ...src, token: currentMattermostSources.get(src.id)?.token ?? '' }
      }),
    },
    matrix: {
      ...next.matrix,
      // Restore masked Matrix access tokens from the persisted source.
      sources: next.matrix.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        return { ...src, token: currentMatrixSources.get(src.id)?.token ?? '' }
      }),
    },
    googlechat: {
      ...next.googlechat,
      // Restore masked Google Chat webhook URLs / access tokens from the persisted source.
      sources: next.googlechat.sources.map((src) => {
        const existing = currentGoogleChatSources.get(src.id)
        return {
          ...src,
          webhookUrl: src.webhookUrl && isSecretMask(src.webhookUrl) ? (existing?.webhookUrl ?? '') : src.webhookUrl,
          accessToken: src.accessToken && isSecretMask(src.accessToken) ? (existing?.accessToken ?? '') : src.accessToken,
        }
      }),
    },
    botControl: {
      ...next.botControl,
      telegram: {
        ...next.botControl.telegram,
        botToken: isSecretMask(next.botControl.telegram.botToken) ? current.botControl.telegram.botToken : next.botControl.telegram.botToken,
      },
      whatsapp: {
        ...next.botControl.whatsapp,
        accessToken: isSecretMask(next.botControl.whatsapp.accessToken) ? current.botControl.whatsapp.accessToken : next.botControl.whatsapp.accessToken,
        verifyToken: isSecretMask(next.botControl.whatsapp.verifyToken) ? current.botControl.whatsapp.verifyToken : next.botControl.whatsapp.verifyToken,
        appSecret: isSecretMask(next.botControl.whatsapp.appSecret) ? current.botControl.whatsapp.appSecret : next.botControl.whatsapp.appSecret,
      },
    },
    voice: {
      ...next.voice,
      apiKey: isSecretMask(next.voice.apiKey) ? current.voice.apiKey : next.voice.apiKey,
      elevenLabsApiKey: isSecretMask(next.voice.elevenLabsApiKey) ? current.voice.elevenLabsApiKey : next.voice.elevenLabsApiKey,
    },
    remoteControl: {
      ...next.remoteControl,
      authToken: isSecretMask(next.remoteControl.authToken) ? current.remoteControl.authToken : next.remoteControl.authToken,
    },
    remoteClient: {
      ...next.remoteClient,
      authToken: isSecretMask(next.remoteClient.authToken) ? current.remoteClient.authToken : next.remoteClient.authToken,
    },
    remoteOrchestrators: next.remoteOrchestrators.map((host) => {
      const currentHost = current.remoteOrchestrators.find((candidate) => candidate.id === host.id)
      return {
        ...host,
        authToken: isSecretMask(host.authToken) ? (currentHost?.authToken ?? current.remoteClient.authToken ?? '') : host.authToken,
      }
    }),
  }
}

export function mergeMcpServerSecrets(next: McpServerConfig, existing?: McpServerConfig | null): McpServerConfig {
  if (!existing) return next
  const previousEnv = existing.env ?? {}
  const env = Object.fromEntries(Object.entries(next.env ?? {}).map(([key, value]) => [key, isSecretMask(value) ? (previousEnv[key] ?? '') : value]))
  return { ...next, env }
}

export function assertProviderEndpoint(provider: Provider): void {
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli' || provider.transport === 'opencode' || provider.transport === 'kilo' || provider.transport === 'pi-cli' || provider.transport === 'cursor-cli' || provider.transport === 'gemini-cli') return
  const url = new URL(provider.baseUrl)
  if (url.username || url.password) throw new Error('Provider URLs must not include embedded credentials.')
  if (isManagedComputeProvider(provider)) {
    if (url.protocol !== 'https:' && !isLoopbackUrl(url.toString())) throw new Error('Managed compute endpoints require HTTPS or a verified loopback forward.')
    if (isLinkLocalOrMetadataHost(url.hostname)) throw new Error('Managed compute endpoints cannot use metadata or link-local destinations.')
    return
  }
  if (provider.kind === 'local') {
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Local provider URLs must use http or https.')
    return
  }
  if (url.protocol !== 'https:') throw new Error('Cloud provider URLs must use https.')
  if (isPrivateHost(url.hostname)) throw new Error('Cloud provider URLs cannot point at local/private addresses.')
}

// Cloud provider guard that also resolves DNS and rejects if ANY resolved address
// is internal — closes the DNS-rebinding bypass before a Bearer key is sent.
export async function assertProviderEndpointResolved(provider: Provider): Promise<PinnedHost | null> {
  assertProviderEndpoint(provider)
  if (provider.transport === 'codex-cli' || provider.transport === 'claude-cli' || provider.transport === 'opencode' || provider.transport === 'kilo' || provider.transport === 'pi-cli' || provider.transport === 'cursor-cli' || provider.transport === 'gemini-cli') return null
  const target = await resolvePinnedHost(provider.baseUrl, (address) => {
    if (isAlwaysForbiddenConnectorIp(address)) {
      throw new Error(`Provider host resolves to a link-local, metadata, or unspecified address (${address}); refusing to send credentials.`)
    }
    if (isManagedComputeProvider(provider)) { assertRemoteOrchestratorAddress(new URL(provider.baseUrl), address); return }
    if (provider.kind === 'local') {
      if (!isPrivateIp(address)) throw new Error(`Local provider host resolves to a public address (${address}); mark public endpoints as cloud providers.`)
      return
    }
    if (!isGloballyRoutableIp(address)) throw new Error(`Cloud provider host resolves to a private or non-public address (${address}); refusing to send credentials.`)
  })
  return target
}

export async function resolveProviderRequest(provider: Provider, rawRequestUrl: string | URL): Promise<PinnedHost> {
  const configured = new URL(provider.baseUrl)
  const request = rawRequestUrl instanceof URL ? new URL(rawRequestUrl.toString()) : new URL(String(rawRequestUrl))
  if (request.origin !== configured.origin) throw new Error('Provider request origin does not match the configured provider endpoint.')
  const target = await assertProviderEndpointResolved({ ...provider, baseUrl: request.toString() })
  if (!target) throw new Error('CLI providers do not support direct HTTP requests.')
  return target
}

export interface ConnectorEndpointOptions {
  // Mattermost and Matrix are commonly self-hosted on RFC1918/loopback networks. Keep that
  // capability explicit per source so an attacker cannot silently turn an arbitrary connector
  // URL into an authenticated intranet fetch. Link-local/metadata/unspecified destinations stay
  // forbidden even when this is enabled.
  allowPrivateNetwork?: boolean
  /** Test seam for DNS-to-socket pinning. Production callers use the system resolver. */
  lookup?: PinnedHostResolver
}

// Guard for a renderer-configured live-connector origin. Public endpoints are accepted;
// private/loopback destinations require an explicit per-source opt-in because self-hosted
// Mattermost/Matrix deployments legitimately live there. Metadata/link-local targets and
// embedded URL credentials are never accepted.
export function assertConnectorBaseUrl(raw: string, label: string, options: ConnectorEndpointOptions = {}): void {
  const trimmed = String(raw || '').trim()
  if (!trimmed) return // caller already requires non-empty; nothing to validate
  let url: URL
  try { url = new URL(trimmed) } catch { throw new Error(`${label} URL is invalid.`) }
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error(`${label} URL must start with http(s)://.`)
  if (url.username || url.password) throw new Error(`${label} URL must not include credentials.`)
  if (isAlwaysForbiddenConnectorIp(url.hostname)) throw new Error(`${label} URL cannot point at a local/private/link-local/metadata or unspecified address.`)
  if (isPrivateHost(url.hostname) && !options.allowPrivateNetwork) {
    throw new Error(`${label} URL points at a local/private/loopback address. Enable private-network access for this source if it is an intentional self-hosted server.`)
  }
  if (url.protocol === 'http:' && !options.allowPrivateNetwork) {
    throw new Error(`${label} URL must use HTTPS unless private-network access is explicitly enabled for an intentional self-hosted server.`)
  }
}

// DNS-resolved half of the connector base-URL guard — closes the DNS-rebinding bypass (a
// public-looking hostname that resolves to an internal/metadata address) before the Bearer
// token is sent. Call this at the top of every live fetch, not just at config-save time:
// a URL that passed save-time validation can rebind to an internal address later.
export async function assertConnectorBaseUrlResolved(raw: string, label: string, options: ConnectorEndpointOptions = {}): Promise<PinnedHost | null> {
  assertConnectorBaseUrl(raw, label, options)
  const trimmed = String(raw || '').trim()
  if (!trimmed) return null
  const url = new URL(trimmed)
  return resolvePinnedHost(url, (address) => {
    if (isAlwaysForbiddenConnectorIp(address) || (!options.allowPrivateNetwork && !isGloballyRoutableIp(address))) {
      throw new Error(`${label} host ${url.hostname} resolves to a private or non-public address (${address}); refusing to send credentials.`)
    }
    if (url.protocol === 'http:' && !isPrivateIp(address)) {
      throw new Error(`${label} uses HTTP but resolves to a public address (${address}); public connector endpoints must use HTTPS.`)
    }
  }, { lookup: options.lookup })
}

export async function resolveConnectorRequest(
  rawBaseUrl: string,
  label: string,
  options: ConnectorEndpointOptions,
  rawRequestUrl: string | URL,
): Promise<PinnedHost> {
  const configured = new URL(String(rawBaseUrl || '').trim())
  const request = rawRequestUrl instanceof URL ? new URL(rawRequestUrl.toString()) : new URL(String(rawRequestUrl))
  if (request.origin !== configured.origin) throw new Error(`${label} request origin does not match its configured endpoint.`)
  const target = await assertConnectorBaseUrlResolved(request.toString(), label, options)
  if (!target) throw new Error(`${label} URL is required.`)
  return target
}

// OpenCode/Kilo are normally loopback/LAN services and may use HTTP there. Public remote
// servers are accepted only over HTTPS. Metadata/link-local/unspecified destinations are
// never valid, even though they otherwise fall inside the broad "private" class.
export async function assertLocalServiceUrlResolved(raw: string, label: string): Promise<PinnedHost> {
  const trimmed = String(raw || '').trim()
  let url: URL
  try { url = new URL(trimmed) } catch { throw new Error(`${label} URL is invalid.`) }
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error(`${label} URL must start with http(s)://.`)
  if (url.username || url.password) throw new Error(`${label} URL must not include credentials.`)
  return resolvePinnedHost(url, (address) => {
    if (isAlwaysForbiddenConnectorIp(address)) throw new Error(`${label} cannot target link-local, metadata, or unspecified addresses.`)
    if (url.protocol === 'http:' && !isPrivateIp(address)) throw new Error(`${label} may use HTTP only on loopback or a private network.`)
  })
}

export async function resolveLocalServiceRequest(rawBaseUrl: string, label: string, rawRequestUrl: string | URL): Promise<PinnedHost> {
  const configured = new URL(String(rawBaseUrl || '').trim())
  const request = rawRequestUrl instanceof URL ? new URL(rawRequestUrl.toString()) : new URL(String(rawRequestUrl))
  if (request.origin !== configured.origin) throw new Error(`${label} request origin does not match its configured endpoint.`)
  return assertLocalServiceUrlResolved(request.toString(), label)
}

// Guard for "local provider" discovery (ollama/lmstudio/omlx). The handler sends the
// renderer-supplied apiKey as a Bearer token to baseUrl/models — so an attacker-controlled
// public baseUrl would exfiltrate the key. Local discovery only ever legitimately targets
// loopback or the LAN (networked oMLX), so require the host to be private/loopback. Literal
// private hosts pass without DNS; a real hostname is resolved and every address must be
// private (fail closed). An empty baseUrl uses the built-in localhost default — nothing to
// validate.
export async function assertLocalProviderDiscoveryUrl(raw: string): Promise<PinnedHost | null> {
  const trimmed = String(raw || '').trim()
  if (!trimmed) return null
  let url: URL
  try { url = new URL(trimmed) } catch { throw new Error('Local provider URL is invalid.') }
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Local provider URLs must use http or https.')
  if (url.username || url.password) throw new Error('Local provider URLs must not include credentials.')
  return resolvePinnedHost(url, (address) => {
    if (isAlwaysForbiddenConnectorIp(address)) throw new Error(`Local provider host ${url.hostname} resolves to a link-local, metadata, or unspecified address; refusing to connect.`)
    if (!isPrivateIp(address)) throw new Error(`Local provider host ${url.hostname} resolves to a non-private address; refusing to send credentials to it.`)
  })
}

export async function resolveLocalProviderRequest(rawBaseUrl: string, rawRequestUrl: string | URL): Promise<PinnedHost> {
  const configured = new URL(String(rawBaseUrl || '').trim())
  const request = rawRequestUrl instanceof URL ? new URL(rawRequestUrl.toString()) : new URL(String(rawRequestUrl))
  if (request.origin !== configured.origin) throw new Error('Local provider request origin does not match its configured endpoint.')
  const target = await assertLocalProviderDiscoveryUrl(request.toString())
  if (!target) throw new Error('Local provider URL is required.')
  return target
}

export async function assertRemoteOrchestratorUrl(raw: string): Promise<string> {
  return (await resolveRemoteOrchestratorUrl(raw)).baseUrl
}

/** Validate and canonicalize a configured Orchestrator URL without performing DNS. */
export function normalizeRemoteOrchestratorUrl(raw: string): string {
  return normalizedRemoteOrchestratorBase(raw).toString().replace(/\/+$/, '')
}

function normalizedRemoteOrchestratorBase(raw: string): URL {
  const trimmed = String(raw || '').trim()
  if (!trimmed) throw new Error('Remote Orchestrator URL is required.')
  const url = new URL(trimmed)
  if (url.username || url.password) throw new Error('Remote Orchestrator URLs must not include credentials.')
  if (url.search || url.hash) throw new Error('Remote Orchestrator URLs must not include query strings or fragments.')
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Remote Orchestrator URLs must use http or https.')
  if (url.protocol === 'http:' && !isLoopbackHost(url.hostname)) {
    throw new Error('Remote Orchestrator URLs must use https, except loopback URLs on this machine.')
  }
  if (isLinkLocalOrMetadataHost(url.hostname)) {
    throw new Error('Remote Orchestrator URLs cannot point at link-local or metadata addresses.')
  }
  url.pathname = url.pathname.replace(/\/+$/, '')
  return url
}

function assertRemoteOrchestratorAddress(url: URL, address: string): void {
  const literalPrivate = isPrivateHost(url.hostname)
  const loopbackName = isLoopbackHost(url.hostname)
  if (isAlwaysForbiddenConnectorIp(address)) throw new Error(`Remote Orchestrator host ${url.hostname} resolves to a link-local, metadata, or unspecified address (${address}).`)
  if (url.protocol === 'http:' && !isLoopbackIp(address)) throw new Error('Remote Orchestrator HTTP endpoints must resolve only to loopback addresses.')
  if (url.protocol === 'https:' && loopbackName && !isLoopbackIp(address)) throw new Error('Remote Orchestrator loopback names must resolve only to loopback addresses.')
  // HTTPS endpoints are explicit user-configured trust roots and must support Tailscale,
  // MagicDNS, and LAN names. Strict TLS hostname verification plus socket pinning protects
  // the bearer token; metadata/link-local destinations remain unconditionally forbidden.
}

export async function resolveRemoteOrchestratorUrl(raw: string, lookup?: PinnedHostResolver): Promise<{ baseUrl: string; target: PinnedHost }> {
  const url = normalizedRemoteOrchestratorBase(raw)
  const baseUrl = normalizeRemoteOrchestratorUrl(raw)
  const target = await resolvePinnedHost(baseUrl, (address) => assertRemoteOrchestratorAddress(url, address), { lookup })
  return { baseUrl, target }
}

export async function resolveRemoteOrchestratorRequest(rawBaseUrl: string, rawRequestUrl: string | URL): Promise<{ baseUrl: string; target: PinnedHost }> {
  const configured = normalizedRemoteOrchestratorBase(rawBaseUrl)
  const baseUrl = configured.toString().replace(/\/+$/, '')
  const request = rawRequestUrl instanceof URL ? new URL(rawRequestUrl.toString()) : new URL(String(rawRequestUrl))
  if (request.origin !== configured.origin) throw new Error('Remote Orchestrator request origin does not match its configured endpoint.')
  const target = await resolvePinnedHost(request, (address) => assertRemoteOrchestratorAddress(configured, address))
  return { baseUrl, target }
}

export function isSafeExternalUrl(raw: string, opts: { allowHttpLocalhost?: boolean } = {}): boolean {
  try {
    const url = new URL(raw)
    if (url.protocol === 'https:' || url.protocol === 'mailto:') return true
    if (opts.allowHttpLocalhost && url.protocol === 'http:' && isLoopbackHost(url.hostname)) return true
    return false
  } catch {
    return false
  }
}

// Outbound webhook SSRF guard, DNS-resolved. isSafeExternalUrl only enforces the scheme
// (https, or http+loopback for local dev); isPrivateHost alone only inspects the literal
// hostname STRING, so a public name whose DNS record points (or is later rebound, TTL=0)
// at an internal/ULA/link-local address would sail through. This layers the same
// DNS-resolved isPrivateIp check used elsewhere (assertCacheableUrl, assertProviderEndpointResolved)
// on top: loopback literals are allowed without DNS (documented localhost-dev case), any
// other private host literal is rejected outright, and everything else is resolved with EVERY
// returned address checked. A resolver FAILURE itself is not fatal here (mirrors
// assertProviderEndpointResolved, not assertCacheableUrl) — delivery goes through its own
// fetch() right after this, which will itself fail/retry on a bad host, so we don't add a
// hard dependency on DNS in the validator. Call this at BOTH webhook save/validate time and
// EVERY delivery time — delivery-time is what actually defends against rebinding, since the
// hostname can re-resolve differently between approval and send.
export async function assertOutboundWebhookUrlResolved(raw: string): Promise<PinnedHost> {
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw new Error('Outbound webhook URL is invalid.')
  }
  if (url.username || url.password) throw new Error('Outbound webhook URL must not include credentials.')
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && isLoopbackHost(url.hostname))) {
    throw new Error('Outbound webhook URL must use HTTPS, except localhost during development.')
  }
  const host = url.hostname
  if (isPrivateHost(host) && !isLoopbackHost(host)) {
    throw new Error('Outbound webhook URL must not target a private or internal address.')
  }
  return resolvePinnedHost(url, (address) => {
    if (url.protocol === 'http:' && !isLoopbackIp(address)) throw new Error('Outbound webhook localhost URLs must resolve only to loopback addresses.')
    if (!isGloballyRoutableIp(address)) {
      if (url.protocol === 'http:' && isLoopbackIp(address)) return
      throw new Error(`Outbound webhook host ${host} resolves to a private or non-public address (${address}); refusing to deliver to it.`)
    }
  })
}

export function assertUpdateFeedUrl(raw: string): void {
  const url = new URL(raw)
  if (url.protocol === 'https:') return
  if (url.protocol === 'http:' && isLoopbackHost(url.hostname)) return
  throw new Error('Update feed URL must use https, except localhost during development.')
}

function redactAuditPreviewText(input: string): string {
  const structured = input
    .replace(/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, SECRET_MASK)
    .replace(/\bBearer\s+[A-Za-z0-9._~+/-]{12,}=*/g, `Bearer ${SECRET_MASK}`)
    // HTTP Basic credentials (base64 user:pass) and cookie headers carry reusable
    // credentials the Bearer/key-value rules never match.
    .replace(/\bAuthorization\s*:\s*Basic\s+[A-Za-z0-9+/=]{8,}/gi, `Authorization: Basic ${SECRET_MASK}`)
    .replace(/\bCookie\s*:\s*[^\r\n]{4,}/gi, `Cookie: ${SECRET_MASK}`)
    // Presigned-URL query credentials (S3/CDN Signature, X-Amz-Signature, key/token
    // params): the hex value is deliberately excluded from opaque-secret detection as a
    // presumed checksum, so it must be caught by its key name instead.
    .replace(/([?&](?:[A-Za-z0-9_-]*[Ss]ignature|[Xx]-[Aa]mz-[Ss]ignature|[Aa]uth|[Kk]ey|[Tt]oken)=)[A-Za-z0-9%._~+/=-]{8,}/g, `$1${SECRET_MASK}`)
    .replace(/\b(?:ghp|gho|ghu|ghs|ghr|github_pat|sk|xox[baprs]|slack|discord|telegram|whatsapp|npm)_[A-Za-z0-9_=-]{12,}\b/g, SECRET_MASK)
    .replace(/(?:[A-Za-z0-9]*[_-])?(?:api[_-]?key|secret|token|password|passwd|pwd|access[_-]?token|auth[_-]?token|client[_-]?secret|private[_-]?key)\b\s*['"]?\s*[=:]\s*['"]?[^\s'",;)}\]]{6,}/gi, (match) => {
      const sep = match.search(/[=:]/)
      return sep >= 0 ? `${match.slice(0, sep + 1)}${SECRET_MASK}` : SECRET_MASK
    })
    .replace(/\b[a-z][a-z0-9+.-]*:\/\/([^\s:@/]+):([^\s:@/]{3,})@/gi, (match) => match.replace(/:[^\s:@/]{3,}@/, `:${SECRET_MASK}@`))
  return redactOpaqueSecrets(structured, SECRET_MASK)
}

export function auditPreview(value: unknown, max = 600): string {
  return redactAuditPreviewText(String(value ?? '').replace(/\s+/g, ' ').trim()).slice(0, max)
}
