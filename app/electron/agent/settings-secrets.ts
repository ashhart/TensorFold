import type { EmailSourceConfig, Settings, McpServerConfig } from './types'

export const SECRET_MASK = '••••••••'

export function isSecretMask(value: unknown): boolean {
  return value === SECRET_MASK || value === '[set]'
}

function normalizedEndpoint(value: unknown): string {
  return typeof value === 'string' ? value.trim().replace(/\/+$/, '') : ''
}

function sameEndpoint(incoming: unknown, stored: unknown): boolean {
  return normalizedEndpoint(incoming) === normalizedEndpoint(stored)
}

function boundSecret(incoming: string, stored: string | undefined, sameDestination: boolean): string
function boundSecret(incoming: string | undefined, stored: string | undefined, sameDestination: boolean): string | undefined
function boundSecret(incoming: string | undefined, stored: string | undefined, sameDestination: boolean): string | undefined {
  if (!isSecretMask(incoming)) return incoming
  return sameDestination ? stored ?? '' : ''
}

function sameMailEndpoint(incoming: EmailSourceConfig, stored?: EmailSourceConfig): boolean {
  if (!stored || incoming.kind !== stored.kind) return false
  const secure = incoming.tls !== false, previousSecure = stored.tls !== false
  return incoming.host?.trim().toLowerCase() === stored.host?.trim().toLowerCase()
    && secure === previousSecure
    && (incoming.port ?? (secure ? 993 : 143)) === (stored.port ?? (previousSecure ? 993 : 143))
    && incoming.user?.trim() === stored.user?.trim()
}

export function maskedSecretUrlError(masked: boolean, incomingUrl: unknown, storedUrl: unknown): string | null {
  if (!masked || !normalizedEndpoint(incomingUrl) || !normalizedEndpoint(storedUrl) || sameEndpoint(incomingUrl, storedUrl)) return null
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
      sources: input.email.sources.map((src) => ({ ...src, password: src.password ? SECRET_MASK : src.password })),
    },
    slack: {
      ...input.slack,
      sources: input.slack.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    discord: {
      ...input.discord,
      sources: input.discord.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    mattermost: {
      ...input.mattermost,
      sources: input.mattermost.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    matrix: {
      ...input.matrix,
      sources: input.matrix.sources.map((src) => ({ ...src, token: src.token ? SECRET_MASK : src.token })),
    },
    googlechat: {
      ...input.googlechat,
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
      const sameApi = !!existing && sameEndpoint(provider.baseUrl, existing.baseUrl)
      const sameServer = !!existing && sameEndpoint(provider.serverUrl, existing.serverUrl)
      return {
        ...provider,
        apiKey: boundSecret(provider.apiKey, existing?.apiKey, sameApi),
        serverPassword: boundSecret(provider.serverPassword, existing?.serverPassword, sameServer),
        cliEnv: Object.fromEntries(Object.entries(provider.cliEnv ?? {}).map(([key, value]) => [key, isSecretMask(value) ? (existing?.cliEnv?.[key] ?? '') : value])),
        accounts: provider.accounts?.map((account) => {
          const existingAccount = accountsById.get(account.id)
          return {
            ...account,
            apiKey: boundSecret(account.apiKey, existingAccount?.apiKey, sameApi),
            env: Object.fromEntries(Object.entries(account.env ?? {}).map(([key, value]) => [key, isSecretMask(value) ? (existingAccount?.env?.[key] ?? '') : value])),
          }
        }),
      }
    }),
    email: {
      ...next.email,
      sources: next.email.sources.map((src) => {
        if (!src.password || !isSecretMask(src.password)) return src
        const stored = currentEmailSources.get(src.id)
        return { ...src, password: boundSecret(src.password, stored?.password, sameMailEndpoint(src, stored)) }
      }),
    },
    slack: {
      ...next.slack,
      sources: next.slack.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        return { ...src, token: currentSlackSources.get(src.id)?.token ?? '' }
      }),
    },
    discord: {
      ...next.discord,
      sources: next.discord.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        return { ...src, token: currentDiscordSources.get(src.id)?.token ?? '' }
      }),
    },
    mattermost: {
      ...next.mattermost,
      sources: next.mattermost.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        const stored = currentMattermostSources.get(src.id)
        return { ...src, token: boundSecret(src.token, stored?.token, !!stored && sameEndpoint(src.baseUrl, stored.baseUrl)) }
      }),
    },
    matrix: {
      ...next.matrix,
      sources: next.matrix.sources.map((src) => {
        if (!src.token || !isSecretMask(src.token)) return src
        const stored = currentMatrixSources.get(src.id)
        return { ...src, token: boundSecret(src.token, stored?.token, !!stored && sameEndpoint(src.baseUrl, stored.baseUrl)) }
      }),
    },
    googlechat: {
      ...next.googlechat,
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
      apiKey: boundSecret(next.voice.apiKey, current.voice.apiKey, sameEndpoint(next.voice.apiBaseUrl, current.voice.apiBaseUrl)),
      elevenLabsApiKey: isSecretMask(next.voice.elevenLabsApiKey) ? current.voice.elevenLabsApiKey : next.voice.elevenLabsApiKey,
    },
    remoteControl: {
      ...next.remoteControl,
      authToken: isSecretMask(next.remoteControl.authToken) ? current.remoteControl.authToken : next.remoteControl.authToken,
    },
    remoteClient: {
      ...next.remoteClient,
      authToken: boundSecret(next.remoteClient.authToken, current.remoteClient.authToken,
        sameEndpoint(next.remoteClient.serverUrl, current.remoteClient.serverUrl)),
    },
    remoteOrchestrators: next.remoteOrchestrators.map((host) => {
      const currentHost = current.remoteOrchestrators.find((candidate) => candidate.id === host.id)
      return {
        ...host,
        authToken: boundSecret(host.authToken, currentHost?.authToken,
          !!currentHost && sameEndpoint(host.serverUrl, currentHost.serverUrl)),
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
