import type { SecretRotationMetadata, Settings, WebhookEndpoint } from './types'

export const SECRET_STALE_AFTER_MS = 90 * 24 * 60 * 60_000

type Carrier = { secretRotations?: SecretRotationMetadata }

function secret(value: unknown): string { return typeof value === 'string' ? value : '' }

export function stampSecretRotations<T extends Carrier>(
  next: T,
  current: T | null | undefined,
  slots: Record<string, [unknown, unknown]>,
  now = Date.now(),
): T {
  // Renderer/client input cannot backdate or refresh credential age. Only the authoritative
  // current carrier is trusted; a real value transition below stamps the new server time.
  const rotations: SecretRotationMetadata = { ...(current?.secretRotations ?? {}) }
  for (const [slot, [nextValue, currentValue]] of Object.entries(slots)) {
    const after = secret(nextValue)
    const before = secret(currentValue)
    if (after !== before) rotations[slot] = after ? now : null
    else if (!Object.prototype.hasOwnProperty.call(rotations, slot) && after) rotations[slot] = null
  }
  return { ...next, secretRotations: rotations }
}

function byId<T extends { id: string }>(items: T[] | undefined): Map<string, T> {
  return new Map((items ?? []).map((item) => [item.id, item]))
}

/** Stamp every Settings-carried secret after masked values have been restored. */
export function stampSettingsSecretRotations(next: Settings, current: Settings, now = Date.now()): Settings {
  const currentProviders = byId(current.providers)
  const providers = next.providers.map((provider) => {
    const prior = currentProviders.get(provider.id)
    const providerSlots: Record<string, [unknown, unknown]> = {
      apiKey: [provider.apiKey, prior?.apiKey],
      serverPassword: [provider.serverPassword, prior?.serverPassword],
    }
    for (const key of new Set([...Object.keys(provider.cliEnv ?? {}), ...Object.keys(prior?.cliEnv ?? {})])) {
      providerSlots[`cliEnv:${key}`] = [provider.cliEnv?.[key], prior?.cliEnv?.[key]]
    }
    const currentAccounts = byId(prior?.accounts)
    const accounts = provider.accounts?.map((account) => {
      const old = currentAccounts.get(account.id)
      const slots: Record<string, [unknown, unknown]> = { apiKey: [account.apiKey, old?.apiKey] }
      for (const key of new Set([...Object.keys(account.env ?? {}), ...Object.keys(old?.env ?? {})])) slots[`env:${key}`] = [account.env?.[key], old?.env?.[key]]
      return stampSecretRotations(account, old, slots, now)
    })
    return stampSecretRotations({ ...provider, accounts }, prior, providerSlots, now)
  })

  const stampSources = <T extends Carrier & { id: string }>(items: T[], oldItems: T[], values: (item: T, old?: T) => Record<string, [unknown, unknown]>) => {
    const old = byId(oldItems)
    return items.map((item) => stampSecretRotations(item, old.get(item.id), values(item, old.get(item.id)), now))
  }

  // Baileys credentials are host-owned encrypted state, not renderer-carried secret values.
  // Preserve their host-authored age marker across ordinary label/enabled/status settings edits;
  // only the protected auth adapter's real `creds.update` persistence path may advance it.
  const currentWhatsAppSources = byId(current.whatsapp.sources)
  const whatsappSources = next.whatsapp.sources.map((source) => ({
    ...source,
    credentialRotatedAt: currentWhatsAppSources.get(source.id)?.credentialRotatedAt ?? null,
  }))

  return {
    ...next,
    providers,
    whatsapp: { ...next.whatsapp, sources: whatsappSources },
    email: { ...next.email, sources: stampSources(next.email.sources, current.email.sources, (item, old) => ({ password: [item.password, old?.password] })) },
    slack: { ...next.slack, sources: stampSources(next.slack.sources, current.slack.sources, (item, old) => ({ token: [item.token, old?.token] })) },
    discord: { ...next.discord, sources: stampSources(next.discord.sources, current.discord.sources, (item, old) => ({ token: [item.token, old?.token] })) },
    mattermost: { ...next.mattermost, sources: stampSources(next.mattermost.sources, current.mattermost.sources, (item, old) => ({ token: [item.token, old?.token] })) },
    matrix: { ...next.matrix, sources: stampSources(next.matrix.sources, current.matrix.sources, (item, old) => ({ token: [item.token, old?.token] })) },
    googlechat: { ...next.googlechat, sources: stampSources(next.googlechat.sources, current.googlechat.sources, (item, old) => ({ webhookUrl: [item.webhookUrl, old?.webhookUrl], accessToken: [item.accessToken, old?.accessToken] })) },
    botControl: {
      telegram: stampSecretRotations(next.botControl.telegram, current.botControl.telegram, { botToken: [next.botControl.telegram.botToken, current.botControl.telegram.botToken] }, now),
      whatsapp: stampSecretRotations(next.botControl.whatsapp, current.botControl.whatsapp, {
        accessToken: [next.botControl.whatsapp.accessToken, current.botControl.whatsapp.accessToken],
        verifyToken: [next.botControl.whatsapp.verifyToken, current.botControl.whatsapp.verifyToken],
        appSecret: [next.botControl.whatsapp.appSecret, current.botControl.whatsapp.appSecret],
      }, now),
    },
    voice: stampSecretRotations(next.voice, current.voice, { apiKey: [next.voice.apiKey, current.voice.apiKey], elevenLabsApiKey: [next.voice.elevenLabsApiKey, current.voice.elevenLabsApiKey] }, now),
    remoteControl: stampSecretRotations(next.remoteControl, current.remoteControl, { authToken: [next.remoteControl.authToken, current.remoteControl.authToken] }, now),
    remoteClient: stampSecretRotations(next.remoteClient, current.remoteClient, { authToken: [next.remoteClient.authToken, current.remoteClient.authToken] }, now),
    remoteOrchestrators: stampSources(next.remoteOrchestrators, current.remoteOrchestrators, (item, old) => ({ authToken: [item.authToken, old?.authToken] })),
  }
}

export function stampWebhookSecretRotation(next: WebhookEndpoint, current: WebhookEndpoint | null | undefined, now = Date.now()): WebhookEndpoint {
  return stampSecretRotations(next, current, { secretRef: [next.secretRef, current?.secretRef] }, now)
}

export type SecretFreshness = 'missing' | 'unknown' | 'fresh' | 'stale'
export function secretFreshness(value: unknown, rotatedAt: number | null | undefined, now = Date.now()): SecretFreshness {
  if (!secret(value)) return 'missing'
  if (!Number.isFinite(rotatedAt)) return 'unknown'
  return now - Number(rotatedAt) > SECRET_STALE_AFTER_MS ? 'stale' : 'fresh'
}
