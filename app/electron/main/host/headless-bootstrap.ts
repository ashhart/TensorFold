import type { RemoteControlSettings } from '../../agent/types'
import { stampSecretRotations } from '../../agent/secret-rotation'

export const HEADLESS_BIND_HOST_ENV = 'VONTRA_HEADLESS_BIND_HOST'
export const HEADLESS_PORT_ENV = 'VONTRA_HEADLESS_PORT'
export const HEADLESS_BOOTSTRAP_ENV = 'VONTRA_HEADLESS_BOOTSTRAP'

export interface HeadlessBootstrapResult {
  remoteControl: RemoteControlSettings
  changed: boolean
  revealCredentials: boolean
  resetPairingSessions: boolean
  completionMarker: number | null
}

/**
 * Terminate a headless process after bootstrap failed. Electron's `app.quit()` can
 * overwrite a previously assigned `process.exitCode` and exit successfully; passing
 * the status directly to `app.exit()` is the only reliable supervisor-facing signal.
 * Keep the small adapter injectable so the non-zero contract has a regression test.
 */
export function terminateFailedHeadlessBoot(exit: (code: number) => void): void {
  process.exitCode = 1
  exit(1)
}

export function isStrongRemoteControlToken(value: unknown): boolean {
  const token = String(value ?? '').trim()
  if (!/^[A-Za-z0-9_-]{43,}$/.test(token)) return false
  try { return Buffer.from(token, 'base64url').byteLength >= 32 } catch { return false }
}

export function prepareHeadlessRemoteControl(
  current: RemoteControlSettings,
  options: {
    env?: Partial<Record<string, string | undefined>>
    now?: () => number
    newToken: () => string
  },
): HeadlessBootstrapResult {
  const env = options.env ?? process.env
  const now = options.now ?? (() => Date.now())
  const forceBootstrap = envEnabled(env[HEADLESS_BOOTSTRAP_ENV])
  const bindHost = parseBindHost(env[HEADLESS_BIND_HOST_ENV])
  const port = parsePort(env[HEADLESS_PORT_ENV], current.port)
  const firstBootstrap = !Number.isFinite(current.headlessBootstrappedAt)
  const tokenMissingOrWeak = !isStrongRemoteControlToken(current.authToken)
  // The first headless launch always replaces inherited/default configuration with a token
  // freshly generated for this explicit bootstrap; length checks alone cannot prove entropy.
  const rotateToken = firstBootstrap || forceBootstrap || tokenMissingOrWeak
  const authToken = rotateToken ? options.newToken() : current.authToken
  if (!isStrongRemoteControlToken(authToken)) throw new Error('Headless bootstrap token generator did not return at least 256 bits of base64url entropy.')

  const revealCredentials = firstBootstrap || rotateToken
  const completionMarker = revealCredentials ? now() : null
  let next: RemoteControlSettings = {
    ...current,
    enabled: true,
    bindHost,
    allowLan: false,
    port,
    authToken,
    // A failed bind must not consume the one-time bootstrap. Clear/leave the marker until
    // the listener, pairing-link persistence, and final settings write all succeed.
    headlessBootstrappedAt: revealCredentials ? null : current.headlessBootstrappedAt,
  }
  if (rotateToken) next = stampSecretRotations(next, current, { authToken: [next.authToken, current.authToken] }, now())
  return {
    remoteControl: next,
    changed: !sameBootstrapSettings(current, next),
    revealCredentials,
    resetPairingSessions: forceBootstrap,
    completionMarker,
  }
}

export function headlessBootstrapMessages(input: {
  url: string
  token?: string | null
  pairingCredential?: string | null
  pairingExpiresAt?: number | null
}): string[] {
  const url = new URL(input.url)
  const port = url.port || (url.protocol === 'https:' ? '443' : '80')
  const lines = [
    `[vontra:headless] Orchestrator listening at ${url.href.replace(/\/$/, '')}`,
    `[vontra:headless] Remote machine access (loopback tunnel): ssh -N -L ${port}:127.0.0.1:${port} user@host`,
  ]
  if (input.token && input.pairingCredential) {
    const pairUrl = new URL(url)
    pairUrl.pathname = '/'
    pairUrl.search = ''
    pairUrl.hash = ''
    // URL fragments are not sent in HTTP requests, access logs, or proxy request targets.
    // The browser redeems this one-time credential after the document loads.
    pairUrl.hash = `pair=${encodeURIComponent(input.pairingCredential)}`
    lines.push(
      '[vontra:headless] ONE-TIME bootstrap credentials (these will not be printed on later starts):',
      `[vontra:headless] Host bearer token: ${input.token}`,
      `[vontra:headless] Browser pairing link: ${pairUrl.href}`,
      `[vontra:headless] Pairing link expires at ${new Date(input.pairingExpiresAt ?? 0).toISOString()}; store the host token in a secret manager.`,
    )
  } else {
    lines.push(
      '[vontra:headless] Bootstrap secrets were already issued and are not being reprinted. Use an existing paired device.',
      `[vontra:headless] To rotate credentials deliberately: ${HEADLESS_BOOTSTRAP_ENV}=1 npm run start:headless`,
    )
  }
  return lines
}

function parseBindHost(value: string | undefined): RemoteControlSettings['bindHost'] {
  const raw = String(value ?? '').trim()
  if (!raw || raw === '127.0.0.1') return '127.0.0.1'
  throw new Error(`${HEADLESS_BIND_HOST_ENV} must be 127.0.0.1; the built-in HTTP server is loopback-only. Use an SSH tunnel or TLS front-end for remote access.`)
}

function parsePort(value: string | undefined, fallback: number): number {
  const raw = String(value ?? '').trim()
  if (!raw) return Number.isInteger(fallback) && fallback >= 1 && fallback <= 65_535 ? fallback : 8_765
  const port = Number(raw)
  if (!Number.isInteger(port) || port < 1 || port > 65_535) throw new Error(`${HEADLESS_PORT_ENV} must be an integer from 1 to 65535.`)
  return port
}

function envEnabled(value: string | undefined): boolean {
  return ['1', 'true', 'yes'].includes(String(value ?? '').toLowerCase())
}

function sameBootstrapSettings(a: RemoteControlSettings, b: RemoteControlSettings): boolean {
  return a.enabled === b.enabled &&
    a.bindHost === b.bindHost &&
    a.allowLan === b.allowLan &&
    a.port === b.port &&
    a.authToken === b.authToken &&
    a.headlessBootstrappedAt === b.headlessBootstrappedAt
}
