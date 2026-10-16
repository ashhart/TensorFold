// Latest-version feeds for the built-in CLI launchers (VST-037). The host fetches each
// CLI's latest published version out-of-band (gated by the user's update-check setting),
// caches it, and passes it into `cliToolStatuses` so `versionAdvisory()` can render
// "update available" in the readiness doctor and provider tools panel.
//
// Only npm-distributed CLIs we can check confidently get a feed. CLIs distributed outside
// npm (cursor-agent's install script, kilo, hermes, pi, grok) have NO feed — absent means
// the doctor says "update status unknown" instead of guessing wrong.

export interface CliUpdateFeed {
  npmPackage: string
  updateCommand: string
}

export const CLI_UPDATE_FEEDS: Record<string, CliUpdateFeed> = {
  codex: { npmPackage: '@openai/codex', updateCommand: 'npm install -g @openai/codex' },
  claude: { npmPackage: '@anthropic-ai/claude-code', updateCommand: 'npm install -g @anthropic-ai/claude-code' },
  gemini: { npmPackage: '@google/gemini-cli', updateCommand: 'npm install -g @google/gemini-cli' },
  opencode: { npmPackage: 'opencode-ai', updateCommand: 'npm install -g opencode-ai' },
}

export interface CliLatestVersion {
  latestVersion: string
  updateCommand: string
}

const MAX_NPM_METADATA_BYTES = 64 * 1024

// Minimal fetch shape so tests inject a fake and never hit the network.
export type FetchLike = (url: string, init?: { signal?: AbortSignal; headers?: Record<string, string> }) => Promise<{
  ok: boolean
  status: number
  json(): Promise<any>
  headers?: { get(name: string): string | null }
  body?: ReadableStream<Uint8Array> | null
}>

// npm registry metadata endpoint. Scoped packages keep the `@` but encode the slash
// (`@openai/codex` → `@openai%2Fcodex`), matching what the registry expects.
export function npmLatestUrl(pkg: string): string {
  return `https://registry.npmjs.org/${pkg.replace('/', '%2F')}/latest`
}

async function cancelBody(res: Awaited<ReturnType<FetchLike>>): Promise<void> {
  try { await res.body?.cancel() } catch { /* best-effort socket cleanup */ }
}

async function boundedRegistryJson(res: Awaited<ReturnType<FetchLike>>): Promise<any> {
  // Test doubles and older fetch shims may expose only json(); production Response objects
  // always have headers + a streaming body and take the bounded path below.
  if (!res.headers || !res.body) return res.json()
  const contentType = String(res.headers.get('content-type') ?? '').toLowerCase()
  if (!/^(?:application\/json|application\/[a-z0-9.+-]+\+json)\b/.test(contentType)) {
    await cancelBody(res)
    throw new Error('npm registry returned a non-JSON response.')
  }
  const rawLength = res.headers.get('content-length')
  if (rawLength) {
    const length = Number(rawLength)
    if (!Number.isSafeInteger(length) || length < 0 || length > MAX_NPM_METADATA_BYTES) {
      await cancelBody(res)
      throw new Error('npm registry metadata exceeds the response-size limit.')
    }
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let text = ''
  let bytes = 0
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      bytes += value.byteLength
      if (bytes > MAX_NPM_METADATA_BYTES) {
        await reader.cancel().catch(() => {})
        throw new Error('npm registry metadata exceeds the response-size limit.')
      }
      text += decoder.decode(value, { stream: true })
    }
    text += decoder.decode()
  } finally {
    reader.releaseLock()
  }
  return JSON.parse(text)
}

export async function latestNpmVersion(pkg: string, fetchImpl: FetchLike, signal?: AbortSignal): Promise<string | null> {
  try {
    const res = await fetchImpl(npmLatestUrl(pkg), { signal, headers: { Accept: 'application/json' } })
    if (!res.ok) { await cancelBody(res); return null }
    const json = await boundedRegistryJson(res)
    const version = json?.version
    return typeof version === 'string' && version.trim() ? version.trim() : null
  } catch {
    return null
  }
}

export interface FetchCliLatestOptions {
  feeds?: Record<string, CliUpdateFeed>
  timeoutMs?: number
}

// Fetch every feed in parallel with per-feed isolation: one unreachable registry entry
// yields nothing for that CLI and never sinks the others.
export async function fetchCliLatestVersions(fetchImpl: FetchLike, options: FetchCliLatestOptions = {}): Promise<Record<string, CliLatestVersion>> {
  const feeds = options.feeds ?? CLI_UPDATE_FEEDS
  const timeoutMs = options.timeoutMs ?? 8000
  const entries = await Promise.all(Object.entries(feeds).map(async ([cliId, feed]) => {
    const signal = typeof AbortSignal?.timeout === 'function' ? AbortSignal.timeout(timeoutMs) : undefined
    const latestVersion = await latestNpmVersion(feed.npmPackage, fetchImpl, signal)
    return latestVersion ? [cliId, { latestVersion, updateCommand: feed.updateCommand }] as const : null
  }))
  return Object.fromEntries(entries.filter((entry): entry is readonly [string, CliLatestVersion] => entry != null))
}
