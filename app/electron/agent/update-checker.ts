import { verify } from 'node:crypto'
import type { AppUpdateChecksum, AppUpdateStatus } from './types'
import { resolveFetchableUrl } from './tool-web'
import { discardResponseBody, pinnedFetch, type PinnedHostResolver } from './pinned-fetch'
import { assertUpdateFeedUrl } from './security'
import { updateTrustRootPublicKey } from './update-trust'
import { compareSemver, parseSemver } from './cli-version-advisory'

interface ParsedFeed {
  version: string
  releaseName: string | null
  releaseNotes: string | null
  downloadUrl: string | null
  checksum: AppUpdateChecksum | null
  signature: string | null
  signaturePayload: string
  automaticUpdate: AppUpdateStatus['automaticUpdate']
}

export const MAX_UPDATE_MANIFEST_BYTES = 512 * 1024

function supportedManifestContentType(contentType: string): boolean {
  const mime = contentType.split(';', 1)[0].trim().toLowerCase()
  return mime === 'application/json' || mime.endsWith('+json') || mime === 'application/rss+xml' || mime === 'application/xml' || mime === 'text/xml'
}

async function readManifestResponse(response: Response): Promise<{ text: string; contentType: string }> {
  const contentType = response.headers.get('content-type') ?? ''
  if (!supportedManifestContentType(contentType)) {
    await discardResponseBody(response)
    throw new Error(`Update feed returned unsupported Content-Type ${contentType || '(missing)'}`)
  }
  const declared = response.headers.get('content-length')
  if (declared != null) {
    if (!/^\d+$/.test(declared.trim()) || Number(declared) > MAX_UPDATE_MANIFEST_BYTES) {
      await discardResponseBody(response)
      throw new Error(`Update manifest exceeds the ${MAX_UPDATE_MANIFEST_BYTES}-byte limit`)
    }
  }
  if (!response.body) throw new Error('Update feed returned an empty manifest body')
  const reader = response.body.getReader()
  const chunks: Uint8Array[] = []
  let bytes = 0
  try {
    for (;;) {
      const item = await reader.read()
      if (item.done) break
      bytes += item.value.byteLength
      if (bytes > MAX_UPDATE_MANIFEST_BYTES) {
        await reader.cancel('manifest size limit exceeded').catch(() => {})
        throw new Error(`Update manifest exceeds the ${MAX_UPDATE_MANIFEST_BYTES}-byte limit`)
      }
      chunks.push(item.value)
    }
  } catch (error) {
    await reader.cancel().catch(() => {})
    throw error
  }
  return { text: Buffer.concat(chunks.map((chunk) => Buffer.from(chunk))).toString('utf8'), contentType }
}

export function emptyUpdateStatus(currentVersion: string, feedUrl: string | null = null): AppUpdateStatus {
  return {
    state: feedUrl ? 'not_checked' : 'not_configured',
    currentVersion,
    latestVersion: null,
    releaseName: null,
    releaseNotes: null,
    downloadUrl: null,
    checksum: null,
    signature: null,
    signatureVerified: false,
    integrity: 'not_checked',
    feedUrl,
    checkedAt: null,
    error: null,
    automaticUpdate: null,
    automaticInstall: { supported: false, reason: 'Check for a signed update first.' },
    downloadProgress: null,
  }
}

export function checkingUpdateStatus(currentVersion: string, feedUrl: string): AppUpdateStatus {
  return { ...emptyUpdateStatus(currentVersion, feedUrl), state: 'checking' }
}

export interface UpdateCheckDependencies {
  /** Tests/key-rotation verification only. Production callers omit this and use the embedded root. */
  publicKey?: string
  lookup?: PinnedHostResolver
  fetchImpl?: typeof fetch
}

export async function checkUpdateFeed(
  feedUrl: string,
  currentVersion: string,
  signal?: AbortSignal,
  deps: UpdateCheckDependencies = {},
): Promise<AppUpdateStatus> {
  assertUpdateFeedUrl(feedUrl)
  const target = await resolveFetchableUrl(feedUrl, deps.lookup)
  const res = await pinnedFetch(target, {
    headers: { Accept: 'application/json, application/rss+xml, application/xml, text/xml, */*' },
    signal,
  }, deps.fetchImpl ?? fetch)
  if (!res.ok) {
    await discardResponseBody(res)
    throw new Error(`Update feed returned HTTP ${res.status}`)
  }
  const { text, contentType } = await readManifestResponse(res)
  const parsed = parseFeed(text, contentType)
  if (!parsed.downloadUrl) throw new Error('Update manifest did not include a download URL')
  if (!parsed.checksum) throw new Error('Update manifest did not include a valid SHA-256 checksum')
  if (!parsed.signature) throw new Error('Update manifest did not include an Ed25519 signature')
  const signaturePublicKey = deps.publicKey ?? updateTrustRootPublicKey()
  if (!verifyUpdateSignature(parsed, signaturePublicKey)) {
    throw new Error('Update manifest Ed25519 signature verification failed')
  }
  const newer = compareVersions(parsed.version, currentVersion) > 0
  return {
    state: newer ? 'available' : 'current',
    currentVersion,
    latestVersion: parsed.version,
    releaseName: parsed.releaseName,
    releaseNotes: parsed.releaseNotes,
    downloadUrl: parsed.downloadUrl,
    checksum: parsed.checksum,
    signature: parsed.signature,
    signatureVerified: true,
    integrity: 'signature_verified',
    feedUrl,
    checkedAt: Date.now(),
    error: null,
    automaticUpdate: parsed.automaticUpdate,
    downloadProgress: null,
  }
}

function parseFeed(text: string, contentType: string): ParsedFeed {
  const trimmed = text.trim()
  if (contentType.includes('json') || trimmed.startsWith('{') || trimmed.startsWith('[')) return parseJsonFeed(trimmed)
  return parseAppcast(trimmed)
}

function parseJsonFeed(text: string): ParsedFeed {
  const data = JSON.parse(text)
  const candidates = jsonReleaseCandidates(data)
  const source = candidates[0] ?? data
  const asset = pickReleaseAsset(source)
  const version = stringValue(source?.version ?? source?.latestVersion ?? source?.tag_name ?? source?.name)
  if (!version) throw new Error('Update manifest did not include a version')
  const releaseNotes = normalizeReleaseNotes(
    source?.releaseNotes ?? source?.notes ?? source?.body ?? source?.description ?? source?.changelog ?? source?.changes ?? source?.highlights,
    source?.releaseNotesUrl ?? source?.releaseNotesURL ?? source?.release_notes_url ?? source?.releaseNotesLink ?? source?.release_notes_link ?? source?.changelogUrl ?? source?.changelogURL ?? source?.changelog_url ?? source?.notesUrl ?? source?.notesURL ?? source?.notes_url,
  )
  return {
    version,
    releaseName: stringValue(source?.releaseName ?? source?.name ?? source?.title),
    releaseNotes,
    downloadUrl: publicHttpsUrlValue(source?.downloadUrl ?? source?.download_url ?? source?.download ?? asset?.browser_download_url ?? asset?.downloadUrl ?? asset?.download_url ?? asset?.url ?? source?.url ?? source?.html_url, 'update download URL'),
    checksum: parseChecksum(source?.sha256 ?? source?.checksumSha256 ?? source?.downloadSha256 ?? source?.checksum ?? source?.integrity ?? asset?.sha256 ?? asset?.checksumSha256 ?? asset?.checksum ?? asset?.integrity),
    signature: stringValue(source?.signature ?? source?.ed25519Signature ?? source?.edSignature ?? source?.sparkleEdSignature),
    signaturePayload: canonicalJson(stripSignatureFields(source)),
    automaticUpdate: parseAutomaticUpdate(source?.automaticUpdate),
  }
}

function jsonReleaseCandidates(data: any): any[] {
  const list = Array.isArray(data)
    ? data
    : [
      data?.latest,
      data?.release,
      data?.update,
      ...(Array.isArray(data?.releases) ? data.releases : []),
      ...(Array.isArray(data?.versions) ? data.versions : []),
      data,
    ]
  return list
    .filter((item) => item && typeof item === 'object')
    .filter((item, index, arr) => arr.indexOf(item) === index)
    .sort((a, b) => compareVersions(stringValue(b?.version ?? b?.latestVersion ?? b?.tag_name ?? b?.name) ?? '0', stringValue(a?.version ?? a?.latestVersion ?? a?.tag_name ?? a?.name) ?? '0'))
}

function parseAutomaticUpdate(value: unknown): AppUpdateStatus['automaticUpdate'] {
  if (value == null) return null
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Update manifest automatic update metadata is invalid')
  const raw = value as Record<string, unknown>
  if (raw.platform !== 'macos' || !['arm64', 'x64', 'universal'].includes(String(raw.arch))) throw new Error('Update manifest automatic update platform is unsupported')
  const zipUrl = publicHttpsUrlValue(raw.zipUrl, 'automatic update ZIP URL')
  if (!zipUrl || !new URL(zipUrl).pathname.toLowerCase().endsWith('.zip')) throw new Error('Automatic updates require a ZIP URL')
  const checksum = parseChecksum(raw.sha256)
  if (!checksum) throw new Error('Automatic update ZIP requires a valid SHA-256 checksum')
  return { zipUrl, checksum, arch: raw.arch as 'arm64' | 'x64' | 'universal' }
}

function pickReleaseAsset(source: any): any | null {
  const assets = Array.isArray(source?.assets) ? source.assets : []
  if (!assets.length) return null
  const scored = assets.map((asset: any, index: number) => {
    const haystack = [asset?.name, asset?.label, asset?.platform, asset?.os, asset?.arch, asset?.browser_download_url, asset?.downloadUrl, asset?.url].map((value) => String(value ?? '').toLowerCase()).join(' ')
    let score = 0
    if (/\b(mac|macos|darwin|osx)\b/.test(haystack)) score += 20
    if (/\b(universal|arm64|aarch64|x64|x86_64)\b/.test(haystack)) score += 8
    if (/\.dmg(\?|$|\s)/.test(haystack)) score += 6
    if (/\.zip(\?|$|\s)/.test(haystack)) score += 4
    if (asset?.browser_download_url || asset?.downloadUrl || asset?.download_url) score += 3
    return { asset, score, index }
  })
  scored.sort((a: { score: number; index: number }, b: { score: number; index: number }) => b.score - a.score || a.index - b.index)
  return scored[0]?.asset ?? assets[0]
}

function parseAppcast(xml: string): ParsedFeed {
  const item = xml.match(/<item\b[\s\S]*?<\/item>/i)?.[0] ?? xml
  const enclosureAttrs = parseAttrs(item.match(/<enclosure\b([^>]*)>/i)?.[1] ?? '')
  const version =
    enclosureAttrs['sparkle:shortVersionString']
    ?? enclosureAttrs['sparkle:version']
    ?? tagValue(item, 'sparkle:shortVersionString')
    ?? tagValue(item, 'sparkle:version')
  if (!version) throw new Error('Update appcast did not include a Sparkle version')
  const downloadUrl = publicHttpsUrlValue(enclosureAttrs.url ?? tagValue(item, 'link'), 'update download URL')
  const checksum = parseChecksum(enclosureAttrs['sparkle:sha256'] ?? enclosureAttrs.sha256 ?? enclosureAttrs.checksum ?? tagValue(item, 'sparkle:sha256') ?? tagValue(item, 'sha256'))
  return {
    version,
    releaseName: tagValue(item, 'title'),
    releaseNotes: normalizeReleaseNotes(
      tagValue(item, 'description'),
      enclosureAttrs['sparkle:releaseNotesLink'] ?? enclosureAttrs.releaseNotesLink ?? enclosureAttrs.releaseNotesUrl ?? tagValue(item, 'sparkle:releaseNotesLink') ?? tagValue(item, 'releaseNotesLink') ?? tagValue(item, 'releaseNotesUrl'),
    ),
    downloadUrl,
    checksum,
    signature: enclosureAttrs['sparkle:edSignature'] ?? enclosureAttrs.edSignature ?? tagValue(item, 'sparkle:edSignature'),
    signaturePayload: canonicalJson({
      version,
      downloadUrl,
      checksum: checksum?.value ?? null,
    }),
    automaticUpdate: null,
  }
}

function parseAttrs(raw: string): Record<string, string> {
  const attrs: Record<string, string> = {}
  const re = /([\w:-]+)\s*=\s*["']([^"']+)["']/g
  let m: RegExpExecArray | null
  while ((m = re.exec(raw))) attrs[m[1]] = decodeEntities(m[2])
  return attrs
}

function tagValue(xml: string, tag: string): string | null {
  const escaped = tag.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const m = xml.match(new RegExp(`<${escaped}\\b[^>]*>([\\s\\S]*?)<\\/${escaped}>`, 'i'))
  return m ? decodeEntities(stripTags(m[1]).trim()) || null : null
}

function stripTags(s: string): string {
  return s.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, '$1').replace(/<[^>]+>/g, ' ')
}

function stringValue(value: unknown): string | null {
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function normalizeReleaseNotes(notes: unknown, notesUrl: unknown): string | null {
  const text = notesValue(notes)
  const url = publicHttpsUrlValue(notesUrl, 'release notes URL')
  if (text && looksLikeUrl(text)) return publicHttpsUrlValue(text, 'release notes URL')
  return text ?? url
}

function notesValue(value: unknown): string | null {
  if (typeof value === 'string') return stringValue(value)
  if (Array.isArray(value)) {
    const lines = value.map((item) => {
      if (typeof item === 'string') return `- ${item.trim()}`
      if (item && typeof item === 'object') {
        const title = stringValue((item as any).title ?? (item as any).label ?? (item as any).heading)
        const body = notesValue((item as any).items ?? (item as any).changes ?? (item as any).body ?? (item as any).text)
        return [title, body].filter(Boolean).join('\n')
      }
      return null
    }).filter(Boolean)
    return lines.length ? lines.join('\n') : null
  }
  if (value && typeof value === 'object') {
    const lines = Object.entries(value as Record<string, unknown>).flatMap(([key, item]) => {
      const body = notesValue(item)
      return body ? [`${humanizeKey(key)}:`, body] : []
    })
    return lines.length ? lines.join('\n') : null
  }
  return null
}

function humanizeKey(key: string): string {
  return key.replace(/[_-]+/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())
}

function publicHttpsUrlValue(value: unknown, label: string): string | null {
  const raw = stringValue(value)
  if (!raw) return null
  assertPublicHttpsUrl(raw, label)
  return raw
}

function looksLikeUrl(value: string): boolean {
  return /^[a-z][a-z0-9+.-]*:\/\//i.test(value.trim())
}

function assertPublicHttpsUrl(raw: string, label: string): void {
  let url: URL
  try {
    url = new URL(raw)
  } catch {
    throw new Error(`Update manifest included an invalid ${label}`)
  }
  if (url.protocol !== 'https:') throw new Error(`Update manifest ${label} must use https`)
  if (url.username || url.password) throw new Error(`Update manifest ${label} must not include credentials`)
  if (isUnsafeUpdateHost(url.hostname)) throw new Error(`Update manifest ${label} must use a public host`)
}

function isUnsafeUpdateHost(hostname: string): boolean {
  const host = hostname.toLowerCase().replace(/^\[|\]$/g, '')
  if (!host || host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local')) return true
  if (host.includes(':')) return isUnsafeIpv6Host(host)
  const ipv4 = host.match(/^(\d+)\.(\d+)\.(\d+)\.(\d+)$/)
  if (!ipv4) return false
  const [a, b, c, d] = ipv4.slice(1).map((part) => Number(part))
  if ([a, b, c, d].some((part) => !Number.isInteger(part) || part < 0 || part > 255)) return true
  if (a === 0 || a === 10 || a === 127 || a >= 224) return true
  if (a === 100 && b >= 64 && b <= 127) return true
  if (a === 169 && b === 254) return true
  if (a === 172 && b >= 16 && b <= 31) return true
  if (a === 192 && b === 168) return true
  if (a === 198 && (b === 18 || b === 19)) return true
  return false
}

function isUnsafeIpv6Host(host: string): boolean {
  return host === '::' || host === '::1' || host.startsWith('fc') || host.startsWith('fd') || host.startsWith('fe80:')
}

function parseChecksum(value: unknown): AppUpdateChecksum | null {
  const raw = stringValue(value)
  if (!raw) return null
  const match = raw.match(/(?:sha256[-:=\s])?([a-f0-9]{64})/i)
  if (!match) return null
  return { algorithm: 'sha256', value: match[1].toLowerCase() }
}

function verifyUpdateSignature(parsed: ParsedFeed, publicKey: string): boolean {
  try {
    return verify(null, Buffer.from(parsed.signaturePayload), publicKey, Buffer.from(parsed.signature ?? '', 'base64'))
  } catch {
    return false
  }
}

function stripSignatureFields(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stripSignatureFields)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value as Record<string, unknown>)
    .filter(([key]) => !['signature', 'ed25519Signature', 'edSignature', 'sparkleEdSignature'].includes(key))
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => [key, stripSignatureFields(item)]))
}

function canonicalJson(value: unknown): string {
  return JSON.stringify(sortJson(value))
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJson)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, sortJson(item)]))
}

function decodeEntities(value: string): string {
  return value
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
}

// Semver order, so a beta sorts below its release (1.0.0-beta.1 < 1.0.0 < 1.0.1). Counting every
// number instead read 1.0.0-beta.1 as 1.0.0.1, and beta testers were never offered 1.0.0.
export function compareVersions(a: string, b: string): number {
  return compareSemver(parseSemver(a), parseSemver(b))
}
