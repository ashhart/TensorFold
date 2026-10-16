// Inbound media cache. Connector messages (Discord attachments, WhatsApp media, email
// inline images, etc.) arrive as REMOTE URLs that (a) expire — a CDN link is dead in
// hours, so by the time the assistant looks the image is gone — and (b) the vision tool
// cannot read, because it reads local file paths, not URLs. This module downloads inbound
// media to ~/.vontra/cache/{images,documents,audio} and returns a LOCAL path that the
// assistant / vision tool can read by path, with three safety properties:
//
//   1. SSRF VALIDATION — the URL is validated (scheme=https, non-private host, DNS-resolved
//      so a public name can't resolve to 169.254.169.254) BEFORE any byte is fetched, and
//      RE-validated on every redirect hop (a public page can 302 to an internal target).
//      Reuses the repo's existing host/IP guards (security.ts isPrivateHost/isPrivateIp).
//   2. SIZE CAP — enforced WHILE streaming (not after), so a malicious/huge response can't
//      fill the disk: the moment the running byte count crosses the cap the stream is
//      aborted and the partial file removed. A declared Content-Length over the cap is
//      rejected up front.
//   3. LOCAL PATH — the bytes land in a content-addressed file under the cache dir and the
//      path is returned for storage on the ConnectorEvent (mediaPaths).
//
// The pure pieces (URL gate + size-cap accounting + classification + path derivation) are
// unit-tested (media-cache.test.ts). The streaming download + the periodic prune are the
// only I/O and run in the main host.

import { createHash, randomUUID } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { isGloballyRoutableIp, isPrivateHost, resolveConnectorRequest, type ConnectorEndpointOptions } from './security'
import { createPinnedFetch, discardResponseBody, resolvePinnedHost, type PinnedHost } from './pinned-fetch'

// ~25 MB default cap on a single download. Connectors can pass a tighter cap.
export const MEDIA_SIZE_CAP_BYTES = 25 * 1024 * 1024
// How long a cached file lives before the periodic prune removes it (7 days).
export const MEDIA_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000
// Cap the redirect chain so a malicious server can't loop us forever.
const MAX_REDIRECTS = 5

export type MediaKind = 'images' | 'documents' | 'audio'

export interface MediaCacheResult {
  ok: boolean
  // Absolute local path to the cached file when ok; null otherwise.
  path: string | null
  kind: MediaKind
  bytes: number
  contentType: string | null
  sourceUrl: string
  error?: string
}

// ---- 1. SSRF URL gate (pure: scheme + host literal) ---------------------------
// A re-implementation of the tool-web/security contract specialized for inbound media:
// only https is allowed (inbound media is always cloud-hosted; http would also let a
// downgrade attack reach a loopback service), credentials in the URL are stripped, and
// any private/loopback/link-local/metadata host literal is rejected without DNS. The
// DNS-resolved check (assertCacheableUrl) layers on top for hostnames.

/**
 * Validate a media URL by its literal form and return the normalized URL (credentials
 * stripped). Throws on a non-https scheme or a private/loopback/link-local host literal.
 * Pure — no DNS, no network — so it is unit-testable and also the synchronous fast-path
 * inside assertCacheableUrl.
 */
export function safeMediaUrl(raw: string): URL {
  const trimmed = String(raw || '').trim()
  if (!trimmed) throw new Error('Media URL is empty.')
  let url: URL
  try {
    url = new URL(trimmed)
  } catch {
    throw new Error('Media URL is invalid.')
  }
  if (url.protocol !== 'https:') throw new Error('Media URLs must use https.')
  if (isPrivateHost(url.hostname)) throw new Error('Refusing to download media from a local, private, link-local, or metadata host.')
  url.username = ''
  url.password = ''
  return url
}

/** Non-throwing predicate form of safeMediaUrl (for filtering candidate URLs). */
export function isCacheableMediaUrl(raw: string): boolean {
  try {
    safeMediaUrl(raw)
    return true
  } catch {
    return false
  }
}

/**
 * Validate + DNS-resolve a media URL. Layers a DNS check on top of safeMediaUrl so a
 * public hostname that resolves to an internal address (DNS-rebinding / metadata SSRF) is
 * rejected before any byte is fetched. A resolver FAILURE is fatal here (fail closed) —
 * unlike a page fetch, we will not stream attacker-controlled bytes to disk on a host we
 * could not vet. Returns the normalized URL.
 */
export async function assertCacheableUrl(raw: string): Promise<URL> {
  return (await assertCacheableUrlPinned(raw)).url
}

/**
 * Like assertCacheableUrl, but also returns the vetted resolved addresses so the caller
 * can PIN the connection to one of them. Without pinning, fetch() re-resolves the hostname
 * independently after this check — a DNS-rebinding TOCTOU window where a malicious resolver
 * can return a public IP for the safety lookup and a private/metadata IP for the actual
 * fetch. Sharing the validated address between check and use closes that window.
 */
export async function assertCacheableUrlPinned(raw: string): Promise<PinnedHost> {
  const url = safeMediaUrl(raw)
  return resolvePinnedHost(url, (address) => {
    if (!isGloballyRoutableIp(address)) {
      throw new Error(`Refusing to download media from ${url.hostname}: it resolves to a private or non-public address (${address}).`)
    }
  })
}

// ---- 2. Size-cap accounting (pure) --------------------------------------------

/**
 * Decide whether a declared Content-Length already exceeds the cap (reject up front,
 * before streaming a single byte). A missing/unparseable length returns false — the cap
 * is then enforced by the running counter while streaming. A non-positive cap disables
 * the up-front check.
 */
export function exceedsDeclaredSize(contentLength: string | number | null | undefined, capBytes = MEDIA_SIZE_CAP_BYTES): boolean {
  if (capBytes <= 0) return false
  const n = typeof contentLength === 'number' ? contentLength : parseInt(String(contentLength ?? ''), 10)
  return Number.isFinite(n) && n > capBytes
}

/**
 * Running size-cap check used while streaming chunks. Given the bytes already written and
 * the next chunk's size, returns whether appending it would cross the cap. The caller
 * aborts + deletes the partial file when this is true, so the cap is enforced INCREMENTALLY
 * (a server lying about Content-Length still can't overflow the disk). A non-positive cap
 * means "no cap".
 */
export function wouldExceedSizeCap(writtenBytes: number, nextChunkBytes: number, capBytes = MEDIA_SIZE_CAP_BYTES): boolean {
  if (capBytes <= 0) return false
  return writtenBytes + nextChunkBytes > capBytes
}

// ---- 3. Classification + path derivation (pure) -------------------------------

const IMAGE_EXT = new Set(['png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'svg', 'heic', 'heif', 'tiff', 'avif'])
const AUDIO_EXT = new Set(['mp3', 'm4a', 'aac', 'wav', 'ogg', 'oga', 'opus', 'flac', 'amr', 'weba'])

/**
 * Classify media into one of the cache subdirectories from its content-type and/or its
 * URL/file extension. Content-type wins (it's what the server actually sent); the
 * extension is the fallback. Anything that isn't clearly an image or audio file is a
 * 'document' (the catch-all PDFs/office files land in).
 */
export function classifyMedia(contentType: string | null | undefined, urlOrName: string | null | undefined): MediaKind {
  const ct = String(contentType ?? '').toLowerCase()
  if (ct.startsWith('image/')) return 'images'
  if (ct.startsWith('audio/')) return 'audio'
  if (ct.startsWith('video/')) return 'documents' // videos are bucketed with documents (not vision-read)
  if (ct.startsWith('application/') || ct.startsWith('text/')) return 'documents'
  const ext = mediaExtension(urlOrName, null)
  if (ext && IMAGE_EXT.has(ext)) return 'images'
  if (ext && AUDIO_EXT.has(ext)) return 'audio'
  return 'documents'
}

/** Lower-cased file extension (no dot) from a URL path or filename; '' when none. */
export function mediaExtension(urlOrName: string | null | undefined, contentType: string | null | undefined): string {
  const raw = String(urlOrName ?? '')
  let pathname = raw
  try {
    pathname = new URL(raw).pathname
  } catch {
    /* not a URL — treat as a bare name */
  }
  const base = pathname.split('/').pop() ?? ''
  const dot = base.lastIndexOf('.')
  const ext = dot > 0 ? base.slice(dot + 1).toLowerCase().replace(/[^a-z0-9]+/g, '') : ''
  if (ext && ext.length <= 5) return ext
  return extFromContentType(contentType)
}

function extFromContentType(contentType: string | null | undefined): string {
  const ct = String(contentType ?? '').toLowerCase().split(';')[0].trim()
  const map: Record<string, string> = {
    'image/png': 'png', 'image/jpeg': 'jpg', 'image/jpg': 'jpg', 'image/gif': 'gif', 'image/webp': 'webp', 'image/svg+xml': 'svg', 'image/heic': 'heic', 'image/avif': 'avif',
    'audio/mpeg': 'mp3', 'audio/mp4': 'm4a', 'audio/aac': 'aac', 'audio/wav': 'wav', 'audio/ogg': 'ogg', 'audio/opus': 'opus', 'audio/flac': 'flac',
    'application/pdf': 'pdf', 'application/zip': 'zip', 'text/plain': 'txt', 'text/csv': 'csv', 'application/json': 'json',
  }
  return map[ct] ?? ''
}

/**
 * Content-addressed file name for a media URL: a stable hash of the URL (so the same
 * attachment is cached once and re-syncs are idempotent) plus the derived extension.
 * Deterministic and pure, so the unit test pins it.
 */
export function mediaFileName(url: string, contentType: string | null | undefined): string {
  const hash = createHash('sha256').update(String(url ?? '')).digest('hex').slice(0, 24)
  const ext = mediaExtension(url, contentType)
  return ext ? `${hash}.${ext}` : hash
}

/**
 * Absolute local path a media URL maps to under a cache root, given its kind and an
 * optional content-type for the extension. Pure path math (no I/O) so callers can compute
 * the target, the dedup key, and the unit test all the same way.
 */
export function mediaCachePath(cacheRoot: string, kind: MediaKind, url: string, contentType: string | null | undefined): string {
  return path.join(cacheRoot, kind, mediaFileName(url, contentType))
}

// ---- 4. Streaming download (I/O — host) ----------------------------------------

const MEDIA_FETCH_HEADERS = {
  'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
  accept: 'image/avif,image/webp,image/png,image/*,application/pdf,audio/*,*/*;q=0.8',
}

export interface DownloadMediaOptions {
  cacheRoot: string
  capBytes?: number
  timeoutMs?: number
  // Injectable fetch for tests; defaults to global fetch.
  fetchImpl?: typeof fetch
  // Extra request headers — used to carry an Authorization Bearer for media that needs auth
  // (Slack url_private requires the bot token). SECURITY: an auth header is sensitive, so it
  // is sent ONLY on the original host and any SAME-HOST redirect hop; a redirect that changes
  // host drops it (see headersForRedirect) so the secret never leaks to a third-party CDN.
  headers?: Record<string, string>
  // Authenticated media from a self-hosted connector may intentionally live on its
  // opted-in private origin. The exception is scoped to that exact origin; any cross-origin
  // redirect loses auth headers and returns to the default public-HTTPS media policy.
  endpoint?: { baseUrl: string; label: string; options?: ConnectorEndpointOptions }
}

// Headers we will not forward across a cross-host redirect (case-insensitive). Authorization
// is the sensitive one; cookie is included defensively in case a caller ever sets it.
const CROSS_HOST_STRIPPED_HEADERS = new Set(['authorization', 'cookie'])

// Drop sensitive headers when a redirect changes host so a Bearer token can't be replayed to a
// different origin. Slack's url_private is already a files.slack.com URL, so the token is sent
// directly to that host on the FIRST request (no redirect narrowing applies there); this guard
// only fires if such a URL ever 302s onward — in which case stripping the token is the safe
// default (the onward target is either same-host, or a pre-signed URL needing no auth). Pure.
export function headersForRedirect(headers: Record<string, string> | undefined, fromOrigin: string, toOrigin: string): Record<string, string> | undefined {
  if (!headers) return headers
  if (fromOrigin.toLowerCase() === toOrigin.toLowerCase()) return headers
  const out: Record<string, string> = {}
  for (const [k, v] of Object.entries(headers)) {
    if (CROSS_HOST_STRIPPED_HEADERS.has(k.toLowerCase())) continue
    out[k] = v
  }
  return out
}

async function resolveMediaTarget(raw: string, endpoint?: DownloadMediaOptions['endpoint']): Promise<PinnedHost> {
  if (endpoint) {
    const request = new URL(raw)
    const configured = new URL(endpoint.baseUrl)
    if (request.origin === configured.origin) {
      return resolveConnectorRequest(endpoint.baseUrl, endpoint.label, endpoint.options ?? {}, request)
    }
  }
  return assertCacheableUrlPinned(raw)
}

/**
 * Download one inbound media URL to the cache and return its local path. Performs the full
 * safety contract: SSRF validation (scheme + host + DNS), manual redirect-following with a
 * re-validation of every hop, an up-front Content-Length cap, and an incremental byte cap
 * while streaming (partial file removed on overflow). Never throws — returns a structured
 * result so a connector sync can attach the path when ok and otherwise carry on. Idempotent:
 * a URL already cached (by content-addressed name) is returned without re-downloading.
 */
export async function downloadMedia(rawUrl: string, opts: DownloadMediaOptions): Promise<MediaCacheResult> {
  const capBytes = opts.capBytes ?? MEDIA_SIZE_CAP_BYTES
  const doFetch = opts.fetchImpl ?? fetch
  const base: Omit<MediaCacheResult, 'ok' | 'path' | 'error'> = { kind: 'documents', bytes: 0, contentType: null, sourceUrl: String(rawUrl ?? '') }
  let current: URL
  let target: PinnedHost
  try {
    target = await resolveMediaTarget(rawUrl, opts.endpoint)
    current = target.url
  } catch (e: any) {
    return { ...base, ok: false, path: null, error: String(e?.message ?? e) }
  }

  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), opts.timeoutMs ?? 30_000)
  try {
    let res: Response | null = null
    // Caller-supplied headers (e.g. an Authorization Bearer for Slack url_private) are layered
    // ON TOP of the default media headers, then narrowed on each cross-host redirect so the
    // secret can't be replayed to a third-party origin.
    let authHeaders = opts.headers
    for (let hop = 0; hop <= MAX_REDIRECTS; hop++) {
      res = await createPinnedFetch(target, doFetch)(current.toString(), {
        headers: { ...MEDIA_FETCH_HEADERS, ...(authHeaders ?? {}) },
        redirect: 'manual',
        signal: controller.signal,
      })
      if (res.status >= 300 && res.status < 400) {
        const location = res.headers.get('location')
        if (!location) break
        if (hop >= MAX_REDIRECTS) {
          await discardResponseBody(res)
          return { ...base, ok: false, path: null, error: 'Media redirect limit exceeded.' }
        }
        // Re-validate the redirect target against the SSRF guard BEFORE following it, and
        // capture its vetted addresses to pin the next hop.
        const next = new URL(location, current)
        const validated = await resolveMediaTarget(next.toString(), opts.endpoint)
        // Drop the Authorization header if the host changed so the Bearer token doesn't leak.
        authHeaders = headersForRedirect(authHeaders, current.origin, validated.url.origin)
        current = validated.url
        target = validated
        await discardResponseBody(res)
        continue
      }
      break
    }
    if (!res) return { ...base, ok: false, path: null, error: 'No response.' }
    if (!res.ok) {
      await discardResponseBody(res)
      return { ...base, ok: false, path: null, error: `Download failed (${res.status}).` }
    }

    const contentType = res.headers.get('content-type')
    if (exceedsDeclaredSize(res.headers.get('content-length'), capBytes)) {
      await discardResponseBody(res)
      return { ...base, ok: false, contentType, path: null, error: `Media exceeds the ${capBytes}-byte cap (declared Content-Length).` }
    }
    const declaredKind = classifyMedia(contentType, current.toString())
    let kind = declaredKind
    let fileTarget = mediaCachePath(opts.cacheRoot, declaredKind, base.sourceUrl, contentType)
    await fs.mkdir(path.dirname(fileTarget), { recursive: true, mode: 0o700 })

    // Idempotent: already cached → return without re-downloading.
    const existing = await statSize(fileTarget)
    if (existing != null) {
      await discardResponseBody(res)
      return { ...base, ok: true, path: fileTarget, kind, bytes: existing, contentType }
    }

    if (!res.body) {
      // No streaming body (e.g. a mocked fetch returning a buffer): fall back to arrayBuffer
      // and still enforce the cap.
      const buf = Buffer.from(await res.arrayBuffer())
      if (wouldExceedSizeCap(0, buf.byteLength, capBytes)) {
        return { ...base, ok: false, contentType, path: null, error: `Media exceeds the ${capBytes}-byte cap.` }
      }
      // Reconcile the declared kind against the actual magic bytes: script bytes served with
      // an image content-type are cached as documents, never fed to a vision/image handler.
      kind = buf.byteLength >= 4 ? reconcileMediaKind(declaredKind, buf) : declaredKind
      fileTarget = mediaCachePath(opts.cacheRoot, kind, base.sourceUrl, contentType)
      // The reconciled kind may differ from the declared one — its subdir may not exist yet.
      await fs.mkdir(path.dirname(fileTarget), { recursive: true, mode: 0o700 })
      // Write to a sibling temp file then rename, so a crash mid-write never leaves a
      // truncated blob at the content-addressed path (which statSize would later mistake
      // for a complete cache hit).
      const temp = `${fileTarget}.tmp-${process.pid}-${randomUUID()}`
      try {
        await fs.writeFile(temp, buf, { mode: 0o600 })
        await fs.rename(temp, fileTarget)
      } catch (e) {
        await fs.rm(temp, { force: true }).catch(() => {})
        throw e
      }
      return { ...base, ok: true, path: fileTarget, kind, bytes: buf.byteLength, contentType }
    }

    // Stream into a sibling temp file and rename on completion. A crash or overflow leaves a
    // partial temp file (never a torn file under the content-hash name), so statSize can never
    // promote a truncated download into a cache hit.
    const streamTemp = `${fileTarget}.tmp-${process.pid}-${randomUUID()}`
    let written = 0
    let overflowed = false
    let head: Uint8Array | null = null
    const handle = await fs.open(streamTemp, 'w', 0o600)
    try {
      // @ts-expect-error Web ReadableStream is async-iterable at runtime in Node 18+.
      for await (const chunk of res.body as AsyncIterable<Uint8Array>) {
        const bytes = chunk.byteLength
        if (wouldExceedSizeCap(written, bytes, capBytes)) {
          overflowed = true
          break
        }
        if (!head && bytes > 0) head = chunk.slice(0, 16) // leading bytes for the magic-byte check
        await handle.write(chunk)
        written += bytes
      }
    } finally {
      await handle.close().catch(() => {})
    }
    if (overflowed) {
      // Drop the partial temp file — never the content-addressed target — so the cap is
      // enforced without leaving a torn blob at the cache path.
      await fs.rm(streamTemp, { force: true })
      return { ...base, ok: false, contentType, path: null, error: `Media exceeds the ${capBytes}-byte cap (truncated while streaming).` }
    }
    // Same magic-byte reconciliation as the buffer path above.
    kind = head && head.byteLength >= 4 ? reconcileMediaKind(declaredKind, head) : declaredKind
    fileTarget = mediaCachePath(opts.cacheRoot, kind, base.sourceUrl, contentType)
    // The reconciled kind may differ from the declared one — its subdir may not exist yet.
    await fs.mkdir(path.dirname(fileTarget), { recursive: true, mode: 0o700 })
    try {
      await fs.rename(streamTemp, fileTarget)
    } catch (e) {
      await fs.rm(streamTemp, { force: true }).catch(() => {})
      throw e
    }
    return { ...base, ok: true, path: fileTarget, kind, bytes: written, contentType }
  } catch (e: any) {
    return { ...base, ok: false, path: null, error: String(e?.message ?? e) }
  } finally {
    clearTimeout(timeout)
  }
}

/**
 * Download a set of media URLs for one inbound message, returning the local paths of the
 * ones that succeeded (in input order, de-duplicated). Failures are skipped — a bad/expired
 * attachment must never block ingesting the message itself. Caps the number of attachments
 * processed per message so a single message can't trigger an unbounded fan-out of fetches.
 */
export async function cacheInboundMedia(urls: string[], opts: DownloadMediaOptions & { maxAttachments?: number }): Promise<string[]> {
  const seen = new Set<string>()
  const candidates = (urls ?? []).filter((u) => {
    if (!u || seen.has(u) || !isCacheableMediaUrl(u)) return false
    seen.add(u)
    return true
  }).slice(0, opts.maxAttachments ?? 10)
  const out: string[] = []
  for (const url of candidates) {
    const result = await downloadMedia(url, opts)
    if (result.ok && result.path) out.push(result.path)
  }
  return out
}

/**
 * AUTH-aware variant of cacheInboundMedia: download a set of media URLs that require an
 * Authorization header (Slack url_private needs the bot token as `Bearer <token>`). Each item
 * carries its own URL + headers (so a future caller could mix sources). Reuses the SAME SSRF
 * + size-cap + cleanup contract as the public path via downloadMedia; the only addition is the
 * auth header, which is stripped on cross-host redirects (headersForRedirect). De-duplicated by
 * URL, capped count, failures swallowed per item so a bad/expired attachment never blocks ingest.
 */
export async function cacheAuthedMedia(
  items: { url: string; headers?: Record<string, string>; endpoint?: DownloadMediaOptions['endpoint'] }[],
  opts: DownloadMediaOptions & { maxAttachments?: number },
): Promise<string[]> {
  const seen = new Set<string>()
  const candidates = (items ?? []).filter((it) => {
    const u = it?.url
    if (!u || seen.has(u) || (!it.endpoint && !isCacheableMediaUrl(u))) return false
    seen.add(u)
    return true
  }).slice(0, opts.maxAttachments ?? 10)
  const out: string[] = []
  for (const item of candidates) {
    const result = await downloadMedia(item.url, { ...opts, headers: item.headers, endpoint: item.endpoint })
    if (result.ok && result.path) out.push(result.path)
  }
  return out
}

// ---- 4b. Byte-writer (I/O — host): cache media we already hold as bytes ----------
// Email attachments are inline MIME parts (not URLs), so there is nothing to fetch — the IMAP
// client already handed us the decoded bytes. This path classifies + size-caps + writes those
// bytes to the same cache, content-addressed by a hash of the BYTES (so the same attachment is
// cached once and re-syncs are idempotent even though there is no URL to key on).

export interface InboundMediaBytes {
  // Decoded attachment bytes (already base64/quoted-printable-decoded by the IMAP client).
  data: Uint8Array
  // Original filename (drives the extension + classification fallback); optional.
  filename?: string | null
  // MIME content-type from the part header (wins for classification); optional.
  contentType?: string | null
}

/**
 * Content-addressed file name for a byte blob: a stable hash of the BYTES plus the extension
 * derived from the filename/content-type. Deterministic + pure so the unit test can pin it and
 * the same attachment de-dupes across syncs.
 */
/**
 * Validate that downloaded bytes' magic signature is consistent with the declared media kind.
 * Catches content-confusion attacks where a server delivers HTML/JS/script bytes with an
 * image/audio content-type — a polyglot file cached and later rendered with the wrong handler
 * could enable stored-XSS or downstream parser confusion. Returns the corrected kind when the
 * bytes clearly disagree with the declaration, or the original kind when consistent/unknown.
 */
const MAGIC_SIGNATURES: Array<{ kind: MediaKind; signatures: number[][] }> = [
  { kind: 'images', signatures: [[0x89, 0x50, 0x4e, 0x47], [0xff, 0xd8, 0xff], [0x47, 0x49, 0x46, 0x38], [0x52, 0x49, 0x46, 0x46]] }, // PNG, JPEG, GIF, WEBP/RIFF
  { kind: 'audio', signatures: [[0x49, 0x44, 0x33], [0xff, 0xfb], [0x4f, 0x67, 0x67, 0x53]] }, // MP3 ID3, MP3 frame, OGG
  { kind: 'documents', signatures: [[0x25, 0x50, 0x44, 0x46], [0x50, 0x4b, 0x03, 0x04]] }, // PDF, ZIP
]
const SCRIPT_SIGNATURES: number[][] = [
  [0x3c, 0x21, 0x44, 0x4f], // <!DO (HTML doctype)
  [0x3c, 0x68, 0x74, 0x6d], // <htm (HTML)
  [0x3c, 0x73, 0x63, 0x72], // <scr (script)
  [0x3c, 0x73, 0x76, 0x67], // <svg
]

export function detectMediaKind(data: Uint8Array): MediaKind | null {
  if (data.length < 4) return null
  for (const { kind, signatures } of MAGIC_SIGNATURES) {
    for (const sig of signatures) {
      if (sig.every((byte, i) => data[i] === byte)) return kind
    }
  }
  return null
}

/** True when bytes start with an HTML/script signature — never a legitimate image/audio. */
function isScriptBytes(data: Uint8Array): boolean {
  if (data.length < 4) return false
  for (const sig of SCRIPT_SIGNATURES) {
    if (sig.every((byte, i) => data[i] === byte)) return true
  }
  return false
}

/**
 * Reconcile the declared kind (from Content-Type/filename) with the actual magic bytes.
 * Rejects script bytes disguised as media; reclassifies when bytes clearly disagree.
 */
export function reconcileMediaKind(declared: MediaKind, data: Uint8Array): MediaKind {
  // Script bytes are never legitimate images/audio — reject to documents (not executable by
  // the vision tool, and flagged for manual review).
  if (declared !== 'documents' && isScriptBytes(data)) return 'documents'
  const detected = detectMediaKind(data)
  if (detected && detected !== declared) return detected
  return declared
}

export function mediaFileNameForBytes(data: Uint8Array, filename: string | null | undefined, contentType: string | null | undefined): string {
  const hash = createHash('sha256').update(data).digest('hex').slice(0, 24)
  const ext = mediaExtension(filename, contentType)
  return ext ? `${hash}.${ext}` : hash
}

/**
 * Write one in-memory attachment to the cache and return its local path (or a structured
 * failure). Enforces the SAME size cap as the URL path (an oversized part is rejected, not
 * written) and classifies by content-type/filename. Idempotent: a blob already cached (same
 * content hash) is returned without rewriting. Never throws.
 */
export async function cacheBytes(item: InboundMediaBytes, opts: { cacheRoot: string; capBytes?: number }): Promise<MediaCacheResult> {
  const capBytes = opts.capBytes ?? MEDIA_SIZE_CAP_BYTES
  const data = item?.data ?? new Uint8Array(0)
  const contentType = item?.contentType ?? null
  // Reconcile the declared kind against the actual magic bytes — script bytes disguised as
  // an image are reclassified to documents so they can't be rendered by a vision/image handler.
  const declaredKind = classifyMedia(contentType, item?.filename ?? null)
  const kind = data.byteLength >= 4 ? reconcileMediaKind(declaredKind, data) : declaredKind
  const base: Omit<MediaCacheResult, 'ok' | 'path' | 'error'> = {
    kind,
    bytes: data.byteLength,
    contentType,
    sourceUrl: item?.filename ? `attachment:${item.filename}` : 'attachment:bytes',
  }
  if (!data.byteLength) return { ...base, ok: false, path: null, error: 'Attachment has no bytes.' }
  if (wouldExceedSizeCap(0, data.byteLength, capBytes)) {
    return { ...base, ok: false, path: null, error: `Attachment exceeds the ${capBytes}-byte cap.` }
  }
  try {
    const name = mediaFileNameForBytes(data, item?.filename ?? null, contentType)
    const target = path.join(opts.cacheRoot, base.kind, name)
    await fs.mkdir(path.dirname(target), { recursive: true, mode: 0o700 })
    // Idempotent: already cached → return without rewriting.
    const existing = await statSize(target)
    if (existing != null) return { ...base, ok: true, path: target }
    // Write to a sibling temp file then rename, so a crash mid-write never leaves a truncated
    // blob at the content-addressed path (which statSize would later mistake for a complete hit).
    const temp = `${target}.tmp-${process.pid}-${randomUUID()}`
    try {
      await fs.writeFile(temp, data, { mode: 0o600 })
      await fs.rename(temp, target)
    } catch (e) {
      await fs.rm(temp, { force: true }).catch(() => {})
      throw e
    }
    return { ...base, ok: true, path: target }
  } catch (e: any) {
    return { ...base, ok: false, path: null, error: String(e?.message ?? e) }
  }
}

/**
 * Write a set of in-memory attachments (one inbound message's MIME parts) to the cache and
 * return the local paths that succeeded (in input order). Caps the number processed per message
 * so a single message can't fan out unbounded writes. Per-item failures are swallowed — a bad
 * attachment must never block ingesting the message itself.
 */
export async function cacheInboundMediaBytes(
  items: InboundMediaBytes[],
  opts: { cacheRoot: string; capBytes?: number; maxAttachments?: number },
): Promise<string[]> {
  const out: string[] = []
  for (const item of (items ?? []).slice(0, opts.maxAttachments ?? 10)) {
    const result = await cacheBytes(item, opts)
    if (result.ok && result.path) out.push(result.path)
  }
  return out
}

// ---- 5. Periodic cleanup (I/O — host) ------------------------------------------

export interface MediaCleanupResult {
  removed: number
  bytesFreed: number
  scanned: number
}

/**
 * Prune cached media older than maxAgeMs across all kind subdirectories. Files are removed
 * by mtime, so a file re-touched by a recent re-download survives. Missing directories are
 * a no-op (nothing cached yet). Returns counts for observability/logging. Best-effort: a
 * file that can't be removed is skipped rather than failing the whole sweep.
 */
export async function cleanupMediaCache(cacheRoot: string, maxAgeMs = MEDIA_MAX_AGE_MS, now = Date.now()): Promise<MediaCleanupResult> {
  const cutoff = now - Math.max(0, maxAgeMs)
  const result: MediaCleanupResult = { removed: 0, bytesFreed: 0, scanned: 0 }
  for (const kind of ['images', 'documents', 'audio'] as MediaKind[]) {
    const dir = path.join(cacheRoot, kind)
    let entries: string[]
    try {
      entries = await fs.readdir(dir)
    } catch {
      continue // dir doesn't exist yet
    }
    for (const name of entries) {
      const file = path.join(dir, name)
      result.scanned++
      try {
        const st = await fs.stat(file)
        if (!st.isFile()) continue
        if (st.mtimeMs < cutoff) {
          await fs.rm(file, { force: true })
          result.removed++
          result.bytesFreed += st.size
        }
      } catch {
        /* skip unreadable/removed entry */
      }
    }
  }
  return result
}

async function statSize(file: string): Promise<number | null> {
  try {
    const st = await fs.stat(file)
    return st.isFile() ? st.size : null
  } catch {
    return null
  }
}
