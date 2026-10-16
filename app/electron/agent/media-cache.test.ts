// Run: jiti electron/agent/media-cache.test.ts
import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import {
  cacheBytes,
  classifyMedia,
  detectMediaKind,
  exceedsDeclaredSize,
  headersForRedirect,
  isCacheableMediaUrl,
  mediaCachePath,
  mediaExtension,
  mediaFileName,
  mediaFileNameForBytes,
  reconcileMediaKind,
  safeMediaUrl,
  wouldExceedSizeCap,
  MEDIA_SIZE_CAP_BYTES,
  downloadMedia,
} from './media-cache.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('media-cache tests:')

// ---- SSRF URL gate (pure: scheme + host literal) ------------------------------

{
  // https public hosts pass; credentials are stripped from the normalized URL.
  assert.equal(safeMediaUrl('https://cdn.discordapp.com/a/b.png').hostname, 'cdn.discordapp.com')
  const stripped = safeMediaUrl('https://user:pass@cdn.example.com/x.jpg')
  assert.equal(stripped.username, '')
  assert.equal(stripped.password, '')
  ok('safeMediaUrl: accepts https public host, strips embedded credentials')
}

{
  // http is rejected (downgrade attack could reach a loopback service).
  assert.throws(() => safeMediaUrl('http://cdn.example.com/x.png'), /https/)
  // Loopback / private / link-local / metadata literals are rejected without DNS.
  for (const bad of [
    'https://127.0.0.1/x.png',
    'https://localhost/x.png',
    'https://10.0.0.5/x.png',
    'https://192.168.1.10/x.png',
    'https://169.254.169.254/latest/meta-data/', // cloud metadata
    'https://[::1]/x.png',
    'https://172.16.0.1/x.png',
    'https://[::ffff:169.254.169.254]/x', // IPv4-mapped metadata
  ]) {
    assert.throws(() => safeMediaUrl(bad), new RegExp('local|private|link-local|metadata'), `should reject ${bad}`)
    assert.equal(isCacheableMediaUrl(bad), false, `isCacheableMediaUrl false for ${bad}`)
  }
  // Empty / non-URL input is rejected, not crashed.
  assert.equal(isCacheableMediaUrl(''), false)
  assert.equal(isCacheableMediaUrl('not a url'), false)
  assert.equal(isCacheableMediaUrl('ftp://example.com/x'), false)
  // A normal public https URL is cacheable.
  assert.equal(isCacheableMediaUrl('https://media.example.com/photo.jpg'), true)
  ok('safeMediaUrl/isCacheableMediaUrl: blocks http, loopback, RFC1918, link-local, metadata, IPv4-mapped')
}

// ---- Size-cap accounting (pure) -----------------------------------------------

{
  // Declared Content-Length over the cap is rejected up front; under/at/missing is allowed.
  assert.equal(exceedsDeclaredSize(MEDIA_SIZE_CAP_BYTES + 1), true)
  assert.equal(exceedsDeclaredSize(MEDIA_SIZE_CAP_BYTES), false)
  assert.equal(exceedsDeclaredSize('1048576'), false)
  assert.equal(exceedsDeclaredSize(null), false, 'missing length falls through to streaming cap')
  assert.equal(exceedsDeclaredSize('not-a-number'), false)
  assert.equal(exceedsDeclaredSize(99, 0), false, 'non-positive cap disables the up-front check')
  // A custom (tighter) cap is honoured.
  assert.equal(exceedsDeclaredSize(2000, 1000), true)
  ok('exceedsDeclaredSize: rejects over-cap Content-Length, allows missing/under, honours custom cap')
}

{
  // Incremental streaming cap: the chunk that would cross the cap is flagged.
  assert.equal(wouldExceedSizeCap(0, 100, 1000), false)
  assert.equal(wouldExceedSizeCap(900, 100, 1000), false, 'exactly at the cap is allowed')
  assert.equal(wouldExceedSizeCap(900, 101, 1000), true, 'one byte over the cap is flagged')
  assert.equal(wouldExceedSizeCap(1000, 1, 1000), true)
  assert.equal(wouldExceedSizeCap(5_000, 5_000, 0), false, 'no cap when capBytes<=0')
  // A server lying about Content-Length (declared small, streams big) is still caught here.
  assert.equal(wouldExceedSizeCap(MEDIA_SIZE_CAP_BYTES, 1), true)
  ok('wouldExceedSizeCap: incremental cap catches over-cap stream even if Content-Length lied')
}

// ---- Classification + path derivation (pure) ----------------------------------

{
  // Content-type wins.
  assert.equal(classifyMedia('image/png', 'https://x/y'), 'images')
  assert.equal(classifyMedia('image/jpeg; charset=binary', 'https://x/y'), 'images')
  assert.equal(classifyMedia('audio/mpeg', 'https://x/y'), 'audio')
  assert.equal(classifyMedia('application/pdf', 'https://x/y'), 'documents')
  assert.equal(classifyMedia('video/mp4', 'https://x/y'), 'documents', 'video bucketed with documents')
  // Extension fallback when content-type is absent/unknown.
  assert.equal(classifyMedia(null, 'https://cdn/x/photo.JPG'), 'images')
  assert.equal(classifyMedia(null, 'https://cdn/x/clip.m4a'), 'audio')
  assert.equal(classifyMedia(null, 'https://cdn/x/report.pdf'), 'documents')
  assert.equal(classifyMedia('', 'https://cdn/x/unknownthing'), 'documents', 'no signal => documents')
  ok('classifyMedia: content-type wins, extension fallback, default documents')
}

{
  // mediaExtension reads the URL path, falls back to content-type.
  assert.equal(mediaExtension('https://cdn/x/a.png?width=64', null), 'png')
  assert.equal(mediaExtension('https://cdn/x/a', 'image/webp'), 'webp')
  assert.equal(mediaExtension('plain-name.gif', null), 'gif')
  assert.equal(mediaExtension('https://cdn/x/noext', null), '')
  ok('mediaExtension: path extension first, then content-type map')
}

{
  // mediaFileName is a deterministic content-addressed name (stable across runs) + ext.
  const url = 'https://cdn.discordapp.com/attachments/1/2/photo.png'
  const a = mediaFileName(url, 'image/png')
  const b = mediaFileName(url, 'image/png')
  assert.equal(a, b, 'same URL => same name (idempotent cache)')
  assert.match(a, /^[0-9a-f]{24}\.png$/)
  assert.notEqual(mediaFileName('https://cdn/other.png', 'image/png'), a, 'different URL => different name')
  // Full path joins the kind subdir.
  const p = mediaCachePath('/root/cache', 'images', url, 'image/png')
  assert.equal(p, `/root/cache/images/${a}`)
  ok('mediaFileName/mediaCachePath: deterministic, content-addressed, under kind subdir')
}

// ---- Auth header redirect narrowing (pure) ------------------------------------

{
  // Same host => headers pass through unchanged (the Bearer is still safe to send).
  const h = { Authorization: 'Bearer xoxb-secret', accept: '*/*' }
  assert.equal(headersForRedirect(h, 'slack.com', 'slack.com'), h, 'same host keeps the auth header')
  // Cross-host => Authorization (and cookie) are stripped; benign headers survive. Slack's
  // url_private 302s from slack.com to files.slack.com (a DIFFERENT host) with a pre-signed URL,
  // so dropping the token there is both safe and correct.
  const narrowed = headersForRedirect({ Authorization: 'Bearer xoxb-secret', Cookie: 'a=b', accept: '*/*' }, 'slack.com', 'files.slack.com')
  assert.deepEqual(narrowed, { accept: '*/*' }, 'cross-host drops Authorization + Cookie, keeps accept')
  // Case-insensitive host compare; undefined headers are a no-op.
  assert.deepEqual(headersForRedirect({ Authorization: 'Bearer x' }, 'CDN.Example.com', 'cdn.example.com'), { Authorization: 'Bearer x' })
  assert.deepEqual(headersForRedirect({ Authorization: 'Bearer x' }, 'https://cdn.example.com', 'https://cdn.example.com:444'), {}, 'different port drops auth')
  assert.equal(headersForRedirect(undefined, 'a.com', 'b.com'), undefined)
  ok('headersForRedirect: keeps headers same-host, strips Authorization/Cookie cross-host')
}

// ---- Byte-blob content-addressed name (pure) ----------------------------------

{
  // mediaFileNameForBytes hashes the BYTES (so the same attachment de-dupes with no URL) and
  // appends the extension from filename, falling back to content-type.
  const bytes = new TextEncoder().encode('hello world pdf bytes')
  const a = mediaFileNameForBytes(bytes, 'invoice.pdf', 'application/pdf')
  const b = mediaFileNameForBytes(bytes, 'invoice.pdf', 'application/pdf')
  assert.equal(a, b, 'same bytes + name => same file name (idempotent)')
  assert.match(a, /^[0-9a-f]{24}\.pdf$/)
  // Different bytes => different name even with the same filename.
  assert.notEqual(mediaFileNameForBytes(new TextEncoder().encode('other'), 'invoice.pdf', 'application/pdf'), a)
  // No filename: extension comes from the content-type map.
  assert.match(mediaFileNameForBytes(bytes, null, 'image/png'), /^[0-9a-f]{24}\.png$/)
  // No filename and no usable content-type: bare hash, no extension.
  assert.match(mediaFileNameForBytes(bytes, null, null), /^[0-9a-f]{24}$/)
  // The byte path classifies the same way as the URL path (content-type wins).
  assert.equal(classifyMedia('application/pdf', 'invoice.pdf'), 'documents')
  assert.equal(classifyMedia('image/png', 'logo.png'), 'images')
  ok('mediaFileNameForBytes: content-addressed by bytes, extension from filename/content-type')
}

// ---- Private connector media policy + pinned transport ----------------------

{
  const cacheRoot = await mkdtemp(path.join(os.tmpdir(), 'vontra-private-media-'))
  let fetches = 0
  let sawDispatcher = false
  const fakeFetch = (async (_input: RequestInfo | URL, init?: RequestInit) => {
    fetches++
    sawDispatcher = typeof (init as RequestInit & { dispatcher?: { dispatch?: unknown } })?.dispatcher?.dispatch === 'function'
    return new Response(new Uint8Array([1, 2, 3]), { headers: { 'content-type': 'image/png' } })
  }) as typeof fetch
  try {
    const url = 'http://127.0.0.1:8065/api/v4/files/file1'
    const denied = await downloadMedia(url, { cacheRoot, fetchImpl: fakeFetch })
    assert.equal(denied.ok, false, 'public media policy rejects private/plaintext connector URLs')
    assert.equal(fetches, 0)

    const allowed = await downloadMedia(url, {
      cacheRoot,
      fetchImpl: fakeFetch,
      headers: { Authorization: 'Bearer private-token' },
      endpoint: { baseUrl: 'http://127.0.0.1:8065', label: 'Mattermost media', options: { allowPrivateNetwork: true } },
    })
    assert.equal(allowed.ok, true, allowed.error)
    assert.equal(fetches, 1)
    assert.equal(sawDispatcher, true, 'even injected transports receive the DNS-pinned dispatcher contract')

    const metadata = await downloadMedia('http://169.254.169.254/latest', {
      cacheRoot,
      fetchImpl: fakeFetch,
      endpoint: { baseUrl: 'http://169.254.169.254', label: 'Mattermost media', options: { allowPrivateNetwork: true } },
    })
    assert.equal(metadata.ok, false)
    assert.match(metadata.error ?? '', /link-local|metadata|unspecified/i)
    assert.equal(fetches, 1, 'private-network opt-in never reaches metadata')
    ok('self-hosted connector media works only with explicit opt-in, stays pinned, and still blocks metadata')
  } finally {
    await rm(cacheRoot, { recursive: true, force: true })
  }
}

// ---- Discarded-response lifecycle -------------------------------------------

{
  const cacheRoot = await mkdtemp(path.join(os.tmpdir(), 'vontra-media-discard-'))
  let cancelled = 0
  let calls = 0
  const discardedBody = () => new ReadableStream<Uint8Array>({
    start(controller) { controller.enqueue(new Uint8Array([1, 2, 3])) },
    cancel() { cancelled++ },
  })
  const fakeFetch = (async () => {
    calls++
    if (calls === 1) return new Response(discardedBody(), { status: 302, headers: { location: '/missing' } })
    return new Response(discardedBody(), { status: 404, headers: { 'content-type': 'text/plain' } })
  }) as typeof fetch
  try {
    const result = await downloadMedia('http://127.0.0.1:8065/start', {
      cacheRoot,
      fetchImpl: fakeFetch,
      endpoint: { baseUrl: 'http://127.0.0.1:8065', label: 'Mattermost media', options: { allowPrivateNetwork: true } },
    })
    assert.equal(result.ok, false)
    assert.equal(calls, 2, 'manual redirect reaches the validated next hop')
    assert.equal(cancelled, 2, 'redirect and rejected terminal response bodies are both cancelled')
    ok('downloadMedia cancels every discarded response body across redirect and error paths')
  } finally {
    await rm(cacheRoot, { recursive: true, force: true })
  }
}

// ---- Magic-byte reconciliation (content-confusion guard) ----------------------

const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4])
const SCRIPT_BYTES = Uint8Array.from('<script>alert(1)</script>', (c) => c.charCodeAt(0))

{
  // Pure predicates: script bytes declared as media are demoted to documents; genuine magic
  // bytes keep (or correct) the declared kind.
  assert.equal(detectMediaKind(PNG_BYTES), 'images')
  assert.equal(detectMediaKind(SCRIPT_BYTES), null, 'script bytes match no media signature')
  assert.equal(reconcileMediaKind('images', SCRIPT_BYTES), 'documents', 'script bytes are never an image')
  assert.equal(reconcileMediaKind('audio', SCRIPT_BYTES), 'documents', 'script bytes are never audio')
  assert.equal(reconcileMediaKind('images', PNG_BYTES), 'images', 'genuine PNG stays an image')
  assert.equal(reconcileMediaKind('documents', PNG_BYTES), 'images', 'mislabeled kind is corrected to the detected one')
  ok('reconcileMediaKind/detectMediaKind: script bytes demoted to documents, genuine images kept')
}

{
  // Attachment path (cacheBytes): script bytes with an image content-type land under
  // documents/, a genuine PNG under images/.
  const cacheRoot = await mkdtemp(path.join(os.tmpdir(), 'vontra-media-reconcile-bytes-'))
  try {
    const bad = await cacheBytes({ data: SCRIPT_BYTES, filename: 'evil.png', contentType: 'image/png' }, { cacheRoot })
    assert.equal(bad.ok, true, bad.error)
    assert.equal(bad.kind, 'documents', 'script bytes with an image content-type must not cache as an image')
    assert.ok(bad.path?.includes(`${path.sep}documents${path.sep}`), `expected documents dir, got ${bad.path}`)

    const good = await cacheBytes({ data: PNG_BYTES, filename: 'logo.png', contentType: 'image/png' }, { cacheRoot })
    assert.equal(good.ok, true, good.error)
    assert.equal(good.kind, 'images', 'genuine PNG bytes stay cached as an image')
    assert.ok(good.path?.includes(`${path.sep}images${path.sep}`), `expected images dir, got ${good.path}`)
    ok('cacheBytes: script-as-image demoted to documents, genuine PNG cached as image')
  } finally {
    await rm(cacheRoot, { recursive: true, force: true })
  }
}

{
  // URL download path (downloadMedia): the SAME reconciliation runs after streaming.
  const cacheRoot = await mkdtemp(path.join(os.tmpdir(), 'vontra-media-reconcile-url-'))
  const endpoint = { baseUrl: 'http://127.0.0.1:8065', label: 'Mattermost media', options: { allowPrivateNetwork: true } }
  const serve = (bytes: Uint8Array<ArrayBuffer>) =>
    (async () => new Response(bytes, { headers: { 'content-type': 'image/png' } })) as typeof fetch
  try {
    const bad = await downloadMedia('http://127.0.0.1:8065/files/evil', { cacheRoot, fetchImpl: serve(SCRIPT_BYTES), endpoint })
    assert.equal(bad.ok, true, bad.error)
    assert.equal(bad.kind, 'documents', 'downloaded script bytes with an image content-type must not cache as an image')
    assert.ok(bad.path?.includes(`${path.sep}documents${path.sep}`), `expected documents dir, got ${bad.path}`)

    const good = await downloadMedia('http://127.0.0.1:8065/files/logo', { cacheRoot, fetchImpl: serve(PNG_BYTES), endpoint })
    assert.equal(good.ok, true, good.error)
    assert.equal(good.kind, 'images', 'downloaded genuine PNG bytes stay cached as an image')
    assert.ok(good.path?.includes(`${path.sep}images${path.sep}`), `expected images dir, got ${good.path}`)
    ok('downloadMedia: script-as-image demoted to documents, genuine PNG cached as image')
  } finally {
    await rm(cacheRoot, { recursive: true, force: true })
  }
}

console.log(`media-cache: ${passed} checks passed`)