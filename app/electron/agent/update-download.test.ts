import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { downloadVerifiedUpdate, reverifyDownloadedUpdate } from './update-download'
import type { PinnedHostResolver } from './pinned-fetch'

const PUBLIC_IP = '93.184.216.34'
const bytes = Buffer.from('verified update artifact')
const checksum = { algorithm: 'sha256' as const, value: createHash('sha256').update(bytes).digest('hex') }

function response(body: BodyInit | null, init: ResponseInit = {}): Response {
  return new Response(body, { status: 200, ...init })
}

function resolver(log: string[] = []): PinnedHostResolver {
  return async (hostname) => {
    log.push(hostname)
    if (hostname === 'private.example') return [{ address: '10.0.0.8', family: 4 }]
    return [{ address: PUBLIC_IP, family: 4 }]
  }
}

async function tempRoot(): Promise<string> {
  return fs.mkdtemp(path.join(os.tmpdir(), 'vontra-update-download-test-'))
}

async function assertNoArtifacts(root: string): Promise<void> {
  assert.deepEqual(await fs.readdir(root), [], 'failed downloads remove partial and final artifacts')
}

async function main() {
  {
    const root = await tempRoot()
    const progress: Array<{ receivedBytes: number; totalBytes: number | null }> = []
    try {
      await downloadVerifiedUpdate({ downloadUrl: 'https://example.com/update.zip', checksum, manifestSignatureVerified: true, downloadRoot: root, onProgress: (p) => progress.push(p) }, {
        lookup: resolver(), fetchImpl: (async () => response(bytes, { headers: { 'content-length': String(bytes.length) } })) as typeof fetch,
      })
      assert.deepEqual(progress[0], { receivedBytes: 0, totalBytes: bytes.length })
      assert.deepEqual(progress.at(-1), { receivedBytes: bytes.length, totalBytes: bytes.length }, 'download progress counts the bytes actually written')
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }
  {
    const root = await tempRoot()
    try {
      await assert.rejects(() => downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/TensorFold-Studio.dmg', checksum,
        manifestSignatureVerified: false, downloadRoot: root,
      }), /manifest signature is not verified/i)
      await assertNoArtifacts(root)
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    try {
      const tampered = Buffer.from('tampered update artifact')
      await assert.rejects(() => downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/TensorFold-Studio.dmg', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, {
        lookup: resolver(),
        fetchImpl: (async () => response(tampered)) as typeof fetch,
      }), /SHA-256 mismatch/i)
      await assertNoArtifacts(root)
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    try {
      let calls = 0
      await assert.rejects(() => downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/start', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, {
        lookup: resolver(),
        fetchImpl: (async () => {
          calls++
          return new Response(null, { status: 302, headers: { location: 'https://127.0.0.1/metadata' } })
        }) as typeof fetch,
      }), /private|local|non-public/i)
      assert.equal(calls, 1, 'private redirect is rejected before a second socket opens')
      await assertNoArtifacts(root)
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    try {
      let calls = 0
      await assert.rejects(() => downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/start', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, {
        lookup: resolver(),
        fetchImpl: (async () => {
          calls++
          return new Response(null, { status: 302, headers: { location: 'https://private.example/artifact' } })
        }) as typeof fetch,
      }), /private or non-public address/i)
      assert.equal(calls, 1, 'non-global DNS redirect is rejected before fetch')
      await assertNoArtifacts(root)
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    try {
      await assert.rejects(() => downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/TensorFold-Studio.dmg', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, {
        maxBytes: 5,
        lookup: resolver(),
        fetchImpl: (async () => response('tiny', { headers: { 'content-length': '6' } })) as typeof fetch,
      }), /exceeds the 5-byte download limit/i)
      await assertNoArtifacts(root)
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    try {
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new Uint8Array([1, 2, 3]))
          controller.enqueue(new Uint8Array([4, 5, 6]))
          controller.close()
        },
      })
      await assert.rejects(() => downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/TensorFold-Studio.dmg', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, {
        maxBytes: 5,
        lookup: resolver(),
        fetchImpl: (async () => response(stream)) as typeof fetch,
      }), /exceeds the 5-byte download limit/i)
      await assertNoArtifacts(root)
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    const lookups: string[] = []
    try {
      const fetchImpl = (async (input: Parameters<typeof fetch>[0]) => {
        const url = new URL(input instanceof Request ? input.url : String(input))
        if (url.hostname === 'download.example') {
          return new Response(null, { status: 302, headers: { location: 'https://cdn.example/releases/TensorFold-Studio-1.2.3-arm64.dmg' } })
        }
        return response(bytes, { headers: { 'content-length': String(bytes.length) } })
      }) as typeof fetch
      const result = await downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/start', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, { lookup: resolver(lookups), fetchImpl, randomId: () => 'fixed' })
      assert.equal(result.sha256, checksum.value)
      assert.equal(result.bytes, bytes.length)
      assert.equal(result.redirects, 1)
      assert.equal(result.finalUrl, 'https://cdn.example/releases/TensorFold-Studio-1.2.3-arm64.dmg')
      assert.equal(await fs.readFile(result.path, 'utf8'), bytes.toString())
      assert.ok(result.path.endsWith('TensorFold-Studio-1.2.3-arm64.dmg'))
      assert.deepEqual(lookups, ['download.example', 'cdn.example'], 'every redirect hop is independently resolved and pinned')
      assert.ok(!(await fs.readdir(path.dirname(result.path))).some((name) => name.endsWith('.part')), 'atomic final directory has no partial file')
      const reverified = await reverifyDownloadedUpdate(result, root)
      assert.equal(reverified.sha256, checksum.value)
      assert.equal((await fs.stat(result.path)).mode & 0o777, 0o600, 'verified artifact remains owner-only')

      await fs.writeFile(result.path, Buffer.from('tampered update artifact'))
      await assert.rejects(
        () => reverifyDownloadedUpdate(result, root),
        /no longer matches/i,
        'same-inode mutation after download is detected before open',
      )
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    try {
      const result = await downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/TensorFold-Studio.dmg', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, {
        lookup: resolver(),
        fetchImpl: (async () => response(bytes)) as typeof fetch,
      })
      const replacement = path.join(path.dirname(result.path), 'replacement.dmg')
      await fs.writeFile(replacement, bytes, { mode: 0o600 })
      await fs.rename(replacement, result.path)
      await assert.rejects(
        () => reverifyDownloadedUpdate(result, root),
        /identity changed after download/i,
        'an identical path replacement is rejected by device/inode identity',
      )
    } finally { await fs.rm(root, { recursive: true, force: true }) }
  }

  {
    const root = await tempRoot()
    const outside = path.join(os.tmpdir(), `vontra-update-outside-${Date.now()}.dmg`)
    try {
      const result = await downloadVerifiedUpdate({
        downloadUrl: 'https://download.example/TensorFold-Studio.dmg', checksum,
        manifestSignatureVerified: true, downloadRoot: root,
      }, {
        lookup: resolver(),
        fetchImpl: (async () => response(bytes)) as typeof fetch,
      })
      await fs.writeFile(outside, bytes)
      await fs.unlink(result.path)
      await fs.symlink(outside, result.path)
      await assert.rejects(
        () => reverifyDownloadedUpdate(result, root),
        /no longer a regular file/i,
        'a symlink swap is rejected before the OS opens the update',
      )
    } finally {
      await fs.rm(root, { recursive: true, force: true })
      await fs.rm(outside, { force: true })
    }
  }

  console.log('verified update download tests passed')
}

void main()
