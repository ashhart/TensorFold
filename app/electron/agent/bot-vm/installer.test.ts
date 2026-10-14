import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { gzipSync } from 'node:zlib'
import { createHash } from 'node:crypto'
import { APPLE_VM_PACKAGE, BotVmInstaller, extractVmPayload } from './installer'
import type { VmRunner } from './service'

const digest = (data: Buffer | string) => createHash('sha256').update(data).digest('hex')
function cpio(entries: { name: string; data?: string; mode?: number; links?: number }[]): Buffer {
  return Buffer.concat([...entries, { name: 'TRAILER!!!', mode: 0 }].flatMap((entry, index) => {
    const name = Buffer.from(entry.name + '\0'), data = Buffer.from(entry.data ?? '')
    const octal = (n: number, length: number) => n.toString(8).padStart(length, '0')
    const header = '070707' + octal(0, 6) + octal(index, 6) + octal(entry.mode ?? 0o100644, 6) + octal(0, 6) + octal(0, 6) + octal(entry.links ?? 1, 6) + octal(0, 6) + octal(0, 11) + octal(name.length, 6) + octal(data.length, 11)
    assert.equal(header.length, 76); return [Buffer.from(header), name, data]
  }))
}
const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-vm-installer-test-'))
try {
  const files = { 'bin/container': { sha256: digest('cli fixture'), bytes: 11, signed: true }, 'bin/container-apiserver': { sha256: digest('helper fixture'), bytes: 14, signed: true } }
  const packageBytes = Buffer.from('fictional signed package fixture')
  const release = { ...APPLE_VM_PACKAGE, bytes: packageBytes.length, sha256: digest(packageBytes) }
  const entries = [{ name: '.', mode: 0o040755, links: 2 }, { name: './bin', mode: 0o040755, links: 2 }, { name: './bin/container', data: 'cli fixture' }, { name: './bin/container-apiserver', data: 'helper fixture' }]
  const payload = gzipSync(cpio(entries))
  let rejectSigner = false, failActivation = false, signatureCalls = 0
  const runner: VmRunner = async (binary, args) => {
    if (binary.endsWith('codesign')) { signatureCalls++; return { code: failActivation && args.at(-1)!.includes('apple-1.5.0') ? 1 : 0, stdout: '', stderr: '' } }
    if (args[0] === '--check-signature') return { code: 0, stdout: rejectSigner ? 'signed by fictional publisher' : `Status: signed by a developer certificate issued by Apple\nNotarization: trusted by the Apple notary service\nDeveloper ID Installer: Apple Inc. - Containerization (${APPLE_VM_PACKAGE.team})`, stderr: '' }
    if (binary.endsWith('xar')) return { code: 0, stdout: 'Bom\nPayload\nPackageInfo\n', stderr: '' }
    if (args[0] === '--expand') {
      await fs.mkdir(args[2]); await fs.writeFile(path.join(args[2], 'Bom'), 'fixture')
      await fs.writeFile(path.join(args[2], 'PackageInfo'), '<pkg-info identifier="com.apple.container-installer" version="1.5.0"/>')
      await fs.writeFile(path.join(args[2], 'Payload'), payload)
    }
    return { code: 0, stdout: '', stderr: '' }
  }
  const fetcher = (async () => new Response(packageBytes, { headers: { 'content-length': String(packageBytes.length) } })) as typeof fetch
  const install = (name: string, download = fetcher) => new BotVmInstaller(path.join(root, name), runner, download, { release, files, freeDiskBytes: async () => 100 * 1024 ** 3 })
  const success = install('success'), runtime = await success.install()
  assert.equal(await fs.readFile(runtime.binary, 'utf8'), 'cli fixture')
  assert.ok(signatureCalls >= 4, 'both executable helpers are checked before and after activation')
  const calls = signatureCalls; await success.install(); assert.ok(signatureCalls > calls, 'cached components are rechecked')
  assert.deepEqual((await fs.readdir(success.root)).sort(), ['apple-1.5.0', 'data'])
  await fs.writeFile(path.join(runtime.installRoot, 'unexpected-plugin'), 'bad')
  await assert.rejects(success.verifyInstalled(), /Unexpected file/)
  await fs.rm(path.join(runtime.installRoot, 'unexpected-plugin'))
  const lowDisk = new BotVmInstaller(path.join(root, 'low-disk'), runner, fetcher, { release, files, freeDiskBytes: async () => 1024 })
  await assert.rejects(lowDisk.install(), /6 GiB/)
  const badDigest = install('bad-digest', (async () => new Response(Buffer.alloc(packageBytes.length))) as typeof fetch)
  await assert.rejects(badDigest.install(), /checksum/)
  assert.equal(badDigest.snapshot().phase, 'error')
  rejectSigner = true; await assert.rejects(install('bad-signer').install(), /expected Apple publisher/); rejectSigner = false
  await assert.rejects(install('oversize', (async () => new Response(Buffer.alloc(packageBytes.length + 1))) as typeof fetch).install(), /size limit/)
  await assert.rejects(install('bad-header', (async () => new Response(packageBytes, { headers: { 'content-length': '1' } })) as typeof fetch).install(), /download size/)
  await assert.rejects(install('redirect', (async () => new Response(null, { status: 302, headers: { location: 'https://untrusted.invalid/package' } })) as typeof fetch).install(), /approved release hosts/)
  const aborted = new AbortController(); aborted.abort(); await assert.rejects(install('cancelled').install(aborted.signal), /cancelled/)
  const streamingAbort = new AbortController(); let streamStarted = false
  const streaming = install('streaming-cancel', (async (_url: unknown, options?: RequestInit) => new Response(new ReadableStream<Uint8Array>({ start(controller) {
    controller.enqueue(packageBytes.subarray(0, 4)); streamStarted = true
    options?.signal?.addEventListener('abort', () => controller.error(new Error('fixture download aborted')), { once: true })
  } }))) as typeof fetch)
  const streamingPending = streaming.install(streamingAbort.signal)
  while (!streamStarted) await new Promise((resolve) => setTimeout(resolve, 1))
  streamingAbort.abort(); await assert.rejects(streamingPending, /aborted|cancelled/)
  assert.equal(streaming.snapshot().phase, 'cancelled')
  assert.equal(streaming.snapshot().active, false)
  assert.deepEqual(await fs.readdir(streaming.root), [], 'stream cancellation removes partial packages and staging files')
  failActivation = true
  const failed = install('failed-activation'); await assert.rejects(failed.install(), /helper failed signature/)
  assert.equal(await fs.access(failed.paths().installRoot).then(() => true, () => false), false)
  assert.deepEqual(await fs.readdir(failed.root), ['data'], 'failed activation leaves no runtime or temporary downloads')
  failActivation = false; await failed.install()
  for (const [name, invalid] of [
    ['traversal', [{ name: '../escaped', data: 'bad' }]],
    ['absolute', [{ name: '/escaped', data: 'bad' }]],
    ['symlink', [{ name: 'bin/container', mode: 0o120777, data: '/outside' }]],
    ['hardlink', [{ name: 'bin/container', data: 'cli fixture', links: 2 }]],
    ['privileged', [{ name: 'bin/container', data: 'cli fixture', mode: 0o104755 }]],
    ['duplicate', [...entries, entries[2]]],
    ['checksum', [{ name: 'bin/container', data: 'bad fixture' }]],
  ] as const) {
    const archive = path.join(root, `${name}.gz`); await fs.writeFile(archive, gzipSync(cpio([...invalid])))
    await assert.rejects(extractVmPayload(archive, path.join(root, `extract-${name}`), undefined, files))
  }
  const linked = path.join(root, 'linked'); await fs.mkdir(linked); await fs.symlink(root, path.join(linked, 'bin'))
  const validArchive = path.join(root, 'valid.gz'); await fs.writeFile(validArchive, payload)
  await assert.rejects(extractVmPayload(validArchive, linked, undefined, files), /Linked runtime/)
  console.log('managed VM installer pin, publisher, bounds, traversal, cancellation and atomic rollback checks passed')
} finally { await fs.rm(root, { recursive: true, force: true }) }
