import assert from 'node:assert/strict'
import { mkdtemp, readFile, rm, stat, writeFile, mkdir, symlink, rename } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { gzipSync } from 'node:zlib'
import {
  archiveRelativePath,
  fetchEngine,
  localArchiveCandidates,
  parseSha256Sums,
  readTar,
  sha256,
  stagedEngineIsCurrent,
  unpackEngineArchive,
} from './fetch-engine.mjs'

function header(name, { type = '0', size = 0, mode = 0o644, link = '' } = {}) {
  const h = Buffer.alloc(512)
  h.write(name, 0, 100, 'utf8')
  h.write(mode.toString(8).padStart(7, '0') + '\0', 100, 'ascii')
  h.write('0000000\0', 108, 'ascii')
  h.write('0000000\0', 116, 'ascii')
  h.write(size.toString(8).padStart(11, '0') + '\0', 124, 'ascii')
  h.write('00000000000\0', 136, 'ascii')
  h.write(type, 156, 'ascii')
  h.write(link, 157, 100, 'utf8')
  h.write('ustar\0', 257, 'ascii')
  h.write('00', 263, 'ascii')
  h.fill(0x20, 148, 156)
  let sum = 0
  for (const byte of h) sum += byte
  h.write(sum.toString(8).padStart(6, '0') + '\0 ', 148, 'ascii')
  return h
}

function tar(entries) {
  const blocks = []
  for (const entry of entries) {
    const data = entry.data ? Buffer.from(entry.data) : Buffer.alloc(0)
    blocks.push(header(entry.name, { type: entry.type ?? '0', size: data.length, mode: entry.mode, link: entry.link }))
    if (data.length) blocks.push(data, Buffer.alloc((512 - (data.length % 512)) % 512))
  }
  blocks.push(Buffer.alloc(1024))
  return Buffer.concat(blocks)
}

const root = 'tensorfold-9.9.9-macos-arm64'
const payload = {
  'bin/tensorfold-native': '#!/bin/sh\necho tensorfold-native 9.9.9\n',
  'lib/libtfgrammar.dylib': 'grammar',
  LICENSE: 'license',
  NOTICE: 'notice',
  'THIRD_PARTY_NOTICES.md': 'notices',
  VERSION: '9.9.9\n',
}

function sums(files) {
  return Object.entries(files).map(([name, data]) => `${sha256(Buffer.from(data))}  ./${name}`).join('\n') + '\n'
}

function archive({ files = payload, extra = [], sumsText } = {}) {
  const entries = [{ name: `${root}/`, type: '5' }, { name: `${root}/bin/`, type: '5' }]
  for (const [name, data] of Object.entries(files)) entries.push({ name: `${root}/${name}`, data })
  entries.push({ name: `${root}/SHA256SUMS`, data: sumsText ?? sums(files) })
  entries.push(...extra)
  return gzipSync(tar(entries))
}

function manifestFor(bytes, overrides = {}) {
  return {
    version: '9.9.9',
    archive: 'engine.tar.gz',
    root,
    url: 'https://example.invalid/engine.tar.gz',
    sha256: sha256(bytes),
    required: ['bin/tensorfold-native', 'lib/libtfgrammar.dylib', 'VERSION', 'SHA256SUMS'],
    executables: ['bin/tensorfold-native', 'lib/libtfgrammar.dylib'],
    ...overrides,
  }
}

// Paths: only the archive root, never absolute, parent or backslash paths.
assert.equal(archiveRelativePath(`${root}/bin/tensorfold-native`, root), 'bin/tensorfold-native')
assert.equal(archiveRelativePath(`${root}/`, root), '')
for (const bad of ['/etc/passwd', `${root}/../x`, `${root}/./x`, `${root}\\x`, 'other/x', `${root}//x`, 'C:/x', '']) {
  assert.throws(() => archiveRelativePath(bad, root), /unsafe path|outside/, bad)
}

assert.throws(() => parseSha256Sums('nonsense\n'), /malformed/)
assert.throws(() => parseSha256Sums(`${'a'.repeat(64)}  ./../x\n`), /unsafe path/)
assert.equal(parseSha256Sums(`${'a'.repeat(64)}  ./bin/x\n`).get('bin/x'), 'a'.repeat(64))

const good = archive()
const files = unpackEngineArchive(good, manifestFor(good))
assert.equal(files.get('VERSION').toString(), '9.9.9\n')
assert.ok(files.has('bin/tensorfold-native'))

assert.throws(() => unpackEngineArchive(good, manifestFor(good, { sha256: 'b'.repeat(64) })), /does not match the pinned/)

const symlinked = archive({ extra: [{ name: `${root}/bin/evil`, type: '2', link: '/bin/sh' }] })
assert.throws(() => unpackEngineArchive(symlinked, manifestFor(symlinked)), /not a regular file or directory \(tar type 2\)/)
const hardlinked = archive({ extra: [{ name: `${root}/bin/evil`, type: '1', link: `${root}/bin/tensorfold-native` }] })
assert.throws(() => unpackEngineArchive(hardlinked, manifestFor(hardlinked)), /tar type 1/)
const escaping = archive({ extra: [{ name: `${root}/../escape`, data: 'x' }] })
assert.throws(() => unpackEngineArchive(escaping, manifestFor(escaping)), /unsafe path/)
const unlisted = archive({ extra: [{ name: `${root}/bin/extra`, data: 'x' }] })
assert.throws(() => unpackEngineArchive(unlisted, manifestFor(unlisted)), /not listed in SHA256SUMS/)
const tampered = archive({ sumsText: sums({ ...payload, 'bin/tensorfold-native': 'other' }) })
assert.throws(() => unpackEngineArchive(tampered, manifestFor(tampered)), /does not match SHA256SUMS/)
const wrongVersion = archive({ files: { ...payload, VERSION: '1.0.0\n' } })
assert.throws(() => unpackEngineArchive(wrongVersion, manifestFor(wrongVersion)), /VERSION 1.0.0 does not match/)
const missingDylib = archive({ files: Object.fromEntries(Object.entries(payload).filter(([name]) => name !== 'lib/libtfgrammar.dylib')) })
assert.throws(() => unpackEngineArchive(missingDylib, manifestFor(missingDylib)), /missing required lib\/libtfgrammar.dylib/)

const corrupt = tar([{ name: `${root}/VERSION`, data: 'x' }])
corrupt[150] = 0x37
assert.throws(() => readTar(corrupt), /corrupt tar header/)

// End to end through a temp repo root: env archive, staging, modes, idempotent re-run.
const repo = await mkdtemp(path.join(os.tmpdir(), 'fetch-engine-test-'))
try {
  await mkdir(path.join(repo, 'build'), { recursive: true })
  const manifest = manifestFor(good)
  await writeFile(path.join(repo, 'build', 'engine.json'), JSON.stringify(manifest))
  const archivePath = path.join(repo, 'engine.tar.gz')
  await writeFile(archivePath, good)
  const logs = []
  const noNetwork = async () => { throw new Error('tests must not download') }
  const first = await fetchEngine({ root: repo, env: { TENSORFOLD_ENGINE_ARCHIVE: archivePath }, fetchImpl: noNetwork, log: (line) => logs.push(line) })
  assert.equal(first.source, 'TENSORFOLD_ENGINE_ARCHIVE')
  const binary = await stat(path.join(repo, 'build', 'engine', 'bin', 'tensorfold-native'))
  assert.equal(binary.mode & 0o777, 0o755)
  assert.equal((await stat(path.join(repo, 'build', 'engine', 'NOTICE'))).mode & 0o777, 0o644)
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), true)
  const second = await fetchEngine({ root: repo, env: {}, fetchImpl: noNetwork, log: () => {} })
  assert.equal(second.source, 'staged')

  await writeFile(path.join(repo, 'build', 'engine', 'bin', 'tensorfold-native'), 'tampered')
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), false, 'a modified staged binary is restaged')

  // Neither self-authored checksums nor hidden extra paths are release provenance.
  await writeFile(path.join(repo, 'build', 'engine', 'SHA256SUMS'), sums({ ...payload, 'bin/tensorfold-native': 'tampered' }))
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), false)
  await fetchEngine({ root: repo, env: { TENSORFOLD_ENGINE_ARCHIVE: archivePath }, force: true, fetchImpl: noNetwork, log: () => {} })
  const engine = path.join(repo, 'build', 'engine')
  await writeFile(path.join(engine, 'injected'), 'not in archive')
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), false)
  await rm(path.join(engine, 'injected'))
  await mkdir(path.join(engine, 'unexpected-empty-directory'))
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), false)
  await rm(path.join(engine, 'unexpected-empty-directory'), { recursive: true })
  await rename(path.join(engine, 'bin'), path.join(repo, 'outside-bin'))
  await symlink(path.join(repo, 'outside-bin'), path.join(engine, 'bin'))
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), false)
  await rm(path.join(engine, 'bin'))
  await rename(path.join(repo, 'outside-bin'), path.join(engine, 'bin'))
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), true)
  const cachedArchive = path.join(repo, 'build', '.engine-cache', manifest.archive)
  await writeFile(cachedArchive, 'untrusted cache')
  assert.equal(await stagedEngineIsCurrent(path.join(repo, 'build'), manifest), false)
  await writeFile(cachedArchive, good)
  await writeFile(path.join(engine, 'bin', 'tensorfold-native'), 'tampered')

  await assert.rejects(
    () => fetchEngine({ root: repo, env: { TENSORFOLD_ENGINE_ARCHIVE: path.join(repo, 'missing.tar.gz') }, fetchImpl: noNetwork, log: () => {} }),
    /not a readable file/,
  )
  const marker = JSON.parse(await readFile(path.join(repo, 'build', '.engine-staged.json'), 'utf8'))
  assert.deepEqual(marker, { version: '9.9.9', sha256: manifest.sha256 })

  // Without the env archive, the only other source is the pinned URL (no folders outside the repo).
  assert.deepEqual(localArchiveCandidates(manifest, {}, path.join(repo, 'build')).map((c) => c.label), ['download cache'])
  await rm(path.join(repo, 'build', '.engine-staged.json'))
  await rm(path.join(repo, 'build', '.engine-cache'), { recursive: true, force: true })
  const fetched = []
  const fromUrl = await fetchEngine({ root: repo, env: {}, fetchImpl: async (url) => { fetched.push(url); return new Response(good) }, log: () => {} })
  assert.equal(fromUrl.source, 'download')
  assert.deepEqual(fetched, [manifest.url])
  await rm(path.join(repo, 'build', '.engine-staged.json'))
  const badBytes = Buffer.from(good); badBytes[badBytes.length - 1] ^= 1
  await rm(path.join(repo, 'build', '.engine-cache'), { recursive: true, force: true })
  await assert.rejects(() => fetchEngine({ root: repo, env: {}, fetchImpl: async () => new Response(badBytes), log: () => {} }), /does not match the pinned sha256/)
} finally {
  await rm(repo, { recursive: true, force: true })
}

console.log('fetch-engine tests passed')
