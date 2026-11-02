#!/usr/bin/env node
// Stages the pinned TensorFold engine archive into build/engine/ for packaging.
// Sources: TENSORFOLD_ENGINE_ARCHIVE, or the pinned release URL (kept in build/.engine-cache after a
// download). Every source must match the pinned sha256 and the archive's own SHA256SUMS; symlinks,
// hard links and paths outside the archive root are refused.
import { createHash } from 'node:crypto'
import { chmod, lstat, mkdir, mkdtemp, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { gunzipSync } from 'node:zlib'

const MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
const MAX_UNPACKED_BYTES = 256 * 1024 * 1024
const STAGED_MARKER = '.engine-staged.json'

export function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex')
}

export function engineTarget(platform = process.platform, arch = process.arch) {
  if (platform === 'darwin' && arch === 'arm64') return 'macos-arm64'
  if (platform === 'linux' && arch === 'x64') return 'linux-x64'
  if (platform === 'linux' && arch === 'arm64') return 'linux-arm64'
  throw new Error(`No pinned engine archive is available for ${platform}/${arch}.`)
}

export async function readEngineManifest(root, target = 'macos-arm64') {
  if (!['macos-arm64', 'linux-x64', 'linux-arm64'].includes(target)) throw new Error('Unknown engine target.')
  const filename = target === 'macos-arm64' ? 'engine.json' : `engine-${target}.json`
  const manifest = JSON.parse(await readFile(path.join(root, 'build', filename), 'utf8'))
  for (const key of ['version', 'archive', 'root', 'url', 'sha256']) {
    if (typeof manifest[key] !== 'string' || !manifest[key]) throw new Error(`build/engine.json is missing ${key}.`)
  }
  if (!/^[a-f0-9]{64}$/.test(manifest.sha256)) throw new Error('build/engine.json sha256 must be 64 lowercase hex characters.')
  if (!manifest.url.startsWith('https://')) throw new Error('build/engine.json url must use https.')
  if (!Array.isArray(manifest.required) || !manifest.required.includes('bin/tensorfold-native')) throw new Error('build/engine.json must require bin/tensorfold-native.')
  return manifest
}

function octal(field) {
  const text = field.toString('latin1').replace(/\0.*$/s, '').trim()
  if (!text) return 0
  if (!/^[0-7]+$/.test(text)) throw new Error('Engine archive has a malformed tar number.')
  return Number.parseInt(text, 8)
}

function cString(field) {
  return field.toString('utf8').replace(/\0.*$/s, '')
}

function parsePax(body) {
  const out = {}
  let offset = 0
  while (offset < body.length) {
    const space = body.indexOf(0x20, offset)
    if (space < 0) break
    const length = Number.parseInt(body.subarray(offset, space).toString('ascii'), 10)
    if (!Number.isSafeInteger(length) || length <= 0) throw new Error('Engine archive has a malformed pax header.')
    const record = body.subarray(space + 1, offset + length - 1).toString('utf8')
    const eq = record.indexOf('=')
    if (eq > 0) out[record.slice(0, eq)] = record.slice(eq + 1)
    offset += length
  }
  return out
}

// Minimal ustar reader: regular files and directories only. Anything else (symlinks, hard links,
// devices, fifos) is an error rather than something to skip.
export function readTar(buffer) {
  const entries = []
  let offset = 0
  let pending = {}
  let total = 0
  while (offset + 512 <= buffer.length) {
    const header = buffer.subarray(offset, offset + 512)
    if (header.every((byte) => byte === 0)) break
    const stored = octal(header.subarray(148, 156))
    let sum = 0
    for (let i = 0; i < 512; i++) sum += i >= 148 && i < 156 ? 0x20 : header[i]
    if (sum !== stored) throw new Error('Engine archive has a corrupt tar header.')
    const type = String.fromCharCode(header[156] || 0x30)
    const size = octal(header.subarray(124, 136))
    const body = buffer.subarray(offset + 512, offset + 512 + size)
    if (body.length !== size) throw new Error('Engine archive is truncated.')
    offset += 512 + Math.ceil(size / 512) * 512
    if (type === 'x') { pending = { ...pending, ...parsePax(body) }; continue }
    if (type === 'g') continue
    if (type === 'L') { pending = { ...pending, path: cString(body) }; continue }
    const prefix = cString(header.subarray(345, 500))
    const name = cString(header.subarray(0, 100))
    const entryPath = pending.path ?? (prefix ? `${prefix}/${name}` : name)
    pending = {}
    if (type !== '0' && type !== '5') throw new Error(`Engine archive entry ${entryPath} is not a regular file or directory (tar type ${type}).`)
    total += size
    if (total > MAX_UNPACKED_BYTES) throw new Error('Engine archive unpacks to more than the size limit.')
    entries.push({ path: entryPath, type: type === '5' ? 'dir' : 'file', mode: octal(header.subarray(100, 108)), data: type === '0' ? Buffer.from(body) : null })
  }
  return entries
}

// Archive path -> path relative to the archive root, or an error for anything that could escape.
export function archiveRelativePath(entryPath, rootName) {
  const raw = String(entryPath)
  if (!raw || raw.includes('\0') || raw.includes('\\') || raw.startsWith('/') || /^[A-Za-z]:/.test(raw)) {
    throw new Error(`Engine archive entry ${JSON.stringify(raw)} has an unsafe path.`)
  }
  const trimmed = raw.replace(/\/+$/, '')
  const parts = trimmed.split('/')
  if (parts.some((part) => part === '' || part === '.' || part === '..')) throw new Error(`Engine archive entry ${JSON.stringify(raw)} has an unsafe path.`)
  if (parts[0] !== rootName) throw new Error(`Engine archive entry ${JSON.stringify(raw)} is outside ${rootName}/.`)
  return parts.slice(1).join('/')
}

export function parseSha256Sums(text) {
  const sums = new Map()
  for (const line of String(text).split(/\r?\n/)) {
    if (!line.trim()) continue
    const match = /^([a-f0-9]{64}) [ *](?:\.\/)?(.+)$/.exec(line)
    if (!match) throw new Error(`SHA256SUMS has a malformed line: ${line.slice(0, 120)}`)
    const file = match[2]
    if (file.startsWith('/') || file.split('/').some((part) => part === '..' || part === '.' || part === '')) throw new Error(`SHA256SUMS names an unsafe path: ${file}`)
    if (sums.has(file)) throw new Error(`SHA256SUMS lists ${file} twice.`)
    sums.set(file, match[1])
  }
  if (!sums.size) throw new Error('SHA256SUMS is empty.')
  return sums
}

// Verifies a gzip tarball against the manifest and returns its files keyed by relative path.
export function unpackEngineArchive(archiveBytes, manifest) {
  const actual = sha256(archiveBytes)
  if (actual !== manifest.sha256) throw new Error(`Engine archive sha256 ${actual} does not match the pinned ${manifest.sha256}.`)
  const files = new Map()
  for (const entry of readTar(gunzipSync(archiveBytes, { maxOutputLength: MAX_UNPACKED_BYTES }))) {
    const rel = archiveRelativePath(entry.path, manifest.root)
    if (entry.type === 'dir') continue
    if (!rel) throw new Error('Engine archive root is a file.')
    if (files.has(rel)) throw new Error(`Engine archive contains ${rel} twice.`)
    files.set(rel, entry.data)
  }
  verifyEngineFiles(files, manifest)
  return files
}

export function verifyEngineFiles(files, manifest) {
  const sumsBytes = files.get('SHA256SUMS')
  if (!sumsBytes) throw new Error('Engine archive has no SHA256SUMS.')
  const sums = parseSha256Sums(sumsBytes.toString('utf8'))
  for (const [file, expected] of sums) {
    const bytes = files.get(file)
    if (!bytes) throw new Error(`Engine archive is missing ${file}, which SHA256SUMS lists.`)
    if (sha256(bytes) !== expected) throw new Error(`Engine file ${file} does not match SHA256SUMS.`)
  }
  for (const file of files.keys()) {
    if (file !== 'SHA256SUMS' && !sums.has(file)) throw new Error(`Engine archive file ${file} is not listed in SHA256SUMS.`)
  }
  for (const file of manifest.required ?? []) {
    if (!files.has(file)) throw new Error(`Engine archive is missing required ${file}.`)
  }
  const version = files.get('VERSION')?.toString('utf8').trim()
  if (version !== manifest.version) throw new Error(`Engine archive VERSION ${version || '(missing)'} does not match the pinned ${manifest.version}.`)
}

export async function writeEngineTree(files, manifest, buildDir) {
  await mkdir(buildDir, { recursive: true })
  const staging = await mkdtemp(path.join(buildDir, '.engine-staging-'))
  try {
    const executables = new Set(manifest.executables ?? [])
    for (const [rel, bytes] of files) {
      const target = path.join(staging, rel)
      if (path.relative(staging, target).startsWith('..')) throw new Error(`Refusing to write ${rel} outside the staging folder.`)
      await mkdir(path.dirname(target), { recursive: true })
      await writeFile(target, bytes, { mode: 0o644, flag: 'wx' })
      await chmod(target, executables.has(rel) ? 0o755 : 0o644)
    }
    await chmod(staging, 0o755)
    const destination = path.join(buildDir, 'engine')
    await rm(destination, { recursive: true, force: true })
    await rename(staging, destination)
    await writeFile(path.join(buildDir, STAGED_MARKER), `${JSON.stringify({ version: manifest.version, sha256: manifest.sha256 }, null, 2)}\n`)
    return destination
  } catch (error) {
    await rm(staging, { recursive: true, force: true }).catch(() => {})
    throw error
  }
}

// The marker is only a hint: the immutable archive digest is the trust anchor.
// Read every staged path, including empty directories, without following links.
export async function stagedEngineIsCurrent(buildDir, manifest) {
  try {
    const marker = JSON.parse(await readFile(path.join(buildDir, STAGED_MARKER), 'utf8'))
    if (marker.version !== manifest.version || marker.sha256 !== manifest.sha256) return false
    const archive = await readIfFile(path.join(buildDir, '.engine-cache', manifest.archive))
    if (!archive) return false
    const expected = unpackEngineArchive(archive, manifest)
    const directories = new Set([''])
    for (const file of expected.keys()) {
      let parent = path.posix.dirname(file)
      while (parent !== '.') { directories.add(parent); parent = path.posix.dirname(parent) }
    }
    const engineDir = path.join(buildDir, 'engine')
    const seen = new Set()
    async function inspect(rel) {
      const full = path.join(engineDir, rel)
      const info = await lstat(full)
      if (info.isDirectory()) {
        if (!directories.has(rel)) throw new Error('Unexpected engine directory.')
        for (const name of await readdir(full)) await inspect(rel ? `${rel}/${name}` : name)
      } else if (info.isFile()) {
        const bytes = expected.get(rel)
        if (!bytes || info.size !== bytes.length || sha256(await readFile(full)) !== sha256(bytes)) throw new Error('Engine file differs from the pinned archive.')
        const mode = (manifest.executables ?? []).includes(rel) ? 0o755 : 0o644
        if ((info.mode & 0o777) !== mode || info.nlink !== 1) throw new Error('Unexpected engine file mode or hard link.')
        seen.add(rel)
      } else {
        throw new Error('Engine paths must be regular files or directories.')
      }
    }
    await inspect('')
    return seen.size === expected.size
  } catch {
    return false
  }
}

async function readIfFile(file) {
  try {
    const info = await lstat(file)
    if (!info.isFile()) return null
    if (info.size > MAX_ARCHIVE_BYTES) throw new Error(`${file} is larger than the engine archive limit.`)
    return await readFile(file)
  } catch (error) {
    if (error?.code === 'ENOENT') return null
    throw error
  }
}

export function localArchiveCandidates(manifest, env, buildDir) {
  const out = []
  if (env.TENSORFOLD_ENGINE_ARCHIVE?.trim()) out.push({ file: path.resolve(env.TENSORFOLD_ENGINE_ARCHIVE.trim()), label: 'TENSORFOLD_ENGINE_ARCHIVE', required: true })
  out.push({ file: path.join(buildDir, '.engine-cache', manifest.archive), label: 'download cache', required: false })
  return out
}

async function download(manifest, buildDir, fetchImpl) {
  const res = await fetchImpl(manifest.url, { redirect: 'follow' })
  if (!res.ok) throw new Error(`Engine download failed: HTTP ${res.status}.`)
  const bytes = Buffer.from(await res.arrayBuffer())
  if (bytes.length > MAX_ARCHIVE_BYTES) throw new Error('Engine download is larger than the archive limit.')
  if (sha256(bytes) !== manifest.sha256) throw new Error('Downloaded engine archive does not match the pinned sha256.')
  const cache = path.join(buildDir, '.engine-cache')
  await mkdir(cache, { recursive: true })
  await writeFile(path.join(cache, manifest.archive), bytes)
  return bytes
}

export async function fetchEngine({ root, target, env = process.env, force = false, fetchImpl = fetch, log = console.log } = {}) {
  const manifest = await readEngineManifest(root, target)
  const buildDir = path.join(root, 'build')
  if (!force && await stagedEngineIsCurrent(buildDir, manifest)) {
    log(`[fetch-engine] TensorFold engine ${manifest.version} already staged in build/engine.`)
    return { manifest, source: 'staged' }
  }
  let bytes = null
  let source = ''
  for (const candidate of localArchiveCandidates(manifest, env, buildDir)) {
    bytes = await readIfFile(candidate.file)
    if (!bytes && candidate.required) throw new Error(`TENSORFOLD_ENGINE_ARCHIVE points at ${candidate.file}, which is not a readable file.`)
    if (bytes) { source = candidate.label; break }
  }
  if (!bytes) {
    log(`[fetch-engine] downloading ${manifest.url}`)
    bytes = await download(manifest, buildDir, fetchImpl)
    source = 'download'
  }
  const files = unpackEngineArchive(bytes, manifest)
  // Keep the verified archive even when supplied explicitly, so packaging can
  // re-establish file provenance offline without trusting staged SHA256SUMS.
  const cache = path.join(buildDir, '.engine-cache')
  await mkdir(cache, { recursive: true })
  await writeFile(path.join(cache, manifest.archive), bytes)
  await writeEngineTree(files, manifest, buildDir)
  log(`[fetch-engine] staged TensorFold engine ${manifest.version} (${files.size} files, sha256 ${manifest.sha256.slice(0, 12)}) from the ${source}.`)
  return { manifest, source }
}

const invoked = process.argv[1] ? pathToFileURL(path.resolve(process.argv[1])).href : ''
if (invoked === import.meta.url) {
  const root = fileURLToPath(new URL('..', import.meta.url))
  fetchEngine({ root, target: engineTarget(), force: process.argv.includes('--force') }).catch((error) => {
    console.error(`[fetch-engine] ${error?.message || error}`)
    process.exitCode = 1
  })
}
