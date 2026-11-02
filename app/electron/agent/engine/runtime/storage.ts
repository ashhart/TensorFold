import { promises as fs, createReadStream, type Stats } from 'node:fs'
import path from 'node:path'
import { createHash, randomUUID } from 'node:crypto'
import { createGunzip } from 'node:zlib'
import { Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { Parser, Unpack, type ReadEntry } from 'tar'
import { INVENTORY_LIMIT, safeRuntimePath, type RuntimePackage } from './catalog'

export async function privateDirectory(dir: string): Promise<void> {
  await fs.mkdir(dir, { recursive: true, mode: 0o700 })
  const stat = await fs.lstat(dir)
  if (!stat.isDirectory() || stat.isSymbolicLink() || path.resolve(await fs.realpath(dir)) !== path.resolve(dir)) throw new Error('Runtime storage must be a private directory without links.')
  await fs.chmod(dir, 0o700)
}
export async function atomicJson(file: string, value: unknown): Promise<void> {
  const temp = `${file}.${randomUUID()}.tmp`
  try {
    const handle = await fs.open(temp, 'wx', 0o600)
    try { await handle.writeFile(JSON.stringify(value)); await handle.sync() } finally { await handle.close() }
    await fs.rename(temp, file)
  } finally { await fs.rm(temp, { force: true }) }
}
export async function readPrivateJson(file: string, max = 512 * 1024): Promise<string> {
  const st = await fs.lstat(file)
  if (!st.isFile() || st.isSymbolicLink() || st.nlink !== 1 || st.size > max) throw new Error('Invalid runtime metadata.')
  return fs.readFile(file, 'utf8')
}
async function digest(file: string): Promise<string> {
  const hash = createHash('sha256')
  for await (const chunk of createReadStream(file)) hash.update(chunk)
  return hash.digest('hex')
}
export async function verifyRuntimeTree(dir: string, pkg: RuntimePackage): Promise<void> {
  await privateDirectory(dir)
  const files = new Map(pkg.files.map(f => [f.path, f]))
  const seen = new Set<string>()
  async function walk(parent: string, prefix = ''): Promise<void> {
    for (const name of await fs.readdir(parent)) {
      const rel = prefix + name
      if (rel === '.catalog.json') { await readPrivateJson(path.join(dir, rel)); continue }
      if (!safeRuntimePath(rel)) throw new Error('Unexpected runtime file.')
      const file = path.join(parent, name), stat = await fs.lstat(file)
      if (stat.isSymbolicLink()) throw new Error('Runtime links are forbidden.')
      if (stat.isDirectory() && pkg.files.some(f => f.path.startsWith(`${rel}/`))) await walk(file, `${rel}/`)
      else {
        const expected = files.get(rel)
        if (!expected || !stat.isFile() || stat.nlink !== 1 || stat.size !== expected.size ||
          await digest(file) !== expected.sha256) throw new Error('Runtime file integrity check failed.')
        if (process.platform !== 'win32' && (stat.mode & 0o111) !== (expected.executable ? 0o100 : 0)) throw new Error('Runtime executable permissions changed.')
        seen.add(rel)
      }
    }
  }
  await walk(dir)
  if (seen.size !== files.size) throw new Error('Runtime package is missing required files.')
}
export async function extractRuntime(archive: string, dir: string, pkg: RuntimePackage, signal?: AbortSignal): Promise<void> {
  await privateDirectory(dir)
  const files = new Map(pkg.files.map(f => [f.path, f]))
  const directories = new Set(pkg.files.flatMap(f => f.path.split('/').slice(0, -1).map((_, i, p) => p.slice(0, i + 1).join('/'))))
  const seen = new Set<string>()
  let invalid: Error | null = null
  let entries = 0
  const filter = (raw: string, entry: Stats | ReadEntry) => {
    if (!('type' in entry)) return false
    const name = raw.replace(/\/$/, '')
    const file = files.get(name)
    const valid = !invalid && ++entries <= 1024 && safeRuntimePath(name) && !seen.has(name.toLowerCase()) &&
      !entry.linkpath && (entry.type === 'Directory' ? directories.has(name) && entry.size === 0 :
        entry.type === 'File' && !!file && file.size === entry.size)
    if (!valid) { invalid = new Error('Runtime archive contains unexpected paths, links, duplicates or file sizes.'); return false }
    seen.add(name.toLowerCase())
    return true
  }
  const stream = async (destination: Parser | Unpack) => {
    let bytes = 0
    const limit = new Transform({ transform(chunk, _encoding, done) {
      bytes += chunk.length
      done(invalid ?? (bytes > INVENTORY_LIMIT + 4 * 1024 * 1024 ? new Error('Runtime archive expansion limit exceeded.') : null), chunk)
    } })
    await pipeline(createReadStream(archive), createGunzip(), limit, destination, { signal })
    if (invalid) throw invalid
  }
  await stream(new Parser({ strict: true, filter, onReadEntry: e => e.resume() }))
  if (pkg.files.some(f => !seen.has(f.path.toLowerCase()))) throw new Error('Runtime archive is incomplete.')
  // Validate the complete archive before allowing the extractor to write any member.
  seen.clear(); entries = 0
  await stream(new Unpack({ cwd: dir, strict: true, filter, preservePaths: false, preserveOwner: false,
    noMtime: true, dmode: 0o700, fmode: 0o600, maxDepth: 32 }))
  for (const f of pkg.files) await fs.chmod(path.join(dir, f.path), f.executable ? 0o700 : 0o600)
  await verifyRuntimeTree(dir, pkg)
}
