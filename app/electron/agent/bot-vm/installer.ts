import { promises as fs, createReadStream } from 'node:fs'
import type { FileHandle } from 'node:fs/promises'
import path from 'node:path'
import os from 'node:os'
import { createHash, randomUUID } from 'node:crypto'
import { createGunzip } from 'node:zlib'
import type { VmRunner } from './service'
import type { BotVmSetupStatus } from '../types/bot-vm'

export const APPLE_VM_PACKAGE = {
  url: 'https://github.com/apple/container/releases/download/1.5.0/container-1.5.0-installer-signed.pkg',
  sha256: 'a24808cb202318fa1c3bbee0c6c6887fe1225fe899d7b687a0ddd939bd6573f8',
  bytes: 118045087,
  team: 'UPBK2H6LZM',
} as const
export const APPLE_VM_FILES: Record<string, { sha256: string; bytes: number; signed?: boolean; executable?: boolean }> = {
  'bin/container': { sha256: '8cc619377e8978a0ce9f22a3e864dcd4445f2bc38fb64c5197bb69a1bab42463', bytes: 68397488, signed: true },
  'bin/container-apiserver': { sha256: '0d05388259e73b63f84680968fa5fa39feeef4219cce45c5655d29d3172a63b9', bytes: 63087088, signed: true },
  'libexec/container/plugins/container-runtime-linux/bin/container-runtime-linux': { sha256: 'e8bee184cc1f6b53982f9cc4ecb02085ebe238b8834112f71f8972c0d488511a', bytes: 62769360, signed: true },
  'libexec/container/plugins/container-runtime-linux/config.toml': { sha256: 'd609af652f3e0224cb7f0cef315f873081506d010d2d1a8ff33508980e3427a7', bytes: 198 },
  'libexec/container/plugins/machine-apiserver/bin/machine-apiserver': { sha256: '1d3f53f6979d92d0bb9f343cc16782bad9a4b23bb380619f4b0d87d9d6df449c', bytes: 62662720, signed: true },
  'libexec/container/plugins/machine-apiserver/config.toml': { sha256: '819edb0d3c20517e8a56e11a9623b3804c6821d3920da3fa66d989f766103b6a', bytes: 281 },
  'libexec/container/plugins/machine-apiserver/resources/init': { sha256: '77a7f83faca9f8656ef129d8f91ddc4e770c80478d07b805a2530b9a902bf15a', bytes: 2736, executable: true },
  'libexec/container/plugins/machine-apiserver/resources/create-user.sh': { sha256: 'bdbb3ceef02861b2b270ac403aa32320db6f23a0042dfde24c6a1275eca82f7b', bytes: 1794, executable: true },
  'libexec/container/plugins/container-core-images/bin/container-core-images': { sha256: 'dc88d9aeb67f8258c3697f8485b6ee77018d63b86b6dc3ba9905fb88f01cb6eb', bytes: 62319904, signed: true },
  'libexec/container/plugins/container-core-images/config.toml': { sha256: '89ebf5415177298d36f4c67c8c03db26fac1377b428f30a5ccf96407d8f63f9d', bytes: 266 },
  'libexec/container/plugins/container-network-vmnet/bin/container-network-vmnet': { sha256: 'bfb0c8b002946071df4428c5fa8e1d3bcb86d9798ae2e25dab1a1388e84c7270', bytes: 60672800, signed: true },
  'libexec/container/plugins/container-network-vmnet/config.toml': { sha256: '7ec0d522dcf9c9bc78b1e0843916bd0a98cfec45ef5b35f04fb8407ecda3db3e', bytes: 198 },
}
const OMITTED_FILES = new Set(['bin/uninstall-container.sh', 'bin/update-container.sh', 'libexec/container/plugins/k8s/bin/k8s', 'libexec/container/plugins/k8s/resources/kindnet.yaml', 'libexec/container/plugins/k8s/config.toml'])
export interface ManagedVmRuntime { installRoot: string; appRoot: string; binary: string }
const MANAGED_SERVICES = [
  ['com.apple.container.apiserver', 'bin/container-apiserver', 'apiserver/apiserver.plist'],
  ['com.apple.container.container-network-vmnet.default', 'libexec/container/plugins/container-network-vmnet/bin/container-network-vmnet', 'networks/default/service.plist'],
  ['com.apple.container.machine-apiserver', 'libexec/container/plugins/machine-apiserver/bin/machine-apiserver', 'plugin-state/machine-apiserver/service.plist'],
  ['com.apple.container.container-core-images', 'libexec/container/plugins/container-core-images/bin/container-core-images', 'plugin-state/container-core-images/service.plist'],
] as const
async function managedServices(runtime: ManagedVmRuntime, runner: VmRunner) {
  const services: Array<{ label: string; plist: string }> = []
  for (const [label, relative, plist] of MANAGED_SERVICES) {
    const result = await runner('/bin/launchctl', ['print', `gui/${process.getuid!()}/${label}`], { timeoutMs: 10000, env: { PATH: '/usr/bin:/bin', LANG: 'en_US.UTF-8' } })
    if (result.code !== 0) continue
    const program = result.stdout.match(/^\s*program = (.+)$/m)?.[1], appRoot = result.stdout.match(/^\s*CONTAINER_APP_ROOT => (.+)$/m)?.[1]
    const canonical = (value: string) => path.resolve(value).replace(/^\/private\/tmp\//, '/tmp/')
    if (!program || !appRoot || canonical(program) !== canonical(path.join(runtime.installRoot, relative)) || canonical(appRoot) !== canonical(runtime.appRoot)) throw new Error('Another Apple container runtime is already registered; TensorFold will preserve it, so use Select runtime for that installation or stop it in its own app before managed setup.')
    services.push({ label, plist: path.join(runtime.appRoot, plist) })
  }
  return services
}
export async function assertManagedRuntimeAvailable(runtime: ManagedVmRuntime, runner: VmRunner): Promise<void> { await managedServices(runtime, runner) }
export async function stopManagedRuntimeServices(runtime: ManagedVmRuntime, runner: VmRunner): Promise<void> {
  for (let round = 0; round < 3; round++) {
    const services = await managedServices(runtime, runner)
    if (!services.length) return
    for (const service of services) {
      // Exact validated service names, never Apple's global stop/prune operation.
      const stopped = await runner('/bin/launchctl', ['bootout', `gui/${process.getuid!()}/${service.label}`], { timeoutMs: 15000, env: { PATH: '/usr/bin:/bin', LANG: 'en_US.UTF-8' } })
      if (stopped.code !== 0) throw new Error('A TensorFold-owned VM runtime service could not stop; its private files have been preserved.')
      const deadline = Date.now() + 5000
      while (true) {
        const check = await runner('/bin/launchctl', ['print', `gui/${process.getuid!()}/${service.label}`], { timeoutMs: 10000, env: { PATH: '/usr/bin:/bin', LANG: 'en_US.UTF-8' } })
        if (check.code !== 0) break
        if (Date.now() >= deadline) throw new Error('A VM runtime service remains registered; private runtime files have been preserved.')
        await new Promise((resolve) => setTimeout(resolve, 50))
      }
    }
    // A cancelled initial boot can have a helper registration already in flight.
    await new Promise((resolve) => setTimeout(resolve, 50))
  }
  if ((await managedServices(runtime, runner)).length) throw new Error('VM runtime cleanup did not settle; its private files have been preserved.')
}
function cancelled(signal?: AbortSignal) { if (signal?.aborted) throw new Error('VM setup cancelled.') }
async function ownedDir(dir: string) { await fs.mkdir(dir, { recursive: true, mode: 0o700 }); const stat = await fs.lstat(dir); if (!stat.isDirectory() || stat.isSymbolicLink()) throw new Error('Refusing a linked VM installation directory.'); await fs.chmod(dir, 0o700) }
async function hashFile(file: string): Promise<string> { const hash = createHash('sha256'); for await (const chunk of createReadStream(file)) hash.update(chunk); return hash.digest('hex') }
async function checkedParent(root: string, relative: string, create = false) {
  let dir = root
  for (const part of relative.split('/').slice(0, -1)) { dir = path.join(dir, part); if (create) { try { await fs.mkdir(dir, { mode: 0o700 }) } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error } }; const info = await fs.lstat(dir); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('Linked runtime component directories are forbidden.') }
}
function cleanArchivePath(value: string) {
  if (value === '.') return ''
  const name = value.startsWith('./') ? value.slice(2) : value
  if (!name || name.length > 4096 || name.startsWith('/') || name.includes('\\') || /[\x00-\x1f\x7f]/.test(name) || name.split('/').some((p) => !p || p === '.' || p === '..')) throw new Error('Unsafe path in the signed runtime archive.')
  return name
}

/** Decode only the package's bounded odc CPIO payload. No tar/cpio extractor,
 * symlink, device, hard link, script or install hook is executed. */
export async function extractVmPayload(payload: string, destination: string, signal?: AbortSignal, manifest = APPLE_VM_FILES) {
  await ownedDir(destination)
  const source = createReadStream(payload, { highWaterMark: 65536 }), gunzip = createGunzip({ chunkSize: 65536 })
  const abort = () => { source.destroy(new Error('VM setup cancelled.')); gunzip.destroy(new Error('VM setup cancelled.')) }
  source.on('error', (error) => gunzip.destroy(error))
  signal?.addEventListener('abort', abort, { once: true }); source.pipe(gunzip)
  const iterator = gunzip[Symbol.asyncIterator](); let buffer = Buffer.alloc(0), expanded = 0, entries = 0
  const refill = async () => {
    cancelled(signal)
    const next = await iterator.next()
    if (next.done) return false
    const chunk = Buffer.from(next.value); expanded += chunk.length
    if (expanded > 512 * 1024 * 1024) throw new Error('The runtime payload exceeds its extraction limit.')
    buffer = buffer.length ? Buffer.concat([buffer, chunk]) : chunk
    return true
  }
  const take = async (size: number) => { while (buffer.length < size) if (!await refill()) throw new Error('The runtime archive is truncated.'); const value = buffer.subarray(0, size); buffer = buffer.subarray(size); return value }
  const seen = new Set<string>()
  try {
    while (true) {
      const header = await take(76)
      if (header.subarray(0, 6).toString() !== '070707') throw new Error('Unsupported runtime payload format.')
      const octal = (a: number, b: number) => { const value = header.subarray(a, b).toString(); if (!/^[0-7]+$/.test(value)) throw new Error('Invalid runtime archive metadata.'); return parseInt(value, 8) }
      const mode = octal(18, 24), links = octal(36, 42), nameSize = octal(59, 65), size = octal(65, 76)
      if (nameSize < 2 || nameSize > 4097 || size > 100 * 1024 * 1024 || ++entries > 200) throw new Error('The runtime archive exceeds its member limits.')
      const nameBytes = await take(nameSize)
      if (nameBytes.at(-1) !== 0) throw new Error('Invalid runtime archive name.')
      const rawName = new TextDecoder('utf-8', { fatal: true }).decode(nameBytes.subarray(0, -1))
      if (rawName === 'TRAILER!!!') {
        if (size !== 0) throw new Error('Invalid runtime archive trailer.')
        while (await refill()) if (buffer.length > 4096) throw new Error('Unexpected runtime archive trailing data.')
        if (buffer.some((byte) => byte !== 0)) throw new Error('Unexpected runtime archive trailing data.')
        break
      }
      const name = cleanArchivePath(rawName), type = mode & 0o170000
      if (seen.has(name)) throw new Error('Duplicate runtime archive member.'); seen.add(name)
      if (type === 0o040000) { if (size !== 0) throw new Error('Invalid archive directory.'); continue }
      if (type !== 0o100000 || links !== 1 || (mode & 0o7000)) throw new Error('Links, special files and privileged files are forbidden in runtime archives.')
      const expected = Object.prototype.hasOwnProperty.call(manifest, name) ? manifest[name] : undefined
      if (!expected && !OMITTED_FILES.has(name)) throw new Error('Unexpected file in the pinned runtime package.')
      if (expected && expected.bytes !== size) throw new Error('A runtime component has an unexpected size.')
      let file: FileHandle | undefined
      const hash = createHash('sha256')
      if (expected) { const target = path.join(destination, name); await checkedParent(destination, name, true); file = await fs.open(target, 'wx', 0o600) }
      try {
        let remaining = size
        while (remaining) { if (!buffer.length && !await refill()) throw new Error('The runtime archive is truncated.'); const chunk = buffer.subarray(0, Math.min(65536, remaining, buffer.length)); buffer = buffer.subarray(chunk.length); remaining -= chunk.length; hash.update(chunk); if (file) await file.writeFile(chunk) }
        if (expected && hash.digest('hex') !== expected.sha256) throw new Error('A runtime component failed its pinned checksum.')
      } finally { await file?.close() }
      if (expected) await fs.chmod(path.join(destination, name), expected.signed || expected.executable ? 0o755 : 0o644)
    }
    if (Object.keys(manifest).some((name) => !seen.has(name))) throw new Error('A required runtime component is missing.')
  } finally { signal?.removeEventListener('abort', abort); source.destroy(); gunzip.destroy() }
}

export class BotVmInstaller {
  readonly root: string
  private current: BotVmSetupStatus = { active: false, phase: 'idle', bytes: 0, total: APPLE_VM_PACKAGE.bytes, detail: 'VM setup downloads Apple’s signed 118 MB runtime package and the pinned 20 MB guest image; it requires at least 6 GiB free before installation.' }
  private readonly release: { url: string; sha256: string; bytes: number; team: string }
  private readonly manifest: typeof APPLE_VM_FILES
  private readonly freeDiskBytes?: () => Promise<number>
  /** Dependency injection is native-only QA plumbing; IPC never accepts pins. */
  constructor(appHome: string, private readonly runner: VmRunner, private readonly fetcher: typeof fetch = fetch, fixture?: { release?: { url: string; sha256: string; bytes: number; team: string }; files?: typeof APPLE_VM_FILES; freeDiskBytes?: () => Promise<number> }) { this.root = path.join(appHome, 'bot-vms', 'managed-runtime'); this.release = fixture?.release ?? APPLE_VM_PACKAGE; this.manifest = fixture?.files ?? APPLE_VM_FILES; this.freeDiskBytes = fixture?.freeDiskBytes }
  snapshot(): BotVmSetupStatus { return { ...this.current } }
  progress(phase: BotVmSetupStatus['phase'], detail: string) { this.current = { ...this.current, phase, detail } }
  paths(): ManagedVmRuntime { const installRoot = path.join(this.root, 'apple-1.5.0'); return { installRoot, appRoot: path.join(this.root, 'data'), binary: path.join(installRoot, 'bin', 'container') } }
  private env(): NodeJS.ProcessEnv { return { HOME: os.homedir(), PATH: '/usr/bin:/bin:/usr/sbin:/sbin', LANG: 'en_US.UTF-8' } }
  private async tool(binary: string, args: string[], signal?: AbortSignal, timeoutMs = 30000) { return this.runner(binary, args, { signal, timeoutMs, env: this.env() }) }
  async verifyInstalled(runtime = this.paths(), signal?: AbortSignal): Promise<void> {
    const root = await fs.lstat(runtime.installRoot)
    if (!root.isDirectory() || root.isSymbolicLink()) throw new Error('The managed runtime directory is invalid.')
    const pending = ['']; let count = 0
    while (pending.length) {
      const relative = pending.pop()!, directory = await fs.opendir(path.join(runtime.installRoot, relative))
      for await (const entry of directory) {
        if (++count > 100) throw new Error('Unexpected files in the private runtime installation.')
        const name = relative ? `${relative}/${entry.name}` : entry.name, info = await fs.lstat(path.join(runtime.installRoot, name))
        if (info.isSymbolicLink() || (info.mode & 0o022)) throw new Error('Linked or writable runtime components are forbidden.')
        if (info.isDirectory()) { if (!Object.keys(this.manifest).some((value) => value.startsWith(`${name}/`))) throw new Error('Unexpected plugin directory in the managed runtime.'); pending.push(name) }
        else if (!info.isFile() || !Object.prototype.hasOwnProperty.call(this.manifest, name)) throw new Error('Unexpected file in the managed runtime.')
      }
    }
    for (const [relative, expected] of Object.entries(this.manifest)) {
      cancelled(signal); await checkedParent(runtime.installRoot, relative); const file = path.join(runtime.installRoot, relative), info = await fs.lstat(file)
      if (!info.isFile() || info.isSymbolicLink() || info.size !== expected.bytes || (info.mode & 0o022) || await hashFile(file) !== expected.sha256) throw new Error('A managed runtime component changed; preserve its data and retry setup.')
      if (expected.signed) { const result = await this.tool('/usr/bin/codesign', ['--verify', '--strict', '-R', `=anchor apple generic and certificate leaf[subject.OU] = "${APPLE_VM_PACKAGE.team}"`, file], signal); if (result.code !== 0) throw new Error('An Apple runtime helper failed signature verification.') }
    }
  }
  private async download(target: string, signal?: AbortSignal) {
    this.current = { active: true, phase: 'downloading', bytes: 0, total: this.release.bytes, detail: 'Downloading Apple container 1.5.0 from its pinned official release.' }
    let url = this.release.url, response: Response | null = null
    for (let redirects = 0; redirects < 4; redirects++) {
      cancelled(signal); const parsed = new URL(url)
      if (parsed.protocol !== 'https:' || parsed.username || parsed.password || parsed.port || !['github.com', 'release-assets.githubusercontent.com', 'objects.githubusercontent.com'].includes(parsed.hostname)) throw new Error('The runtime download redirected outside its approved release hosts.')
      response = await this.fetcher(url, { redirect: 'manual', signal, headers: { 'User-Agent': 'TensorFold-VM-Setup' } })
      if (![301, 302, 303, 307, 308].includes(response.status)) break
      const location = response.headers.get('location'); await response.body?.cancel(); if (!location) throw new Error('The runtime release redirect is invalid.'); url = new URL(location, url).href; response = null
    }
    if (!response?.ok || !response.body) { await response?.body?.cancel(); throw new Error('The official runtime download failed; retry setup.') }
    const length = response.headers.get('content-length')
    if (length && Number(length) !== this.release.bytes) { await response.body.cancel(); throw new Error('The runtime package has an unexpected download size.') }
    const file = await fs.open(target, 'wx', 0o600), reader = response.body.getReader(), hash = createHash('sha256')
    let idle: ReturnType<typeof setTimeout> | undefined
    try {
      while (true) {
        cancelled(signal); idle = setTimeout(() => void reader.cancel('Runtime download stalled.'), 30000)
        const chunk = await reader.read(); clearTimeout(idle)
        if (chunk.done) break
        this.current.bytes += chunk.value.byteLength
        if (this.current.bytes > this.release.bytes) throw new Error('The runtime download exceeded its package size limit.')
        hash.update(chunk.value); await file.writeFile(chunk.value)
      }
      cancelled(signal)
      if (this.current.bytes !== this.release.bytes || hash.digest('hex') !== this.release.sha256) throw new Error('The runtime package failed its pinned release checksum.')
      await file.sync()
    } finally { if (idle) clearTimeout(idle); await reader.cancel().catch(() => {}); await file.close() }
  }
  async install(signal?: AbortSignal): Promise<ManagedVmRuntime> {
    signal = signal ? AbortSignal.any([signal, AbortSignal.timeout(300000)]) : AbortSignal.timeout(300000)
    await ownedDir(this.root); const runtime = this.paths()
    try { await this.verifyInstalled(runtime, signal); this.progress('activating', 'Using the verified private Apple runtime.'); return runtime }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
    const free = this.freeDiskBytes ? await this.freeDiskBytes() : await fs.statfs(this.root).then((stat) => stat.bavail * stat.bsize)
    if (free < 6 * 1024 ** 3) throw new Error('Free at least 6 GiB before installing the local VM runtime.')
    const temporary = path.join(this.root, `.setup-${randomUUID()}`); await fs.mkdir(temporary, { mode: 0o700 })
    const packageFile = path.join(temporary, 'runtime.pkg'), flat = path.join(temporary, 'package'), staged = path.join(temporary, 'runtime')
    let activated = false
    try {
      await this.download(packageFile, signal)
      this.progress('verifying', 'Checking the pinned package checksum, Apple installer identity and notarization.')
      const signature = await this.tool('/usr/sbin/pkgutil', ['--check-signature', packageFile], signal)
      if (signature.code !== 0 || !signature.stdout.includes(`Developer ID Installer: Apple Inc. - Containerization (${APPLE_VM_PACKAGE.team})`) || !/Status:\s+signed by a developer certificate issued by Apple/.test(signature.stdout) || !/Notarization:\s+trusted by the Apple notary service/.test(signature.stdout)) throw new Error('The runtime package is not signed and notarized by the expected Apple publisher.')
      const members = await this.tool('/usr/bin/xar', ['-tf', packageFile], signal)
      const names = members.stdout.trim().split('\n').sort()
      if (members.code !== 0 || JSON.stringify(names) !== JSON.stringify(['Bom', 'PackageInfo', 'Payload'])) throw new Error('Unexpected installer package structure; no installer was executed.')
      this.progress('extracting', 'Extracting verified components privately; installer scripts are never executed.')
      const expanded = await this.tool('/usr/sbin/pkgutil', ['--expand', packageFile, flat], signal, 60000)
      if (expanded.code !== 0) throw new Error('The signed runtime package could not be opened.')
      for (const name of names) { const info = await fs.lstat(path.join(flat, name)); const cap = name === 'PackageInfo' ? 16384 : name === 'Bom' ? 1024 * 1024 : APPLE_VM_PACKAGE.bytes; if (!info.isFile() || info.isSymbolicLink() || info.size > cap) throw new Error('A package member is linked, invalid or too large.') }
      const info = await fs.readFile(path.join(flat, 'PackageInfo'), 'utf8')
      if (info.length > 16384 || !info.includes('identifier="com.apple.container-installer"') || !info.includes('version="1.5.0"') || /<scripts\b|<preinstall\b|<postinstall\b/.test(info)) throw new Error('Unexpected Apple installer metadata or installation hooks.')
      await extractVmPayload(path.join(flat, 'Payload'), staged, signal, this.manifest)
      await this.verifyInstalled({ ...runtime, installRoot: staged, binary: path.join(staged, 'bin', 'container') }, signal)
      this.progress('activating', 'Activating the verified runtime without changing system packages.')
      cancelled(signal); await fs.rename(staged, runtime.installRoot)
      activated = true
      await ownedDir(runtime.appRoot)
      await this.verifyInstalled(runtime, signal)
      return runtime
    } catch (error) { if (activated) await fs.rm(runtime.installRoot, { recursive: true, force: true }); this.progress(signal?.aborted ? 'cancelled' : 'error', signal?.aborted ? 'VM setup cancelled; retry whenever you are ready.' : (error as Error).message); throw error }
    finally { await fs.rm(temporary, { recursive: true, force: true }); this.current.active = false }
  }
}
