import { createHash, randomBytes, randomUUID } from 'node:crypto'
import { spawn } from 'node:child_process'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { BOT_VM_GUEST } from './guest'
import { BotVmInstaller, assertManagedRuntimeAvailable, stopManagedRuntimeServices, type ManagedVmRuntime } from './installer'
import type { BotVmStatus, BotVmResult, BotVmSetupStatus } from '../types/bot-vm'

export const BOT_VM_IMAGE = 'docker.io/library/python@sha256:e041b488453556787172f43845c9e0e90b9fb4a91452498420fb2011b23241b6'
export const BOT_VM_SETUP_URL = 'https://github.com/apple/container/releases/tag/1.5.0'
export const VM_TOOL_NAMES = ['read_file', 'write_file', 'edit_file', 'list_dir', 'search', 'glob', 'run_command'] as const
const OWNER_LABEL = 'dev.tensorfold.owner'
const BOT_LABEL = 'dev.tensorfold.bot'
const SCHEMA_LABEL = 'dev.tensorfold.vm-schema'
const VOLUME_CAPACITY = 8 * 1024 ** 3
const INIT = 'import time,signal; signal.signal(signal.SIGTERM,lambda *a:exit(0)); time.sleep(2147483647)'
const PREPARE_VOLUME = 'import os; os.chown("/workspace",0,0); os.chmod("/workspace",0o700); os.chown("/workspace",1000,1000)'
type RunResult = { code: number; stdout: string; stderr: string }
export type VmRunner = (binary: string, args: string[], options: { signal?: AbortSignal; timeoutMs?: number; stdin?: string; env?: NodeJS.ProcessEnv }) => Promise<RunResult>
interface VmRecord { version: 1; id: string; volume: string; botKey: string; image: string }
interface Runtime { binary: string; version: string; sha256: string }
interface RuntimePin { version: 1; binary: string; sha256: string; managed?: boolean }
export interface BotVmOptions {
  appHome: string
  supported?: boolean
  platform?: NodeJS.Platform
  architecture?: string
  osVersion?: string
  /** Native-only runtime selection, never accepted from a model or HTTP client. */
  runtimePath?: string
  /** Private QA/development runtimes can use a separate Apple application data store. */
  runtimeAppRoot?: string
  runtimeInstallRoot?: string
  runner?: VmRunner
  verifyRuntime?: (binary: string) => Promise<void>
  reserveOperation?: () => { release(): void }
  /** Deterministic QA seam, never exposed across IPC. */
  freeDiskBytes?: () => Promise<number>
  installerFactory?: (appHome: string, runner: VmRunner) => BotVmInstaller
}

export const runVmProcess: VmRunner = async (binary, args, options) => new Promise((resolve, reject) => {
  if (options.signal?.aborted) { reject(new Error('VM operation cancelled.')); return }
  const child = spawn(binary, args, { env: options.env, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true })
  const stdout: Buffer[] = [], stderr: Buffer[] = []
  let bytes = 0, failure: Error | null = null, finished = false
  const stop = (reason: string) => { if (failure) return; failure = new Error(reason); child.kill('SIGTERM'); escalation = setTimeout(() => child.kill('SIGKILL'), 500); escalation.unref() }
  let escalation: ReturnType<typeof setTimeout> | undefined
  const timer = setTimeout(() => stop('VM operation exceeded its time limit.'), options.timeoutMs ?? 15_000)
  const abort = () => stop('VM operation cancelled.')
  options.signal?.addEventListener('abort', abort, { once: true })
  const cleanup = () => { clearTimeout(timer); if (escalation) clearTimeout(escalation); options.signal?.removeEventListener('abort', abort) }
  const output = (list: Buffer[], chunk: Buffer) => { bytes += chunk.length; if (bytes > 2 * 1024 * 1024) { stop('VM output exceeded its limit.'); return }; list.push(chunk) }
  child.stdout.on('data', (chunk: Buffer) => output(stdout, chunk))
  child.stderr.on('data', (chunk: Buffer) => output(stderr, chunk))
  child.stdin.on('error', () => {})
  child.on('error', () => { if (finished) return; finished = true; cleanup(); reject(new Error('The Apple container executable could not be started.')) })
  child.on('close', (code) => { if (finished) return; finished = true; cleanup(); if (failure) reject(failure); else resolve({ code: code ?? -1, stdout: Buffer.concat(stdout).toString('utf8'), stderr: Buffer.concat(stderr).toString('utf8') }) })
  child.stdin.end(options.stdin ?? '')
})

function botKey(botId: string): string {
  if (typeof botId !== 'string' || !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$/.test(botId)) throw new Error('Choose a saved bot before managing its VM.')
  return createHash('sha256').update(botId).digest('hex').slice(0, 24)
}
function parseArray(output: string): any[] { const value: unknown = JSON.parse(output); if (!Array.isArray(value) || value.length > 5000) throw new Error('Unexpected Apple container response.'); return value }
async function regularJson(file: string): Promise<any | null> {
  try { const info = await fs.lstat(file); if (!info.isFile() || info.isSymbolicLink() || info.size > 16384) throw new Error('Invalid VM settings file.'); return JSON.parse(await fs.readFile(file, 'utf8')) }
  catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null; throw new Error('VM settings are unreadable; preserve them before repairing this profile.') }
}
async function secureDirectory(dir: string) { await fs.mkdir(dir, { recursive: true, mode: 0o700 }); const info = await fs.lstat(dir); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('Refusing a linked VM data directory.'); await fs.chmod(dir, 0o700) }
async function atomicJson(file: string, value: unknown) {
  const temp = path.join(path.dirname(file), `.vm-${randomUUID()}.tmp`)
  try { await fs.writeFile(temp, JSON.stringify(value), { flag: 'wx', mode: 0o600 }); await fs.rename(temp, file) } finally { await fs.rm(temp, { force: true }) }
}

export class BotVmService {
  private readonly root: string
  private readonly run: VmRunner
  private profile: string | null = null
  private loading: Promise<void> | null = null
  private queues = new Map<string, Promise<unknown>>()
  private active = new Map<string, AbortController>()
  private closing = false
  private quiescing: Promise<{ stopped: number }> | null = null
  private installer: BotVmInstaller
  private managedRuntime: ManagedVmRuntime | null = null
  private managedVerified = false
  private setupBotId: string | null = null
  private lastSetupBotId: string | null = null
  private setupAbort: AbortController | null = null
  constructor(private readonly options: BotVmOptions) { this.root = path.join(options.appHome, 'bot-vms'); this.run = options.runner ?? runVmProcess; this.installer = options.installerFactory?.(options.appHome, this.run) ?? new BotVmInstaller(options.appHome, this.run) }
  private supported() { return this.options.supported !== false && (this.options.platform ?? process.platform) === 'darwin' && (this.options.architecture ?? process.arch) === 'arm64' && Number((this.options.osVersion ?? os.release()).split('.')[0]) >= (this.options.osVersion ? 26 : 25) }
  private env(): NodeJS.ProcessEnv {
    // Registry credentials, model API keys, proxies and SSH agents are never inherited.
    const appRoot = this.options.runtimeAppRoot ?? this.managedRuntime?.appRoot, installRoot = this.options.runtimeInstallRoot ?? this.managedRuntime?.installRoot
    return { HOME: os.homedir(), PATH: '/usr/bin:/bin:/usr/sbin:/sbin', LANG: 'en_US.UTF-8', ...(appRoot ? { CONTAINER_APP_ROOT: appRoot } : {}), ...(installRoot ? { CONTAINER_INSTALL_ROOT: installRoot } : {}) }
  }
  private async initialize() {
    if (this.profile) return
    if (!this.loading) this.loading = (async () => {
      await secureDirectory(this.root)
      const file = path.join(this.root, 'owner.json')
      let value = await regularJson(file)
      if (!value) {
        value = { version: 1, profile: randomBytes(16).toString('hex') }
        try { await fs.writeFile(file, JSON.stringify(value), { flag: 'wx', mode: 0o600 }) }
        catch (error) { if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error; value = await regularJson(file) }
      }
      if (value?.version !== 1 || !/^[0-9a-f]{32}$/.test(value.profile)) throw new Error('The VM ownership record is invalid; existing VMs have been preserved.')
      this.profile = value.profile
    })().finally(() => { this.loading = null })
    await this.loading
  }
  private ids(botId: string): VmRecord { const key = botKey(botId); return { version: 1, id: `tfbot-${this.profile!.slice(0, 16)}-${key}`, volume: `tfbot-${this.profile!.slice(0, 16)}-${key}-data`, botKey: key, image: BOT_VM_IMAGE } }
  private async record(botId: string): Promise<VmRecord | null> {
    await this.initialize(); const expected = this.ids(botId), value = await regularJson(path.join(this.root, `${expected.botKey}.json`))
    if (!value) return null
    if (Object.keys(expected).some((key) => value[key] !== expected[key as keyof VmRecord])) throw new Error('The VM ownership record does not match this bot; existing data has been preserved.')
    return expected
  }
  private async runtime(): Promise<Runtime | null> {
    if (!this.supported()) return null
    await this.initialize()
    const pin = await regularJson(path.join(this.root, 'runtime.json')) as RuntimePin | null
    if (pin && (pin.version !== 1 || typeof pin.binary !== 'string' || !/^[0-9a-f]{64}$/.test(pin.sha256))) throw new Error('The saved Apple runtime selection is invalid.')
    if (pin?.managed === true) {
      const managed = this.installer.paths()
      if (pin.binary !== managed.binary) throw new Error('The managed runtime selection does not match this profile.')
      if (!this.managedVerified) { await this.installer.verifyInstalled(managed); this.managedVerified = true }
      this.managedRuntime = managed
    } else this.managedRuntime = null
    const binary = this.options.runtimePath ?? pin?.binary ?? '/usr/local/bin/container'
    let info
    try { info = await fs.lstat(binary) } catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null; throw error }
    if (!path.isAbsolute(binary) || path.basename(binary) !== 'container' || !info.isFile() || info.isSymbolicLink() || info.size > 200 * 1024 * 1024 || (info.mode & 0o022) !== 0) throw new Error('Choose the unmodified Apple container executable from its signed installation.')
    const sha256 = createHash('sha256').update(await fs.readFile(binary)).digest('hex')
    if (pin && binary === pin.binary && sha256 !== pin.sha256) throw new Error('The Apple runtime changed; select its signed executable again before using bot VMs.')
    if (this.options.verifyRuntime) await this.options.verifyRuntime(binary)
    else {
      const result = await this.run('/usr/bin/codesign', ['--verify', '--strict', '-R', '=anchor apple generic and certificate leaf[subject.OU] = "UPBK2H6LZM"', binary], { timeoutMs: 15000, env: this.env() })
      if (result.code !== 0) throw new Error('The runtime must have Apple Containerization’s valid Developer ID signature.')
    }
    const result = await this.run(binary, ['system', 'version', '--format', 'json'], { timeoutMs: 15000, env: this.env() })
    if (result.code !== 0) throw new Error('Could not identify the Apple container runtime version.')
    const version = parseArray(result.stdout).find((entry) => entry.appName === 'container')?.version
    if (typeof version !== 'string' || !/^1\.5\.\d+$/.test(version)) throw new Error('Bot VMs require the verified Apple container 1.5 release family; this version has not been qualified.')
    return { binary, version, sha256 }
  }
  private call(runtime: Runtime, args: string[], signal?: AbortSignal, timeoutMs = 15000, stdin?: string) { return this.run(runtime.binary, args, { signal, timeoutMs, stdin, env: this.env() }) }
  private labels(record: VmRecord) { return { [OWNER_LABEL]: this.profile!, [BOT_LABEL]: record.botKey, [SCHEMA_LABEL]: '1' } }
  private assertLabels(value: any, record: VmRecord) { if (!value || Object.entries(this.labels(record)).some(([key, expected]) => value[key] !== expected)) throw new Error('A VM with this name is not owned by this TensorFold profile; it has been left untouched.') }
  private async inspect(runtime: Runtime, record: VmRecord, signal?: AbortSignal): Promise<any | null> {
    const list = await this.call(runtime, ['list', '--all', '--format', 'json'], signal)
    if (list.code !== 0) throw new Error('Apple container services are stopped; start the runtime in VM setup.')
    if (!parseArray(list.stdout).some((value) => value.id === record.id || value.configuration?.id === record.id)) return null
    const result = await this.call(runtime, ['inspect', record.id], signal)
    if (result.code !== 0) throw new Error('The bot VM changed while inspecting it; refresh its status.')
    const rows = parseArray(result.stdout)
    if (rows.length !== 1) throw new Error('Unexpected VM identity response.')
    const value = rows[0], config = value.configuration
    if (value.id !== record.id || config?.id !== record.id) throw new Error('The VM identity does not match this bot.')
    this.assertLabels(config.labels, record)
    const mounts = config.mounts, image = config.image?.reference
    if (!Array.isArray(config.networks) || config.networks.length || !Array.isArray(config.publishedPorts) || config.publishedPorts.length || !Array.isArray(config.publishedSockets) || config.publishedSockets.length || config.ssh !== false || config.virtualization !== false || config.runtimeHandler !== 'container-runtime-linux' || config.readOnly !== true) throw new Error('The bot VM isolation settings changed; execution is blocked.')
    const user = config.initProcess?.user
    const unprivileged = user?.raw?.userString === '1000:1000' || (user?.id?.uid === 1000 && user?.id?.gid === 1000)
    if (!Array.isArray(mounts) || mounts.length !== 2 || !mounts.some((m: any) => m.destination === '/workspace' && m.type?.volume?.name === record.volume) || !mounts.some((m: any) => m.destination === '/tmp' && m.type?.tmpfs && m.options?.includes('size=128M')) || image !== record.image || !unprivileged || !Array.isArray(config.capAdd) || config.capAdd.length || !Array.isArray(config.capDrop) || !config.capDrop.includes('ALL')) throw new Error('The bot VM image, user or workspace mount changed; execution is blocked.')
    if (!await this.inspectVolume(runtime, record, signal)) throw new Error('The private workspace volume is missing; execution is blocked.')
    return value
  }
  private async inspectVolume(runtime: Runtime, record: VmRecord, signal?: AbortSignal): Promise<boolean> {
    const list = await this.call(runtime, ['volume', 'list', '--format', 'json'], signal)
    if (list.code !== 0) throw new Error('Could not inspect the bot’s persistent workspace.')
    if (!parseArray(list.stdout).some((value) => value.id === record.volume || value.configuration?.name === record.volume)) return false
    const result = await this.call(runtime, ['volume', 'inspect', record.volume], signal)
    if (result.code !== 0) throw new Error('The bot workspace changed while inspecting it.')
    const rows = parseArray(result.stdout), config = rows[0]?.configuration
    if (rows.length !== 1 || rows[0]?.id !== record.volume || config?.name !== record.volume) throw new Error('The persistent workspace does not match this bot.')
    this.assertLabels(config.labels, record)
    if (config.driver !== 'local' || config.format !== 'ext4' || !Number.isSafeInteger(config.sizeInBytes) || config.sizeInBytes <= 0 || config.sizeInBytes > VOLUME_CAPACITY) throw new Error('The private VM disk exceeds its 8 GiB limit or has an unsupported format; execution is blocked.')
    return true
  }
  private async checkDiskBudget() {
    const count = (await this.records()).length
    const diskRoot = this.options.runtimeAppRoot ?? this.managedRuntime?.appRoot ?? path.join(os.homedir(), 'Library', 'Application Support', 'com.apple.container')
    const free = this.options.freeDiskBytes ? await this.options.freeDiskBytes() : await fs.statfs(diskRoot).then((value) => value.bavail * value.bsize)
    if (!Number.isFinite(free) || free < count * VOLUME_CAPACITY + 2 * 1024 ** 3) throw new Error('Free disk space must cover each bot’s 8 GiB maximum workspace plus 2 GiB of headroom; free space or erase an unused VM before starting more work.')
  }
  private async cleanupPrepare(runtime: Runtime, record: VmRecord): Promise<void> {
    const id = `${record.id}-prepare`
    const list = await this.call(runtime, ['list', '--all', '--format', 'json'])
    if (list.code !== 0) throw new Error('Apple container services are stopped; setup ownership is preserved for recovery.')
    if (!parseArray(list.stdout).some((value) => value.id === id || value.configuration?.id === id)) return
    const result = await this.call(runtime, ['inspect', id])
    if (result.code !== 0) throw new Error('Could not verify this bot’s temporary workspace setup VM.')
    const rows = parseArray(result.stdout), value = rows[0], config = value?.configuration
    if (rows.length !== 1 || value?.id !== id || config?.id !== id) throw new Error('Unexpected setup VM identity; it has been preserved.')
    this.assertLabels(config.labels, record)
    if (config.image?.reference !== record.image || config.networks?.length !== 0 || config.publishedPorts?.length !== 0 || config.publishedSockets?.length !== 0 || config.mounts?.length !== 1 || config.mounts[0]?.type?.volume?.name !== record.volume || config.mounts[0]?.destination !== '/workspace') throw new Error('The setup VM configuration changed; it has been left untouched.')
    if (value.status?.state === 'running') {
      const stopped = await this.call(runtime, ['stop', '--time', '3', id], undefined, 20000)
      if (stopped.code !== 0) throw new Error('The temporary workspace setup VM could not stop; its ownership record is preserved.')
    }
    // --rm normally removes it on stop. Inspect the exact id before an explicit
    // delete, and never use --all, --force, prune or system stop.
    const after = await this.call(runtime, ['list', '--all', '--format', 'json'])
    if (after.code !== 0) throw new Error('Could not confirm workspace setup cleanup.')
    if (parseArray(after.stdout).some((entry) => entry.id === id || entry.configuration?.id === id)) {
      const removed = await this.call(runtime, ['delete', id])
      if (removed.code !== 0) throw new Error('The temporary setup VM could not be removed; its workspace is preserved.')
    }
  }
  private exclusive<T>(botId: string, work: (signal: AbortSignal) => Promise<T>, externalSignal?: AbortSignal): Promise<T> {
    const key = botKey(botId)
    if (this.closing) return Promise.reject(new Error('TensorFold is shutting down its bot VMs.'))
    const previous = this.queues.get(key) ?? Promise.resolve()
    const next = previous.catch(() => {}).then(async () => {
      if (this.closing) throw new Error('TensorFold is shutting down its bot VMs.')
      const controller = new AbortController(); this.active.set(key, controller)
      const abort = () => controller.abort(); externalSignal?.addEventListener('abort', abort, { once: true }); if (externalSignal?.aborted) controller.abort()
      let lease: { release(): void } | undefined
      try { lease = this.options.reserveOperation?.(); if (controller.signal.aborted) throw new Error('VM operation cancelled.'); return await work(controller.signal) } finally { externalSignal?.removeEventListener('abort', abort); lease?.release(); this.active.delete(key) }
    })
    this.queues.set(key, next)
    void next.finally(() => { if (this.queues.get(key) === next) this.queues.delete(key) }).catch(() => {})
    return next
  }
  async status(botId: string): Promise<BotVmStatus> {
    const key = botKey(botId), base = { botId, runtimeVersion: null as string | null, workspace: '/workspace' as const, network: 'none' as const, capabilities: [] as BotVmStatus['capabilities'], busy: this.queues.has(key), setupUrl: BOT_VM_SETUP_URL }
    if (!this.supported()) return { ...base, state: 'unsupported', detail: 'Bot VMs require Apple silicon and macOS 26 or later.' }
    try {
      const runtime = await this.runtime()
      if (!runtime) return { ...base, state: 'setup-required', detail: 'Install Apple container 1.5 from its signed official installer, then select its executable.' }
      base.runtimeVersion = runtime.version
      const record = await this.record(botId)
      if (!record) {
        const list = await this.call(runtime, ['list', '--all', '--format', 'json'])
        return { ...base, state: list.code === 0 ? 'not-created' : 'runtime-stopped', detail: list.code === 0 ? 'Create this bot’s isolated Linux workspace; the pinned Python image downloads once.' : 'Start Apple container services before creating this bot’s VM.' }
      }
      const current = await this.inspect(runtime, record)
      if (!current) return { ...base, state: 'not-created', detail: 'This bot’s VM is absent; Create restores it while preserving any owned workspace volume.' }
      const running = current.status?.state === 'running'
      return { ...base, state: running ? 'running' : 'stopped', capabilities: running ? ['files', 'commands'] : [], detail: running ? 'Isolated Linux files and commands are ready; network and host folder access are disabled.' : 'VM stopped; its private workspace is preserved.' }
    } catch (error) { const detail = (error as Error).message; return { ...base, state: /services are stopped/.test(detail) ? 'runtime-stopped' : 'blocked', detail } }
  }
  async setRuntimePath(binary: string): Promise<void> {
    if (this.queues.size) throw new Error('Stop current VM operations before changing Apple runtime tools.')
    if (typeof binary !== 'string' || !path.isAbsolute(binary) || binary.length > 4096 || binary.includes('\0')) throw new Error('Choose the signed Apple container executable.')
    const candidate = new BotVmService({ ...this.options, runtimePath: binary })
    // A replaced saved runtime may be re-approved only through this native selector.
    const pinFile = path.join(this.root, 'runtime.json')
    const info = await fs.lstat(binary)
    if (!info.isFile() || info.isSymbolicLink() || (info.mode & 0o022) !== 0 || info.size > 200 * 1024 * 1024 || path.basename(binary) !== 'container') throw new Error('Choose the signed Apple container executable.')
    await this.initialize()
    const previous = await regularJson(pinFile)
    const managed = await this.selectedManagedRuntime()
    if (managed && binary !== managed.binary) {
      if ((await this.records()).length) throw new Error('Existing bot workspaces belong to the managed runtime; keep that runtime until those disks are explicitly erased.')
      await stopManagedRuntimeServices(managed, this.run)
    }
    const sha256 = createHash('sha256').update(await fs.readFile(binary)).digest('hex')
    await atomicJson(pinFile, { version: 1, binary, sha256, ...(managed && binary === managed.binary ? { managed: true } : {}) })
    try { if (!await candidate.runtime()) throw new Error('The selected runtime is unavailable.') }
    catch (error) { if (previous) await atomicJson(pinFile, previous); else await fs.rm(pinFile, { force: true }); throw error }
  }
  async startRuntime(): Promise<void> {
    const runtime = await this.runtime()
    if (!runtime) throw new Error('Install and select the signed Apple container runtime first.')
    if (this.managedRuntime) { await this.installer.verifyInstalled(this.managedRuntime); await assertManagedRuntimeAvailable(this.managedRuntime, this.run) }
    const appRoot = this.options.runtimeAppRoot ?? this.managedRuntime?.appRoot, installRoot = this.options.runtimeInstallRoot ?? this.managedRuntime?.installRoot
    const result = await this.call(runtime, ['system', 'start', '--enable-kernel-install', ...(appRoot ? ['--app-root', appRoot] : []), ...(installRoot ? ['--install-root', installRoot] : [])], undefined, 180000)
    if (result.code !== 0) throw new Error('Apple container services could not start; review Apple’s setup requirements and retry.')
  }
  setupStatus(): BotVmSetupStatus { return { ...this.installer.snapshot(), active: this.setupBotId !== null, botId: this.setupBotId ?? this.lastSetupBotId } }
  cancelSetup(): BotVmSetupStatus { if (!this.setupBotId) return this.setupStatus(); this.setupAbort?.abort(); this.active.get(botKey('managed-setup'))?.abort(); this.installer.progress('cancelled', 'Stopping VM setup; verified files and private workspace data are preserved.'); return this.setupStatus() }
  async managedSetup(botId: string, assertCanStart?: () => void): Promise<BotVmStatus> {
    botKey(botId)
    if (!this.supported()) throw new Error('Managed bot VMs require Apple silicon and macOS 26 or later.')
    if (this.queues.size) throw new Error('Wait for current VM operations before setting up the runtime.')
    this.setupBotId = botId
    this.lastSetupBotId = botId
    this.setupAbort = new AbortController()
    try { await this.exclusive('managed-setup', async (signal) => {
      assertCanStart?.()
      await this.initialize(); const pinFile = path.join(this.root, 'runtime.json'), previous = await regularJson(pinFile)
      if (previous?.managed !== true && (await this.records()).length) throw new Error('Existing bot workspaces belong to the selected runtime; keep using that runtime to preserve them.')
      const candidate = this.installer.paths()
      await assertManagedRuntimeAvailable(candidate, this.run)
      try {
        const installed = await this.installer.install(signal)
        const sha256 = createHash('sha256').update(await fs.readFile(installed.binary)).digest('hex')
        await atomicJson(pinFile, { version: 1, binary: installed.binary, sha256, managed: true })
        this.managedVerified = false
        this.installer.progress('starting', 'Starting TensorFold-owned Apple VM services; Linux support files may download on first use.')
        try {
          const runtime = await this.runtime(); if (!runtime) throw new Error('The managed runtime could not be identified.')
          await assertManagedRuntimeAvailable(installed, this.run)
          const started = await this.call(runtime, ['system', 'start', '--enable-kernel-install', '--app-root', installed.appRoot, '--install-root', installed.installRoot], signal, 180000)
          if (started.code !== 0) throw new Error('The private Apple VM services could not start; retry setup.')
        } catch (error) {
          await stopManagedRuntimeServices(installed, this.run)
          if (previous) await atomicJson(pinFile, previous); else await fs.rm(pinFile, { force: true })
          this.managedRuntime = null; this.managedVerified = false; throw error
        }
        this.installer.progress('provisioning', 'Creating this bot’s private 8 GiB workspace and downloading the pinned 20 MB guest image.')
        assertCanStart?.(); await this.provision(botId, signal); assertCanStart?.(); await this.start(botId, signal)
        this.installer.progress('ready', 'This bot’s isolated Linux workspace is ready; host folders and guest network access are disabled.')
      } catch (error) { this.installer.progress(signal.aborted ? 'cancelled' : 'error', signal.aborted ? 'VM setup cancelled; retry when you are ready.' : (error as Error).message); throw error }
    }, this.setupAbort.signal) } catch (error) { this.installer.progress(this.setupAbort.signal.aborted ? 'cancelled' : 'error', this.setupAbort.signal.aborted ? 'VM setup cancelled; retry when you are ready.' : (error as Error).message); throw error } finally { this.setupBotId = null; this.setupAbort = null }
    return this.status(botId)
  }
  async provision(botId: string, outerSignal?: AbortSignal): Promise<BotVmStatus> {
    await this.exclusive(botId, async (signal) => {
      const runtime = await this.runtime(); if (!runtime) throw new Error('Install and select Apple container before creating a bot VM.')
      await this.initialize(); const record = this.ids(botId)
      const existingRecord = await this.record(botId)
      if (!existingRecord) await atomicJson(path.join(this.root, `${record.botKey}.json`), record)
      const existing = await this.inspect(runtime, record, signal)
      if (existing) return
      await this.checkDiskBudget()
      await this.cleanupPrepare(runtime, record)
      const args = Object.entries(this.labels(record)).flatMap(([key, value]) => ['--label', `${key}=${value}`])
      if (!await this.inspectVolume(runtime, record, signal)) {
        const volume = await this.call(runtime, ['volume', 'create', ...args, '--opt', 'size=8g', record.volume], signal, 30000)
        if (volume.code !== 0 || !await this.inspectVolume(runtime, record, signal)) throw new Error('The bot workspace could not be created; existing data has been preserved.')
      }
      const pull = await this.call(runtime, ['image', 'pull', '--platform', 'linux/arm64', '--scheme', 'https', record.image], signal, 180000)
      if (pull.code !== 0) throw new Error('The pinned bot image download failed; retry Create when the registry is available.')
      // A separate fixed-command helper prepares the guest-owned volume. The bot
      // itself always boots as uid 1000 with every Linux capability dropped.
      let prepared
      try { prepared = await this.call(runtime, ['run', '--name', `${record.id}-prepare`, '--rm', '--network', 'none', '--no-dns', '--read-only', '--cap-drop', 'ALL', '--cap-add', 'CHOWN', '--cpus', '1', '--memory', '256M', '--mount', `type=volume,source=${record.volume},target=/workspace`, '--entrypoint', '/usr/local/bin/python3', ...args, record.image, '-I', '-c', PREPARE_VOLUME], signal, 60000) }
      finally { await this.cleanupPrepare(runtime, record) }
      if (prepared.code !== 0) throw new Error('The bot workspace permissions could not be prepared; its data has been preserved.')
      const created = await this.call(runtime, ['create', '--name', record.id, ...args, '--network', 'none', '--no-dns', '--read-only', '--cap-drop', 'ALL', '--user', '1000:1000', '--cpus', '2', '--memory', '1G', '--ulimit', 'nproc=256:256', '--ulimit', 'nofile=1024:1024', '--ulimit', 'fsize=104857600:104857600', '--tmpfs', '/tmp:size=128M,mode=1777', '--mount', `type=volume,source=${record.volume},target=/workspace`, '--entrypoint', '/usr/local/bin/python3', record.image, '-I', '-c', INIT], signal, 60000)
      if (created.code !== 0) throw new Error('The isolated bot VM could not be created; its owned workspace has been preserved.')
      if (!await this.inspect(runtime, record, signal)) throw new Error('The new bot VM could not be verified.')
    }, outerSignal)
    return this.status(botId)
  }
  async start(botId: string, outerSignal?: AbortSignal): Promise<BotVmStatus> {
    await this.exclusive(botId, async (signal) => {
      const runtime = await this.runtime(), record = await this.record(botId)
      if (!runtime || !record) throw new Error('Create this bot’s VM before starting it.')
      await this.checkDiskBudget()
      const current = await this.inspect(runtime, record, signal)
      if (!current) throw new Error('The VM is missing; Create restores its persistent workspace.')
      if (current.status?.state !== 'running') { const result = await this.call(runtime, ['start', record.id], signal, 60000); if (result.code !== 0) throw new Error('The bot VM could not start; its private workspace is preserved.') }
      const probe = await this.guest(runtime, record, 'list_dir', {}, signal)
      if (!probe.ok) throw new Error('The bot VM started but its private workspace is not ready.')
    }, outerSignal)
    return this.status(botId)
  }
  private async stopOwned(runtime: Runtime, record: VmRecord) {
    const current = await this.inspect(runtime, record)
    if (current?.status?.state === 'running') { const result = await this.call(runtime, ['stop', '--time', '3', record.id], undefined, 20000); if (result.code !== 0 || (await this.inspect(runtime, record))?.status?.state === 'running') throw new Error('The bot VM could not stop; check its runtime status before continuing.') }
  }
  async stop(botId: string): Promise<BotVmStatus> {
    if (this.setupBotId === botId) {
      this.setupAbort?.abort()
      this.active.get(botKey('managed-setup'))?.abort()
      await this.queues.get(botKey('managed-setup'))?.catch(() => {})
    }
    this.active.get(botKey(botId))?.abort()
    await this.exclusive(botId, async () => { const runtime = await this.runtime(), record = await this.record(botId); if (runtime && record) await this.stopOwned(runtime, record) })
    return this.status(botId)
  }
  async rebuild(botId: string): Promise<BotVmStatus> {
    this.active.get(botKey(botId))?.abort()
    await this.exclusive(botId, async () => {
      const runtime = await this.runtime(), record = await this.record(botId)
      if (!runtime || !record) throw new Error('Create this bot’s VM first.')
      const current = await this.inspect(runtime, record)
      if (!current) return
      if (!await this.inspectVolume(runtime, record)) throw new Error('The private workspace is unavailable; rebuild has been refused to preserve data.')
      await this.stopOwned(runtime, record)
      if (!await this.inspect(runtime, record)) return
      const result = await this.call(runtime, ['delete', record.id])
      if (result.code !== 0) throw new Error('The VM could not be rebuilt; its workspace is preserved.')
    })
    return this.provision(botId)
  }
  async requireReady(botId: string): Promise<void> { const status = await this.status(botId); if (status.state !== 'running' || !status.capabilities.includes('commands') || !status.capabilities.includes('files')) throw new Error(status.detail) }
  private async guest(runtime: Runtime, record: VmRecord, tool: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<BotVmResult> {
    const input = JSON.stringify({ tool, args })
    if (Buffer.byteLength(input) > 1090000) throw new Error('VM request exceeds the one MiB text limit.')
    const timeout = tool === 'run_command' ? Math.max(1000, Math.min(120000, Number(args.timeoutMs) || 30000)) + 10000 : 15000
    const result = await this.call(runtime, ['exec', '--interactive', '--user', '1000:1000', '--workdir', '/workspace', record.id, '/usr/local/bin/python3', '-I', '-c', BOT_VM_GUEST], signal, timeout, input)
    if (result.code !== 0) throw new Error('The VM tool process failed; no host fallback was used.')
    const value: BotVmResult = JSON.parse(result.stdout)
    if (!value || typeof value.ok !== 'boolean' || typeof value.output !== 'string' || value.output.length > 100000 || (value.changedPaths && (!Array.isArray(value.changedPaths) || value.changedPaths.some((p) => typeof p !== 'string' || p.startsWith('/') || p.split('/').includes('..'))))) throw new Error('The VM tool returned an invalid response.')
    return value
  }
  async executeTool(botId: string, tool: string, args: Record<string, unknown>, signal?: AbortSignal): Promise<BotVmResult> {
    if (!(VM_TOOL_NAMES as readonly string[]).includes(tool)) return { ok: false, output: 'This tool is unavailable inside this bot VM; no host fallback is permitted.' }
    return this.exclusive(botId, async (inner) => {
      const abort = () => this.active.get(botKey(botId))?.abort()
      signal?.addEventListener('abort', abort, { once: true })
      try {
        if (signal?.aborted) throw new Error('VM operation cancelled.')
        const runtime = await this.runtime(), record = await this.record(botId)
        if (!runtime || !record) throw new Error('Create and start this bot’s VM before sending work.')
        const current = await this.inspect(runtime, record, inner)
        if (current?.status?.state !== 'running') throw new Error('Start this bot’s VM before sending work.')
        try {
          const result = await this.guest(runtime, record, tool, args, inner)
          if (tool === 'run_command') {
            // Recycling the hardware boundary also kills detached sessions and
            // closed-output grandchildren, which a shell process group cannot.
            await this.stopOwned(runtime, record)
            if (!result.interrupted && !inner.aborted && !this.closing) {
              await this.checkDiskBudget()
              const restarted = await this.call(runtime, ['start', record.id], inner, 60000)
              if (restarted.code !== 0 || (await this.inspect(runtime, record, inner))?.status?.state !== 'running') throw new Error('The command finished but its VM could not restart; the private workspace is preserved.')
            }
          }
          return result
        }
        catch (error) { await this.stopOwned(runtime, record); throw error }
      } finally { signal?.removeEventListener('abort', abort) }
    })
  }
  private async records(): Promise<VmRecord[]> {
    await this.initialize()
    const records: VmRecord[] = []
    for (const name of await fs.readdir(this.root)) {
      if (!/^[a-f0-9]{24}\.json$/.test(name)) continue
      const value = await regularJson(path.join(this.root, name))
      const key = name.slice(0, -5), id = `tfbot-${this.profile!.slice(0, 16)}-${key}`
      if (value?.version !== 1 || value.botKey !== key || value.id !== id || value.volume !== `${id}-data` || value.image !== BOT_VM_IMAGE) throw new Error('A saved VM ownership record is invalid; all related data has been preserved.')
      records.push(value)
    }
    return records
  }
  async quiesce(): Promise<{ stopped: number }> {
    if (this.quiescing) return this.quiescing
    this.closing = true
    this.quiescing = (async () => {
      for (const controller of this.active.values()) controller.abort()
      await Promise.allSettled([...this.queues.values()])
      const records = await this.records()
      if (!records.length) return { stopped: 0 }
      const runtime = await this.runtime()
      if (!runtime) throw new Error('Restore the signed Apple runtime to stop or erase saved bot VMs; their ownership records have been preserved.')
      if (this.managedRuntime) {
        const available = await this.call(runtime, ['list', '--all', '--format', 'json'])
        if (available.code !== 0) {
          // Cleanup may follow an earlier completed Quit or an interrupted boot.
          // Reopen only this profile's signed service to inspect and stop its VMs.
          await assertManagedRuntimeAvailable(this.managedRuntime, this.run)
          const restored = await this.call(runtime, ['system', 'start', '--disable-kernel-install', '--app-root', this.managedRuntime.appRoot, '--install-root', this.managedRuntime.installRoot], undefined, 60000)
          if (restored.code !== 0) throw new Error('The private runtime could not reopen for cleanup; bot ownership records and disks have been preserved.')
        }
      }
      let stopped = 0
      for (const record of records) { await this.cleanupPrepare(runtime, record); const value = await this.inspect(runtime, record); if (value?.status?.state === 'running') { await this.stopOwned(runtime, record); stopped++ } }
      return { stopped }
    })()
    try { return await this.quiescing } finally { this.quiescing = null }
  }
  resume(): void { if (this.quiescing) throw new Error('Wait for VM operations to stop before resuming.'); this.closing = false }
  /** Called only from the application's explicit erase-all-data action. A partial
   * failure retains the owner ledger so a later repair can finish erasing disks. */
  async purgeOwnedData(): Promise<{ containers: number; volumes: number }> {
    await this.quiesce()
    const records = await this.records()
    if (!records.length) { await this.purgeManagedRuntime(); return { containers: 0, volumes: 0 } }
    const runtime = await this.runtime()
    if (!runtime) throw new Error('The signed Apple runtime is required to erase bot VM disks; ownership records have been preserved.')
    let containers = 0, volumes = 0
    for (const record of records) {
      if (await this.inspect(runtime, record)) {
        const removed = await this.call(runtime, ['delete', record.id])
        if (removed.code !== 0 || await this.inspect(runtime, record)) throw new Error('A bot VM could not be erased; its ownership record and disk have been preserved.')
        containers++
      }
      if (await this.inspectVolume(runtime, record)) {
        const removed = await this.call(runtime, ['volume', 'delete', record.volume])
        if (removed.code !== 0 || await this.inspectVolume(runtime, record)) throw new Error('A private VM disk could not be erased; its ownership record has been preserved for recovery.')
        volumes++
      }
      await fs.rm(path.join(this.root, `${record.botKey}.json`))
    }
    await this.purgeManagedRuntime()
    return { containers, volumes }
  }
  private async selectedManagedRuntime(): Promise<ManagedVmRuntime | null> {
    await this.initialize(); const pin = await regularJson(path.join(this.root, 'runtime.json'))
    if (pin?.managed !== true) return null
    const runtime = this.installer.paths()
    if (pin.binary !== runtime.binary) throw new Error('The managed runtime ownership selection is invalid; its files have been preserved.')
    return runtime
  }
  private async purgeManagedRuntime() {
    const runtime = await this.selectedManagedRuntime()
    if (runtime) await stopManagedRuntimeServices(runtime, this.run)
    try { const info = await fs.lstat(this.installer.root); if (!info.isDirectory() || info.isSymbolicLink()) throw new Error('The managed runtime cache is linked or invalid; erasure has been refused.'); await fs.rm(this.installer.root, { recursive: true, force: true }) }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
    if (runtime) await fs.rm(path.join(this.root, 'runtime.json'), { force: true })
    this.managedRuntime = null; this.managedVerified = false
  }
  async shutdown(): Promise<void> { await this.quiesce(); const runtime = await this.selectedManagedRuntime(); if (runtime) await stopManagedRuntimeServices(runtime, this.run) }
}
