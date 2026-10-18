import { passiveNodeLinks } from './passive-links'
import { promises as fs, constants } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { randomBytes, randomUUID } from 'node:crypto'
import { validateClusterProfile, validateDistributedProfiles } from '../../shared/cluster-validation'
import type { ClusterNodeLaunch, ClusterProfile } from '../../shared/cluster-workspace'
import { cachedModels, quantizationFormat, readCachedConfig, validateCachedFiles } from './cache'
import { privateDirectory } from './config'
import { nativeCli, type CliOutput } from './command'
import { freeLoopbackPort, NativeSession } from './native-session'
import type { ClusterNodeConfig, EngineAdapter, EngineInspection, EngineSession, NodeAdmissionInput } from './types'

const SINGLE_FAMILIES = new Set(['nemotron_h', 'qwen4_exp', 'glm5_next', 'qwen3_5'])
const GIB = 1024 ** 3
const SERVE_FLAGS = ['--host', '--port', '--name', '--api-key-file', '--context', '--max-tokens', '--temperature', '--parallel', '--prompt-cache-gib', '--no-thinking', '--no-update-check', '--dashboard']
interface Capabilities { version: string; backends: string[]; families: Record<string, string[]>; flags: Set<string>; declaredDistributed: ClusterProfile[] }
export interface OwnedNativeSession extends EngineSession { ready(signal: AbortSignal): Promise<void> }
export interface NativeAdapterOptions {
  config: ClusterNodeConfig
  runCli?: (args: string[], signal: AbortSignal, timeoutMs?: number) => Promise<CliOutput>
  memory?: () => { total: number; free: number }
  platform?: 'darwin' | 'linux'
  arch?: 'arm64' | 'x64'
  passiveSysfsRoot?: string | false
  sessionFactory?: (binary: string, args: string[], env: NodeJS.ProcessEnv, model: string, apiKey: string, keyFile: string, port: number) => OwnedNativeSession
}
/** This adapter implements released single-native routes; missing distributed JSON operations stay refused. */
export class NodeEngineAdapter implements EngineAdapter {
  private platform: 'darwin' | 'linux'
  private arch: 'arm64' | 'x64'
  readonly cacheDir: string
  constructor(private options: NativeAdapterOptions) {
    this.platform = options.platform ?? process.platform as 'darwin' | 'linux'
    this.arch = options.arch ?? process.arch as 'arm64' | 'x64'
    this.cacheDir = options.config.cacheDir ?? path.join(options.config.dataDir, 'model-cache')
  }
  private env(download = false): NodeJS.ProcessEnv {
    const env: NodeJS.ProcessEnv = {}
    for (const key of ['PATH', 'HOME', 'TMPDIR', 'TMP', 'TEMP', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'LC_CTYPE', 'TZ', 'LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'ROCR_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'SSL_CERT_FILE', 'SSL_CERT_DIR']) if (process.env[key] !== undefined) env[key] = process.env[key]
    if (download) for (const key of ['HF_TOKEN', 'HUGGING_FACE_HUB_TOKEN', 'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY', 'https_proxy', 'http_proxy', 'all_proxy', 'no_proxy']) if (process.env[key] !== undefined) env[key] = process.env[key]
    env.HF_HUB_CACHE = this.cacheDir; env.TENSORFOLD_NO_UPDATE_CHECK = '1'
    return env
  }

  private async cli(args: string[], signal: AbortSignal, timeoutMs?: number): Promise<CliOutput> {
    if (this.options.runCli) return this.options.runCli(args, signal, timeoutMs)
    const binary = this.options.config.enginePath
    if (!binary) throw new Error('Choose an approved installed tensorfold-native binary for this node.')
    const stat = await fs.stat(binary)
    if (!stat.isFile() || !(stat.mode & 0o111) || stat.mode & 0o022) throw new Error('The approved native engine must be an executable file without shared write access.')
    return nativeCli(binary, args, this.env(args[0] === 'pull'), signal, timeoutMs)
  }
  private async capabilities(signal: AbortSignal): Promise<Capabilities> {
    const output = await this.cli(['capabilities', '--json'], signal)
    if (output.code !== 0) throw new Error('The installed native engine refused its capabilities inspection.')
    const doc = JSON.parse(output.stdout)
    if (!doc || doc.schema !== 1 || doc.engine !== 'zig' || typeof doc.version !== 'string' || !/^\d+\.\d+\.\d+$/.test(doc.version) || !Array.isArray(doc.backends) || !doc.families || typeof doc.families !== 'object' || Array.isArray(doc.families) || !doc.serve?.flags || typeof doc.serve.flags !== 'object') throw new Error('The installed engine returned unsupported capabilities metadata.')
    const families: Record<string, string[]> = {}
    for (const [name, formats] of Object.entries(doc.families)) if (SINGLE_FAMILIES.has(name) && Array.isArray(formats) && formats.length <= 64 && formats.every(format => typeof format === 'string' && /^[A-Za-z0-9_.-]{1,80}$/.test(format))) families[name] = formats as string[]
    const backend = this.platform === 'darwin' ? 'metal' : 'cuda'
    if (!doc.backends.includes(backend) || !Object.keys(families).length) throw new Error('This node does not have a source-supported native backend and model family.')
    const flags = new Set(Object.keys(doc.serve.flags))
    if (SERVE_FLAGS.some(flag => !flags.has(flag))) throw new Error('The installed engine does not report the complete single-node launch and idle-health flags.')
    const declaredDistributed = doc.distributed === undefined ? [] : validateDistributedProfiles(doc.distributed)
    return { version: doc.version, backends: doc.backends, families, flags, declaredDistributed }
  }
  private profile(caps: Capabilities): ClusterProfile {
    return validateClusterProfile({ id: 'single-native', title: 'Single native engine', mode: 'single', platform: this.platform, arch: this.arch, ranks: 1, transports: ['tcp'], modelTypes: Object.keys(caps.families), poolsWeights: false, requiresFabric: false, engineVersion: caps.version })
  }
  async inspect(signal: AbortSignal): Promise<EngineInspection> {
    const hints = await passiveNodeLinks(this.platform, this.options.passiveSysfsRoot).catch(() => [])
    try {
      const caps = await this.capabilities(signal)
      const models = await cachedModels(this.cacheDir, new Set(Object.keys(caps.families)))
      return { available: true, version: caps.version, profiles: [this.profile(caps)], models, links: [{ transport: 'tcp', available: true, verified: false, checkedAt: Date.now(), devices: [], message: 'Authenticated request transport; no distributed weights or RDMA proof.' }, ...hints], reason: caps.declaredDistributed.length ? 'Distributed profiles require a native JSON admission, immutable checkpoint, link and session adapter; this node does not substitute generic cluster commands.' : 'The installed engine does not report versioned distributed admission, checkpoint, link and health operations; generic cluster launch and heterogeneous weights remain unavailable.' }
    } catch { return { available: false, version: null, profiles: [], models: [], links: hints, reason: 'The installed native engine or its bounded capability/cache inspection is unavailable.' } }
  }
  async admission(input: NodeAdmissionInput, signal: AbortSignal) {
    const memory = this.options.memory?.() ?? { total: os.totalmem(), free: os.freemem() }
    const fraction = input.role === 'workstation' ? 0.6 : 0.7
    const budgetBytes = Math.max(0, Math.floor(Math.min(memory.total * fraction, memory.free - GIB)))
    const refused = (reason: string, cached = false, requiredBytes: number | null = null, profile: ClusterProfile | null = null) => ({ allowed: false, cached, requiredBytes, budgetBytes, profile, reason })
    let caps: Capabilities
    try { caps = await this.capabilities(signal) } catch { return refused('The approved native engine does not provide supported CPU capabilities.') }
    const profile = this.profile(caps)
    if (input.profileId !== profile.id || input.worldSize !== 1 || input.rank !== 0 || input.transport !== 'tcp' || input.links.length) return refused('Distributed activation requires implemented versioned native per-family admission, immutable checkpoint, link and health operations. Generic --cluster and mixed Metal/CUDA fallback are refused.')
    const cached = await readCachedConfig(this.cacheDir, input.model)
    if (!cached) return refused('Stage this exact immutable model revision before requesting native load admission.', false, null, profile)
    if (!profile.modelTypes.includes(cached.config.model_type)) return refused('The cached checkpoint model type is not reported by this native single-host profile.', true, null, profile)
    const format = quantizationFormat(cached.config)
    if (!format || !caps.families[cached.config.model_type]?.includes(format)) return refused('The installed engine does not report this checkpoint quantization for its model type.', true, null, profile)
    try { await validateCachedFiles(this.cacheDir, cached.directory) } catch { return refused('The immutable checkpoint is partial or resolves outside its approved cache.', true, null, profile) }
    const info = await this.cli(['info', cached.directory], signal)
    if (info.code !== 0) return refused('The native engine refused inspection of this immutable cached revision.', true, null, profile)
    const weights = /^weights: .*\((\d+) bytes\)$/m.exec(info.stdout), floor = /^memory floor: ([\d.]+) GiB \(weights plus a ([\d.]+) GiB runtime margin\)$/m.exec(info.stdout)
    if (!weights || !floor) return refused('The native engine did not report a usable bounded memory floor.', true, null, profile)
    const requiredBytes = Number(weights[1]) + Math.round(Number(floor[2]) * GIB)
    if (!Number.isSafeInteger(requiredBytes) || requiredBytes <= 0) return refused('The native engine reported an invalid model memory floor.', true, null, profile)
    const context = Number(/^context: (\d+)$/m.exec(info.stdout)?.[1])
    if (context > 0 && input.generation.contextTokens > context) return refused('The requested context exceeds this checkpoint window.', true, requiredBytes, profile)
    if ((cached.config.model_type === 'qwen4_exp' || cached.config.model_type === 'glm5_next') && input.generation.temperature !== 0) return refused('This native family reports greedy decoding; choose temperature zero.', true, requiredBytes, profile)
    if (requiredBytes > budgetBytes) return refused('The native model floor exceeds this node role budget or current free host memory.', true, requiredBytes, profile)
    return { allowed: true, cached: true, requiredBytes, budgetBytes, profile, reason: 'The native host-memory floor fits its role budget; device, context growth and final load admission are still enforced by the engine.' }
  }
  async ensureModel(model: NodeAdmissionInput['model'], signal: AbortSignal, progress: (message: string) => Promise<void>): Promise<void> {
    const caps = await this.capabilities(signal)
    await fs.mkdir(this.cacheDir, { recursive: true, mode: 0o700 })
    const root = await fs.lstat(this.cacheDir)
    if (!root.isDirectory() || root.isSymbolicLink()) throw new Error('The approved model cache root must be a real directory.')
    const space = await fs.statfs(this.cacheDir)
    if (Number(space.bavail) * Number(space.bsize) < 2 * GIB) throw new Error('The approved model cache needs at least two GiB free before staging.')
    await progress('Staging the requested immutable revision; native download checks and resume rules apply.')
    const result = await this.cli(['pull', `${model.repo}@${model.revision}`], signal, 60 * 60_000)
    if (result.code !== 0) throw new Error('Native model staging failed or remains partial; no loaded model was substituted.')
    const cached = await readCachedConfig(this.cacheDir, model)
    if (!cached) throw new Error('Native staging did not produce the exact requested immutable revision.')
    await validateCachedFiles(this.cacheDir, cached.directory)
    if (!Object.prototype.hasOwnProperty.call(caps.families, cached.config.model_type)) throw new Error('The staged checkpoint type is not reported by this node native profile.')
    await progress('The immutable revision is present; load admission is checked separately.')
  }
  async start(input: ClusterNodeLaunch, signal: AbortSignal, progress: (message: string) => Promise<void>): Promise<EngineSession> {
    const admitted = await this.admission(input, signal)
    if (!admitted.allowed || admitted.profile?.mode !== 'single') throw new Error(admitted.reason)
    const cached = await readCachedConfig(this.cacheDir, input.model)
    if (!cached || !this.options.config.enginePath) throw new Error('The approved model revision or installed native binary is unavailable.')
    await privateDirectory(this.options.config.dataDir)
    const keyFile = path.join(this.options.config.dataDir, `.engine-${randomUUID()}.token`), key = randomBytes(32).toString('hex')
    const handle = await fs.open(keyFile, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600)
    try { await handle.writeFile(key + '\n'); await handle.sync() } finally { await handle.close() }
    let port: number
    try { port = await freeLoopbackPort(); await progress('Load admission rechecked; starting one owned native process on loopback.'); if (signal.aborted) throw new Error('Native load cancelled before process startup.') } catch (error) { await fs.unlink(keyFile).catch(() => {}); throw error }
    const args = ['serve', cached.directory, '--host', '127.0.0.1', '--port', String(port), '--name', input.deploymentId, '--api-key-file', keyFile, '--context', String(input.generation.contextTokens), '--max-tokens', String(input.generation.maxOutputTokens), '--temperature', String(input.generation.temperature), '--parallel', '1', '--prompt-cache-gib', '0', '--no-thinking', '--no-update-check', '--dashboard']
    const createSession = this.options.sessionFactory ?? ((...args: ConstructorParameters<typeof NativeSession>) => new NativeSession(...args))
    const session = createSession(this.options.config.enginePath, args, this.env(), input.deploymentId, key, keyFile, port)
    try { await session.ready(signal); await progress('The owned native model reports authenticated readiness; a canary is still required.'); return session }
    catch (error) { await session.stop(); throw error }
  }
}
