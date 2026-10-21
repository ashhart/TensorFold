import { createHash, verify } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { canUseMacSandbox, macSandboxProfile } from './sandbox'
export { mcpSpawnFingerprint } from './mcp-launch-identity'
import type { McpDockerGatewayConfig, McpDockerMount, McpDockerNetworkMode, McpServerConfig, McpServerProvenance } from './types'

const TRUSTED_MCP_ENV = 'VONTRA_ALLOW_TRUSTED_MCP'
const REQUIRE_SIGNED_MCP_ENV = 'VONTRA_REQUIRE_SIGNED_MCP'
const UNSANDBOXED_MCP_ENV = 'VONTRA_ALLOW_UNSANDBOXED_MCP'

export interface McpSpawnSpec {
  command: string
  args: string[]
  cwd: string
  env: NodeJS.ProcessEnv
  gateway: McpServerProvenance['gateway']
  sandboxed: boolean
}

export interface McpSpawnSpecOptions {
  macSandboxAvailable?: boolean
}

function childEnv(extra: Record<string, string> = {}): NodeJS.ProcessEnv {
  const base: NodeJS.ProcessEnv = {}
  for (const key of ['PATH', 'TMPDIR']) {
    const value = process.env[key]
    if (value) base[key] = value
  }
  return { ...base, ...extra }
}

function normalizeDockerGateway(gateway: McpDockerGatewayConfig | null | undefined): McpDockerGatewayConfig | null {
  if (!gateway) return null
  const image = String(gateway.image ?? '').trim()
  const workdir = String(gateway.workdir ?? '').trim() || null
  return {
    image,
    workdir,
    network: normalizeDockerNetwork(gateway.network),
    readOnlyRootfs: gateway.readOnlyRootfs !== false,
    envAllowlist: uniqueStrings(gateway.envAllowlist ?? []).filter((key) => /^[A-Z_][A-Z0-9_]*$/i.test(key)),
    mounts: (gateway.mounts ?? []).map(normalizeDockerMount).filter((mount): mount is McpDockerMount => !!mount),
  }
}

function normalizeDockerNetwork(value: McpDockerNetworkMode | null | undefined): McpDockerNetworkMode {
  return value === 'bridge' || value === 'host' ? value : 'none'
}

function normalizeDockerMount(mount: McpDockerMount): McpDockerMount | null {
  const source = String(mount.source ?? '').trim()
  const target = String(mount.target ?? '').trim()
  if (!source || !target) return null
  return { source, target, readOnly: mount.readOnly !== false }
}

function uniqueStrings(items: string[]): string[] {
  return [...new Set(items.map((item) => String(item).trim()).filter(Boolean))]
}

export function normalizeServerConfig(config: McpServerConfig): McpServerConfig {
  const approvalMode = config.approvalMode === 'trusted' && !trustedMcpAllowed() ? 'ask' : (config.approvalMode ?? 'ask')
  return {
    ...config,
    transport: 'stdio',
    launchMode: config.launchMode === 'docker_gateway' ? 'docker_gateway' : 'direct',
    args: Array.isArray(config.args) ? config.args.map(String) : [],
    env: { ...config.env },
    dockerGateway: normalizeDockerGateway(config.dockerGateway),
    manifestPath: config.manifestPath ?? null,
    manifestSignature: config.manifestSignature ?? null,
    manifestPublicKey: config.manifestPublicKey ?? null,
    manifestVerifiedAt: config.manifestVerifiedAt ?? null,
    manifestDigest: config.manifestDigest ?? null,
    enabled: config.enabled ?? true,
    autoConnect: config.autoConnect ?? false,
    approvalMode,
  }
}

export function createMcpServer(input: Partial<McpServerConfig>): McpServerConfig {
  const now = Date.now()
  return normalizeServerConfig({
    id: input.id || `mcp_${Math.random().toString(36).slice(2, 9)}`,
    name: input.name || 'New MCP server',
    transport: 'stdio',
    launchMode: input.launchMode === 'docker_gateway' ? 'docker_gateway' : 'direct',
    command: input.command || '',
    args: input.args ?? [],
    cwd: input.cwd ?? null,
    env: input.env ?? {},
    dockerGateway: normalizeDockerGateway(input.dockerGateway),
    manifestPath: input.manifestPath ?? null,
    manifestSignature: input.manifestSignature ?? null,
    manifestPublicKey: input.manifestPublicKey ?? null,
    manifestVerifiedAt: input.manifestVerifiedAt ?? null,
    manifestDigest: input.manifestDigest ?? null,
    enabled: input.enabled ?? true,
    autoConnect: input.autoConnect ?? false,
    approvalMode: input.approvalMode ?? 'ask',
    createdAt: input.createdAt ?? now,
    updatedAt: input.updatedAt ?? now,
  })
}

interface ManifestVerificationResult {
  ok: boolean
  digest: string | null
  error: string | null
}

export async function verifyConfiguredManifest(config: McpServerConfig): Promise<ManifestVerificationResult> {
  if (!config.manifestPath && !config.manifestSignature && !config.manifestPublicKey) {
    return { ok: false, digest: null, error: 'Signed MCP manifest is not configured.' }
  }
  if (!config.manifestPath || !config.manifestSignature || !config.manifestPublicKey) {
    return { ok: false, digest: null, error: 'MCP manifest path, signature, and public key are all required.' }
  }
  try {
    const manifestPath = path.resolve(config.manifestPath)
    const raw = await fs.readFile(manifestPath, 'utf8')
    const manifest = JSON.parse(raw)
    const canonical = canonicalMcpManifest(manifest)
    const digest = createHash('sha256').update(canonical).digest('hex')
    const signature = Buffer.from(config.manifestSignature, 'base64')
    const ok = verify(null, Buffer.from(canonical), config.manifestPublicKey, signature)
    if (!ok) return { ok: false, digest, error: 'MCP manifest signature verification failed.' }
    const expectedCommand = typeof manifest.command === 'string' ? manifest.command : null
    const expectedArgs = Array.isArray(manifest.args) ? manifest.args.map(String) : null
    if (expectedCommand && expectedCommand !== config.command) return { ok: false, digest, error: 'MCP manifest command does not match server config.' }
    if (expectedArgs && JSON.stringify(expectedArgs) !== JSON.stringify(config.args)) return { ok: false, digest, error: 'MCP manifest args do not match server config.' }
    const expectedSha = typeof manifest.commandSha256 === 'string' ? manifest.commandSha256.trim().toLowerCase() : null
    if (mcpSignatureRequired(config) && !/^[a-f0-9]{64}$/.test(expectedSha ?? '')) {
      return { ok: false, digest, error: 'Trusted or auto-connect MCP manifests must pin commandSha256.' }
    }
    if (expectedSha) {
      const commandPath = await resolveCommandPath(config.command, config.cwd || undefined, config.env?.PATH ?? process.env.PATH)
      if (!commandPath) return { ok: false, digest, error: 'MCP manifest pins a command hash but the command could not be resolved on disk.' }
      let actualSha: string
      try {
        actualSha = createHash('sha256').update(await fs.readFile(commandPath)).digest('hex')
      } catch (e: any) {
        return { ok: false, digest, error: `MCP manifest command hash could not be computed: ${String(e?.message || e)}` }
      }
      if (actualSha.toLowerCase() !== expectedSha) return { ok: false, digest, error: 'MCP manifest command hash does not match the on-disk binary.' }
    }
    const launchFiles = await verifyManifestLaunchFiles(config, manifest, digest)
    if (!launchFiles.ok) return launchFiles
    return { ok: true, digest, error: null }
  } catch (error: any) {
    return { ok: false, digest: null, error: `MCP manifest verification failed: ${String(error?.message || error)}` }
  }
}

export function mcpSignatureRequired(config: McpServerConfig): boolean {
  if (requireSignedMcp()) return true
  return config.autoConnect || config.approvalMode === 'trusted'
}

function requireSignedMcp(): boolean {
  const value = process.env[REQUIRE_SIGNED_MCP_ENV]?.toLowerCase()
  return value === '1' || value === 'true' || value === 'yes'
}

export function mcpSpawnSpec(config: McpServerConfig, options: McpSpawnSpecOptions = {}): McpSpawnSpec {
  const cwd = path.resolve(config.cwd || process.cwd())
  if (cwd === path.parse(cwd).root) {
    throw new Error('MCP server working directory cannot be a filesystem root.')
  }
  if (config.launchMode === 'docker_gateway') return dockerGatewaySpawnSpec(config, cwd)
  const sandboxAvailable = options.macSandboxAvailable ?? canUseMacSandbox()
  if (!sandboxAvailable) {
    if (!allowUnsandboxedMcp()) {
      throw new Error(`No supported MCP process sandbox is available on this platform. Use Docker Gateway or set ${UNSANDBOXED_MCP_ENV}=1 only for deliberate unsandboxed local development.`)
    }
    return { command: config.command, args: config.args, cwd, env: childEnv(config.env), gateway: null, sandboxed: false }
  }
  return {
    command: '/usr/bin/sandbox-exec',
    args: ['-p', macSandboxProfile(cwd, false, mcpSandboxReadRoots(config, cwd)), config.command, ...config.args],
    cwd,
    env: childEnv(config.env),
    gateway: null,
    sandboxed: true,
  }
}

function allowUnsandboxedMcp(): boolean {
  const value = process.env[UNSANDBOXED_MCP_ENV]?.toLowerCase()
  return value === '1' || value === 'true' || value === 'yes'
}

function dockerGatewaySpawnSpec(config: McpServerConfig, cwd: string): McpSpawnSpec {
  const gateway = normalizeDockerGateway(config.dockerGateway)
  if (!gateway?.image) throw new Error(`${config.name} needs a Docker image for gateway launch.`)
  const network = gateway.network ?? 'none'
  if (network === 'host') throw new Error(`${config.name} cannot use Docker host networking; choose none or bridge.`)
  if (!/^.+@sha256:[a-f0-9]{64}$/i.test(gateway.image)) {
    throw new Error(`${config.name} Docker image must be pinned by immutable sha256 digest (image@sha256:...).`)
  }
  const args = [
    'run',
    '--rm',
    '-i',
    '--network',
    network,
  ]
  if (gateway.readOnlyRootfs !== false) args.push('--read-only')
  for (const mount of gateway.mounts ?? []) args.push('--mount', dockerMountArg(mount))
  if (gateway.workdir) args.push('-w', gateway.workdir)
  const allowedEnv = pickEnv(config.env ?? {}, gateway.envAllowlist ?? [])
  for (const key of Object.keys(allowedEnv)) args.push('-e', key)
  args.push(gateway.image, config.command, ...config.args)
  return {
    command: 'docker',
    args,
    cwd,
    env: childEnv(allowedEnv),
    sandboxed: true,
    gateway: {
      kind: 'docker',
      image: gateway.image,
      network,
      readOnlyRootfs: gateway.readOnlyRootfs !== false,
      envAllowlist: gateway.envAllowlist ?? [],
      mountCount: gateway.mounts?.length ?? 0,
    },
  }
}

async function verifyManifestLaunchFiles(config: McpServerConfig, manifest: any, digest: string): Promise<ManifestVerificationResult> {
  const cwd = path.resolve(config.cwd || process.cwd())
  const rawEntries = Array.isArray(manifest.launchFiles) ? manifest.launchFiles : []
  const entries = new Map<string, { path: string; sha256: string }>()
  for (const raw of rawEntries) {
    const file = typeof raw?.path === 'string' ? raw.path.trim() : ''
    const sha256 = typeof raw?.sha256 === 'string' ? raw.sha256.trim().toLowerCase() : ''
    if (!file || !/^[a-f0-9]{64}$/.test(sha256)) {
      return { ok: false, digest, error: 'MCP manifest launchFiles entries require path and a 64-character sha256.' }
    }
    const resolved = path.resolve(cwd, file)
    let actual: string
    try { actual = createHash('sha256').update(await fs.readFile(resolved)).digest('hex') } catch (e: any) {
      return { ok: false, digest, error: `MCP manifest launch file could not be read: ${file} (${String(e?.message || e)})` }
    }
    if (actual !== sha256) return { ok: false, digest, error: `MCP manifest launch file hash does not match: ${file}.` }
    entries.set(resolved, { path: file, sha256 })
  }

  if (!mcpSignatureRequired(config)) return { ok: true, digest, error: null }
  const commandBase = path.basename(config.command).toLowerCase()
  const interpreter = /^(node|nodejs|python(?:3(?:\.\d+)?)?|bash|sh|zsh|ruby|deno|bun)(?:\.exe)?$/.test(commandBase)
  if (interpreter && config.args.includes('-m')) {
    return { ok: false, digest, error: 'Trusted MCP interpreter module launches (-m) are unsupported unless converted to an explicitly pinned script path.' }
  }
  for (const arg of config.args) {
    if (!arg || arg.startsWith('-')) continue
    const candidate = path.resolve(cwd, arg)
    let isFile = false
    try { isFile = (await fs.stat(candidate)).isFile() } catch { /* ordinary value, not a launch file */ }
    if (isFile && !entries.has(candidate)) {
      return { ok: false, digest, error: `Trusted or auto-connect MCP manifest must pin mutable launch file ${arg} in launchFiles.` }
    }
  }
  return { ok: true, digest, error: null }
}

function dockerMountArg(mount: McpDockerMount): string {
  const source = path.resolve(mount.source)
  const parts = [`type=bind`, `src=${source}`, `dst=${mount.target}`]
  if (mount.readOnly !== false) parts.push('readonly')
  return parts.join(',')
}

function pickEnv(source: Record<string, string>, allowlist: string[]): Record<string, string> {
  const picked: Record<string, string> = {}
  for (const key of allowlist) {
    if (Object.prototype.hasOwnProperty.call(source, key)) picked[key] = source[key]
  }
  return picked
}

function mcpSandboxReadRoots(config: McpServerConfig, cwd: string): string[] {
  const roots: string[] = []
  const add = (value: string | null | undefined) => {
    const raw = String(value || '').trim()
    if (!raw || (!path.isAbsolute(raw) && !raw.includes('/') && !raw.includes(path.sep))) return
    roots.push(path.dirname(path.isAbsolute(raw) ? raw : path.resolve(cwd, raw)))
  }
  add(config.command)
  for (const arg of config.args) add(arg)
  return roots
}

export function canonicalMcpManifest(value: unknown): string {
  return JSON.stringify(sortJson(value))
}

function sortJson(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(sortJson)
  if (!value || typeof value !== 'object') return value
  return Object.fromEntries(Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => [key, sortJson(item)]))
}

function trustedMcpAllowed(): boolean {
  const value = process.env[TRUSTED_MCP_ENV]?.toLowerCase()
  return value === '1' || value === 'true' || value === 'yes'
}

export async function commandProvenance(config: McpServerConfig): Promise<McpServerProvenance | null> {
  const spec = mcpSpawnSpec(config)
  const commandPath = await resolveCommandPath(spec.command, config.cwd || undefined, config.env?.PATH ?? process.env.PATH)
  if (!commandPath) return null
  const stat = await fs.stat(commandPath)
  if (!stat.isFile()) return null
  const bytes = await fs.readFile(commandPath)
  return {
    commandPath,
    commandSha256: createHash('sha256').update(bytes).digest('hex'),
    checkedAt: Date.now(),
    manifestDigest: config.manifestDigest ?? null,
    sandboxed: spec.sandboxed,
    gateway: spec.gateway,
  }
}

async function resolveCommandPath(command: string, cwd?: string, pathEnv?: string): Promise<string | null> {
  const raw = command.trim()
  if (!raw) return null
  if (raw.includes('/') || raw.includes(path.sep)) {
    const candidate = path.isAbsolute(raw) ? raw : path.resolve(cwd || process.cwd(), raw)
    return await isExecutableFile(candidate) ? candidate : null
  }
  for (const dir of ((pathEnv ?? process.env.PATH) || '').split(path.delimiter).filter(Boolean)) {
    const candidate = path.join(dir, raw)
    if (await isExecutableFile(candidate)) return candidate
  }
  return null
}

async function isExecutableFile(file: string): Promise<boolean> {
  try {
    const stat = await fs.stat(file)
    return stat.isFile()
  } catch {
    return false
  }
}

export function splitShellArgs(value: string): string[] {
  const args: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaping = false
  for (const char of value) {
    if (escaping) {
      current += char
      escaping = false
      continue
    }
    if (char === '\\') {
      escaping = true
      continue
    }
    if (quote) {
      if (char === quote) quote = null
      else current += char
      continue
    }
    if (char === '"' || char === "'") {
      quote = char
      continue
    }
    if (/\s/.test(char)) {
      if (current) {
        args.push(current)
        current = ''
      }
      continue
    }
    current += char
  }
  if (current) args.push(current)
  return args
}
