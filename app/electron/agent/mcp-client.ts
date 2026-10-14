import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { createHash, verify } from 'node:crypto'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { canUseMacSandbox, macSandboxProfile } from './sandbox'
import { redactSecretValues } from './secret-detection'
import { SECRET_MASK } from './security'
import type {
  McpPromptDefinition,
  McpDockerGatewayConfig,
  McpDockerMount,
  McpDockerNetworkMode,
  McpResourceDefinition,
  McpServerConfig,
  McpServerProvenance,
  McpServerRuntime,
  McpSnapshot,
  McpToolCallResult,
  McpToolDefinition,
} from './types'

const MCP_PROTOCOL_VERSION = '2025-06-18'
const REQUEST_TIMEOUT_MS = 20_000
const MAX_LOG_LINES = 80
// Cap *_list pagination so a server that returns a non-advancing or never-terminating
// nextCursor can't wedge connect() in an unbounded request loop.
const MAX_LIST_PAGES = 1000
// Cap the unframed stdout buffer: a server that never emits a newline must not grow it without bound.
const MAX_STDOUT_BUFFER_BYTES = 8 * 1024 * 1024
const TRUSTED_MCP_ENV = 'VONTRA_ALLOW_TRUSTED_MCP'
const REQUIRE_SIGNED_MCP_ENV = 'VONTRA_REQUIRE_SIGNED_MCP'
const UNSANDBOXED_MCP_ENV = 'VONTRA_ALLOW_UNSANDBOXED_MCP'

type JsonRpcId = string | number

interface PendingRequest {
  resolve: (value: any) => void
  reject: (error: Error) => void
  timer: ReturnType<typeof setTimeout>
}

interface ClientState {
  config: McpServerConfig
  runtime: McpServerRuntime
  proc: ChildProcessWithoutNullStreams | null
  buffer: string
  pending: Map<JsonRpcId, PendingRequest>
  nextId: number
  connectPromise: Promise<McpServerRuntime> | null
}

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

export class McpHost {
  private configs: McpServerConfig[] = []
  private clients = new Map<string, ClientState>()

  setConfigs(configs: McpServerConfig[]) {
    this.configs = configs.map(normalizeServerConfig)
    const ids = new Set(this.configs.map((config) => config.id))
    for (const [id, state] of this.clients.entries()) {
      if (!ids.has(id)) {
        this.stopState(state)
        this.clients.delete(id)
      }
    }
    for (const config of this.configs) {
      const state = this.clients.get(config.id)
      if (state) state.config = config
    }
  }

  snapshot(): McpSnapshot {
    const servers = this.configs.map((config) => ({
      ...redactConfig(config),
      runtime: this.getRuntime(config),
    }))
    return {
      servers,
      tools: servers.flatMap((server) => server.runtime.tools),
      resources: servers.flatMap((server) => server.runtime.resources),
      prompts: servers.flatMap((server) => server.runtime.prompts),
      updatedAt: Date.now(),
    }
  }

  toolDefinitions(): McpToolDefinition[] {
    return this.snapshot().tools
  }

  async connectEnabled() {
    await Promise.all(this.configs.filter((config) => config.enabled && config.autoConnect).map((config) => this.connect(config.id).catch(() => null)))
  }

  async connect(id: string): Promise<McpServerRuntime> {
    const config = this.getConfig(id)
    if (!config.enabled) throw new Error(`${config.name} is disabled.`)
    if (!config.command.trim()) throw new Error(`${config.name} needs a command.`)
    const verified = await verifyConfiguredManifest(config)
    if (mcpSignatureRequired(config) && !verified.ok) throw new Error(verified.error || `${config.name} requires a signed MCP manifest.`)
    if (verified.ok) {
      config.manifestVerifiedAt = Date.now()
      config.manifestDigest = verified.digest
    }
    const state = this.getState(config)
    if (state.runtime.status === 'connected') return state.runtime
    if (state.connectPromise) return state.connectPromise
    state.connectPromise = this.connectState(state).finally(() => { state.connectPromise = null })
    return state.connectPromise
  }

  async refresh(id: string): Promise<McpServerRuntime> {
    const state = this.clients.get(id)
    if (!state || state.runtime.status !== 'connected') return this.connect(id)
    await this.refreshTools(state)
    await this.refreshResources(state)
    await this.refreshPrompts(state)
    return state.runtime
  }

  async disconnect(id: string): Promise<McpServerRuntime> {
    const state = this.clients.get(id)
    if (!state) return this.defaultRuntime(id)
    this.stopState(state)
    state.runtime = {
      ...state.runtime,
      status: 'stopped',
      pid: null,
      tools: [],
      resources: [],
      prompts: [],
      lastError: null,
    }
    return state.runtime
  }

  async callTool(callName: string, args: Record<string, any>): Promise<McpToolCallResult> {
    const tool = this.toolDefinitions().find((candidate) => candidate.callName === callName)
    if (!tool) throw new Error(`MCP tool not found: ${callName}`)
    const state = await this.ensureConnected(tool.serverId)
    const result = await this.request(state, 'tools/call', { name: tool.name, arguments: args ?? {} }, 120_000)
    return formatToolResult(tool, result)
  }

  async shutdown() {
    for (const state of this.clients.values()) this.stopState(state)
    this.clients.clear()
  }

  private async ensureConnected(id: string): Promise<ClientState> {
    const state = this.clients.get(id)
    if (state?.runtime.status === 'connected') return state
    await this.connect(id)
    const next = this.clients.get(id)
    if (!next || next.runtime.status !== 'connected') throw new Error('MCP server is not connected.')
    return next
  }

  private getConfig(id: string): McpServerConfig {
    const config = this.configs.find((candidate) => candidate.id === id)
    if (!config) throw new Error('MCP server not found.')
    return config
  }

  private getState(config: McpServerConfig): ClientState {
    const existing = this.clients.get(config.id)
    if (existing) {
      existing.config = config
      return existing
    }
    const state: ClientState = {
      config,
      runtime: this.defaultRuntime(config.id),
      proc: null,
      buffer: '',
      pending: new Map(),
      nextId: 1,
      connectPromise: null,
    }
    this.clients.set(config.id, state)
    return state
  }

  private getRuntime(config: McpServerConfig): McpServerRuntime {
    const state = this.clients.get(config.id)
    return state?.runtime ?? this.defaultRuntime(config.id)
  }

  private defaultRuntime(id: string): McpServerRuntime {
    return {
      id,
      status: 'stopped',
      pid: null,
      provenance: null,
      serverInfo: null,
      capabilities: null,
      tools: [],
      resources: [],
      prompts: [],
      lastConnectedAt: null,
      lastError: null,
      logs: [],
    }
  }

  private async connectState(state: ClientState): Promise<McpServerRuntime> {
    this.stopState(state)
    state.runtime = {
      ...state.runtime,
      status: 'connecting',
      pid: null,
      tools: [],
      resources: [],
      prompts: [],
      lastError: null,
    }

    const config = state.config
    const provenance = await commandProvenance(config).catch(() => null)
    state.runtime = { ...state.runtime, provenance }

    try {
      const spawnSpec = mcpSpawnSpec(config)
      const proc = spawn(spawnSpec.command, spawnSpec.args, {
        cwd: spawnSpec.cwd,
        detached: detachedProcessGroup(),
        env: spawnSpec.env,
        stdio: 'pipe',
      })
      state.proc = proc
      state.runtime.pid = proc.pid ?? null

      proc.stdout.on('data', (chunk: Buffer) => this.handleStdout(state, chunk))
      proc.stderr.on('data', (chunk: Buffer) => this.appendLog(state, chunk.toString('utf8')))
      proc.on('error', (error) => {
        this.rejectAll(state, error)
        state.runtime = { ...state.runtime, status: 'error', pid: null, lastError: error.message }
      })
      proc.on('close', (code, signal) => {
        this.rejectAll(state, new Error(`${config.name} exited${code == null ? '' : ` with code ${code}`}${signal ? ` (${signal})` : ''}.`))
        if (state.runtime.status !== 'error') {
          state.runtime = {
            ...state.runtime,
            status: 'stopped',
            pid: null,
            tools: [],
            resources: [],
            prompts: [],
            lastError: code || signal ? `Exited${code == null ? '' : ` ${code}`}${signal ? ` (${signal})` : ''}` : state.runtime.lastError,
          }
        }
        state.proc = null
      })

      const initialized = await this.request(state, 'initialize', {
        protocolVersion: MCP_PROTOCOL_VERSION,
        capabilities: {},
        clientInfo: { name: 'TensorFold', version: '0.1.0' },
      })
      state.runtime = {
        ...state.runtime,
        capabilities: isRecord(initialized?.capabilities) ? initialized.capabilities : {},
        serverInfo: isRecord(initialized?.serverInfo) ? initialized.serverInfo as { name?: string; version?: string } : null,
      }
      this.notify(state, 'notifications/initialized')
      await this.refreshTools(state)
      await this.refreshResources(state)
      await this.refreshPrompts(state)
      state.runtime = {
        ...state.runtime,
        status: 'connected',
        pid: proc.pid ?? null,
        lastConnectedAt: Date.now(),
        lastError: null,
      }
      return state.runtime
    } catch (error: any) {
      const message = error?.message ? String(error.message) : String(error)
      state.runtime = { ...state.runtime, status: 'error', pid: null, tools: [], resources: [], prompts: [], lastError: message }
      this.stopState(state)
      throw new Error(message)
    }
  }

  private async refreshTools(state: ClientState) {
    if (!serverSupports(state, 'tools')) {
      state.runtime = { ...state.runtime, tools: [] }
      return
    }
    const tools: McpToolDefinition[] = []
    const usedCallNames = new Set<string>()
    const seenCursors = new Set<string>()
    let cursor: string | undefined
    let page = 0
    do {
      assertListPageBudget(state.config.name, 'tools/list', ++page, cursor, seenCursors)
      const result = await this.request(state, 'tools/list', cursor ? { cursor } : undefined)
      for (const tool of Array.isArray(result?.tools) ? result.tools : []) {
        const name = String(tool?.name ?? '').trim()
        if (!name) continue
        tools.push(normalizeTool(state.config, tool, name, tools.length, usedCallNames))
      }
      cursor = typeof result?.nextCursor === 'string' && result.nextCursor ? result.nextCursor : undefined
    } while (cursor)
    state.runtime = { ...state.runtime, tools, lastError: null }
  }

  private async refreshResources(state: ClientState) {
    if (!serverSupports(state, 'resources')) {
      state.runtime = { ...state.runtime, resources: [] }
      return
    }
    const resources: McpResourceDefinition[] = []
    const seenCursors = new Set<string>()
    let cursor: string | undefined
    let page = 0
    do {
      assertListPageBudget(state.config.name, 'resources/list', ++page, cursor, seenCursors)
      const result = await this.request(state, 'resources/list', cursor ? { cursor } : undefined)
      for (const resource of Array.isArray(result?.resources) ? result.resources : []) {
        const normalized = normalizeMcpResource(state.config, resource, resources.length)
        if (normalized) resources.push(normalized)
      }
      cursor = typeof result?.nextCursor === 'string' && result.nextCursor ? result.nextCursor : undefined
    } while (cursor)
    state.runtime = { ...state.runtime, resources, lastError: null }
  }

  private async refreshPrompts(state: ClientState) {
    if (!serverSupports(state, 'prompts')) {
      state.runtime = { ...state.runtime, prompts: [] }
      return
    }
    const prompts: McpPromptDefinition[] = []
    const seenCursors = new Set<string>()
    let cursor: string | undefined
    let page = 0
    do {
      assertListPageBudget(state.config.name, 'prompts/list', ++page, cursor, seenCursors)
      const result = await this.request(state, 'prompts/list', cursor ? { cursor } : undefined)
      for (const prompt of Array.isArray(result?.prompts) ? result.prompts : []) {
        const normalized = normalizeMcpPrompt(state.config, prompt, prompts.length)
        if (normalized) prompts.push(normalized)
      }
      cursor = typeof result?.nextCursor === 'string' && result.nextCursor ? result.nextCursor : undefined
    } while (cursor)
    state.runtime = { ...state.runtime, prompts, lastError: null }
  }

  private request(state: ClientState, method: string, params?: Record<string, unknown>, timeoutMs = REQUEST_TIMEOUT_MS): Promise<any> {
    if (!state.proc || !state.proc.stdin.writable) return Promise.reject(new Error(`${state.config.name} is not running.`))
    const id = state.nextId++
    const message: Record<string, unknown> = { jsonrpc: '2.0', id, method }
    if (params !== undefined) message.params = params
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        state.pending.delete(id)
        reject(new Error(`${method} timed out after ${Math.round(timeoutMs / 1000)}s.`))
      }, timeoutMs)
      state.pending.set(id, { resolve, reject, timer })
      state.proc!.stdin.write(`${JSON.stringify(message)}\n`, 'utf8')
    })
  }

  private notify(state: ClientState, method: string, params?: Record<string, unknown>) {
    if (!state.proc || !state.proc.stdin.writable) return
    const message: Record<string, unknown> = { jsonrpc: '2.0', method }
    if (params !== undefined) message.params = params
    state.proc.stdin.write(`${JSON.stringify(message)}\n`, 'utf8')
  }

  private handleStdout(state: ClientState, chunk: Buffer) {
    state.buffer += chunk.toString('utf8')
    let index = state.buffer.indexOf('\n')
    while (index >= 0) {
      const raw = state.buffer.slice(0, index).trim()
      state.buffer = state.buffer.slice(index + 1)
      if (raw) this.handleMessage(state, raw)
      index = state.buffer.indexOf('\n')
    }
    // A misbehaving server that never delimits a line would otherwise grow state.buffer without
    // bound; tear down the connection once a single unframed line crosses the cap.
    if (state.buffer.length > MAX_STDOUT_BUFFER_BYTES) {
      this.appendLog(state, `${state.config.name} exceeded ${MAX_STDOUT_BUFFER_BYTES} bytes of stdout with no line delimiter; disconnecting.`)
      state.runtime = { ...state.runtime, status: 'error', pid: null, lastError: 'MCP server sent an oversized line with no newline delimiter.' }
      this.rejectAll(state, new Error('MCP server sent an oversized line with no newline delimiter.'))
      this.stopState(state)
    }
  }

  private handleMessage(state: ClientState, raw: string) {
    let message: any
    try {
      message = JSON.parse(raw)
    } catch {
      this.appendLog(state, `Invalid MCP stdout from ${state.config.name}: ${raw.slice(0, 240)}`)
      return
    }

    if (message.id != null && (message.result !== undefined || message.error !== undefined)) {
      const pending = state.pending.get(message.id)
      if (!pending) return
      clearTimeout(pending.timer)
      state.pending.delete(message.id)
      if (message.error) pending.reject(new Error(message.error?.message || JSON.stringify(message.error)))
      else pending.resolve(message.result)
      return
    }

    if (message.method === 'notifications/tools/list_changed') {
      this.refreshTools(state).catch((error) => {
        state.runtime = { ...state.runtime, lastError: error?.message ? String(error.message) : String(error) }
      })
      return
    }

    if (message.method === 'notifications/resources/list_changed') {
      this.refreshResources(state).catch((error) => {
        state.runtime = { ...state.runtime, lastError: error?.message ? String(error.message) : String(error) }
      })
      return
    }

    if (message.method === 'notifications/prompts/list_changed') {
      this.refreshPrompts(state).catch((error) => {
        state.runtime = { ...state.runtime, lastError: error?.message ? String(error.message) : String(error) }
      })
      return
    }

    if (message.id != null && message.method) {
      this.sendUnsupportedResponse(state, message.id, message.method)
    }
  }

  private sendUnsupportedResponse(state: ClientState, id: JsonRpcId, method: string) {
    if (!state.proc || !state.proc.stdin.writable) return
    state.proc.stdin.write(`${JSON.stringify({
      jsonrpc: '2.0',
      id,
      error: { code: -32601, message: `TensorFold does not implement client method ${method}.` },
    })}\n`, 'utf8')
  }

  private appendLog(state: ClientState, text: string) {
    // MCP subprocess stderr can contain connection strings, tokens, and request payloads.
    // Redact before storing so the snapshot (exposed via mcp:snapshot) never leaks secrets.
    const redacted = redactSecretValues(text, SECRET_MASK).text
    const lines = redacted.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
    if (!lines.length) return
    state.runtime = {
      ...state.runtime,
      logs: [...state.runtime.logs, ...lines].slice(-MAX_LOG_LINES),
    }
  }

  private rejectAll(state: ClientState, error: Error) {
    for (const pending of state.pending.values()) {
      clearTimeout(pending.timer)
      pending.reject(error)
    }
    state.pending.clear()
  }

  private stopState(state: ClientState) {
    this.rejectAll(state, new Error(`${state.config.name} stopped.`))
    state.connectPromise = null
    if (state.proc) {
      state.proc.removeAllListeners()
      state.proc.stdout.removeAllListeners()
      state.proc.stderr.removeAllListeners()
      terminateChildProcess(state.proc, { processGroup: true })
    }
    state.proc = null
    state.buffer = ''
  }
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
    env: config.env ?? {},
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
    // Pin the actual binary, not just its name: a matching `command` string can still resolve
    // to a swapped executable on PATH. When the (signed) manifest pins commandSha256, hash the
    // resolved on-disk binary and require it to match before trusting the server.
    const expectedSha = typeof manifest.commandSha256 === 'string' ? manifest.commandSha256.trim().toLowerCase() : null
    if (mcpSignatureRequired(config) && !/^[a-f0-9]{64}$/.test(expectedSha ?? '')) {
      return { ok: false, digest, error: 'Trusted or auto-connect MCP manifests must pin commandSha256.' }
    }
    if (expectedSha) {
      // Hash the pinned server command itself (resolved through the CHILD's PATH, where the
      // swap would happen) — not the sandbox/gateway wrapper from mcpSpawnSpec.
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

function mcpSignatureRequired(config: McpServerConfig): boolean {
  if (requireSignedMcp()) return true
  return config.autoConnect || config.approvalMode === 'trusted'
}

function requireSignedMcp(): boolean {
  const value = process.env[REQUIRE_SIGNED_MCP_ENV]?.toLowerCase()
  return value === '1' || value === 'true' || value === 'yes'
}

export function mcpSpawnSpec(config: McpServerConfig, options: McpSpawnSpecOptions = {}): McpSpawnSpec {
  const cwd = path.resolve(config.cwd || process.cwd())
  // The macOS sandbox profile makes the cwd a WRITE root; a filesystem root like "/"
  // must never be configured as one, no matter what the config says.
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

async function commandProvenance(config: McpServerConfig): Promise<McpServerProvenance | null> {
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
  // Resolve through the SAME PATH the child will execvp under (config.env.PATH overrides
  // process.env.PATH in childEnv) — otherwise a manifest hash check could hash a benign binary
  // on the host PATH while the child runs a swapped one from config.env.PATH.
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

function normalizeTool(config: McpServerConfig, tool: any, name: string, index: number, used?: Set<string>): McpToolDefinition {
  const annotations = isRecord(tool?.annotations) ? tool.annotations : {}
  return {
    serverId: config.id,
    serverName: config.name,
    approvalMode: config.approvalMode,
    name,
    callName: uniqueCallName(config.id, name, index, used),
    title: typeof tool?.title === 'string' ? tool.title : undefined,
    description: typeof tool?.description === 'string' ? tool.description : undefined,
    inputSchema: isRecord(tool?.inputSchema) ? tool.inputSchema : { type: 'object', properties: {} },
    readOnlyHint: annotations.readOnlyHint === true,
    destructiveHint: annotations.destructiveHint === true,
    openWorldHint: annotations.openWorldHint === true,
  }
}

export function normalizeMcpResource(config: McpServerConfig, resource: any, index: number): McpResourceDefinition | null {
  const uri = typeof resource?.uri === 'string' ? resource.uri.trim() : ''
  if (!uri) return null
  const title = typeof resource?.title === 'string' && resource.title.trim() ? resource.title.trim() : undefined
  const rawName = typeof resource?.name === 'string' ? resource.name.trim() : ''
  const name = rawName || title || uri || `resource-${index + 1}`
  return {
    serverId: config.id,
    serverName: config.name,
    uri,
    name,
    title,
    description: typeof resource?.description === 'string' && resource.description.trim() ? resource.description.trim() : undefined,
    mimeType: typeof resource?.mimeType === 'string' && resource.mimeType.trim() ? resource.mimeType.trim() : undefined,
  }
}

export function normalizeMcpPrompt(config: McpServerConfig, prompt: any, _index: number): McpPromptDefinition | null {
  const name = typeof prompt?.name === 'string' ? prompt.name.trim() : ''
  if (!name) return null
  const args: any[] = Array.isArray(prompt?.arguments) ? prompt.arguments : []
  return {
    serverId: config.id,
    serverName: config.name,
    name,
    title: typeof prompt?.title === 'string' && prompt.title.trim() ? prompt.title.trim() : undefined,
    description: typeof prompt?.description === 'string' && prompt.description.trim() ? prompt.description.trim() : undefined,
    arguments: args.flatMap((arg) => {
      const argName = typeof arg?.name === 'string' ? arg.name.trim() : ''
      if (!argName) return []
      return [{
        name: argName,
        description: typeof arg?.description === 'string' && arg.description.trim() ? arg.description.trim() : undefined,
        required: arg?.required === true,
      }]
    }),
  }
}

function serverSupports(state: ClientState, key: 'tools' | 'resources' | 'prompts'): boolean {
  const capabilities = state.runtime.capabilities
  if (!capabilities) return false
  return isRecord(capabilities[key])
}

function uniqueCallName(serverId: string, toolName: string, index: number, used?: Set<string>): string {
  const base = base60(`mcp__${safeName(serverId)}__${safeName(toolName)}`, index)
  if (!used) return base
  // Disambiguate ANY collision (e.g. `foo.bar` and `foo bar` both sanitize to the same name), not
  // only length overflows — otherwise a request routes to the first match and shadows the rest.
  if (!used.has(base)) { used.add(base); return base }
  let n = 1
  let name = `${base.slice(0, 57)}_${n}`
  while (used.has(name)) { n++; name = `${base.slice(0, 57)}_${n}` }
  used.add(name)
  return name
}

function base60(name: string, index: number): string {
  return name.length <= 60 ? name : `${name.slice(0, 52)}_${index}`
}

function safeName(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]/g, '_').replace(/_+/g, '_').replace(/^_+|_+$/g, '') || 'tool'
}

function formatToolResult(tool: McpToolDefinition, result: any): McpToolCallResult {
  const chunks: string[] = []
  const content = Array.isArray(result?.content) ? result.content : []
  for (const item of content) {
    if (item?.type === 'text' && typeof item.text === 'string') chunks.push(item.text)
    else if (item?.type === 'resource_link' && item.uri) chunks.push(`[resource] ${item.name || item.uri}: ${item.uri}`)
    else if (item?.type === 'image') chunks.push(`[image] ${item.mimeType || 'image'} ${item.data ? `${String(item.data).length} bytes base64` : ''}`.trim())
    else if (item?.type) chunks.push(`[${item.type}] ${JSON.stringify(item).slice(0, 2000)}`)
  }
  if (isRecord(result?.structuredContent)) chunks.push(JSON.stringify(result.structuredContent, null, 2))
  const output = chunks.join('\n\n').trim() || JSON.stringify(result ?? {}, null, 2)
  return {
    ok: result?.isError !== true,
    output,
    title: tool.title || tool.name,
    meta: `${tool.serverName} MCP`,
  }
}

function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

// Guards a paginated *_list loop: throws if a server exceeds the page cap or repeats a cursor it
// already returned (a non-advancing cursor would otherwise loop forever). Call before each request
// with the cursor that is ABOUT to be sent (undefined on the first page). `page` is 1-based.
export function assertListPageBudget(
  serverName: string,
  method: string,
  page: number,
  cursor: string | undefined,
  seenCursors: Set<string>,
  maxPages: number = MAX_LIST_PAGES,
): void {
  if (page > maxPages) {
    throw new Error(`${serverName} ${method} exceeded ${maxPages} pages of pagination.`)
  }
  if (cursor !== undefined) {
    if (seenCursors.has(cursor)) {
      throw new Error(`${serverName} ${method} returned a repeated pagination cursor.`)
    }
    seenCursors.add(cursor)
  }
}

function redactConfig(config: McpServerConfig): McpServerConfig {
  return {
    ...config,
    env: Object.fromEntries(Object.keys(config.env ?? {}).map((key) => [key, '[set]'])),
  }
}
