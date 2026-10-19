import { StringDecoder } from 'node:string_decoder'
import { isMcpEnvelope, isRecord } from './mcp-envelope'
import { normalizeTool, normalizeMcpResource, normalizeMcpPrompt, formatToolResult } from './mcp-catalog'
import { normalizeServerConfig, verifyConfiguredManifest, mcpSignatureRequired, mcpSpawnSpec, commandProvenance, mcpSpawnFingerprint } from './mcp-process'
export { normalizeMcpResource, normalizeMcpPrompt } from './mcp-catalog'
export { normalizeServerConfig, createMcpServer, verifyConfiguredManifest, mcpSpawnSpec, canonicalMcpManifest, splitShellArgs, mcpSpawnFingerprint } from './mcp-process'
export type { McpSpawnSpec, McpSpawnSpecOptions } from './mcp-process'
import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process'
import { detachedProcessGroup, terminateChildProcess } from './child-process-lifecycle'
import { redactSecretValues } from './secret-detection'
import { SECRET_MASK } from './security'
import type {
  McpPromptDefinition,
  McpResourceDefinition,
  McpServerConfig,
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
// Bound each complete frame and any unfinished stdout frame before JSON parsing.
const MAX_STDOUT_BUFFER_BYTES = 8 * 1024 * 1024

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
  decoder: StringDecoder
  pending: Map<JsonRpcId, PendingRequest>
  nextId: number
  generation: number
  connectPromise: Promise<McpServerRuntime> | null
}

export class McpHost {
  private configs: McpServerConfig[] = []
  private clients = new Map<string, ClientState>()

  setConfigs(configs: McpServerConfig[]) {
    this.configs = configs.map(normalizeServerConfig)
    const nextConfigs = new Map(this.configs.map((config) => [config.id, config]))
    for (const [id, state] of this.clients.entries()) {
      const next = nextConfigs.get(id)
      if (!next || mcpSpawnFingerprint(next) !== mcpSpawnFingerprint(state.config)) {
        this.stopState(state)
        this.clients.delete(id)
      } else {
        state.config = next
      }
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

  async connect(id: string, expectedFingerprint?: string): Promise<McpServerRuntime> {
    const config = this.getConfig(id)
    const fingerprint = mcpSpawnFingerprint(config)
    if (expectedFingerprint && fingerprint !== expectedFingerprint) throw new Error('MCP server configuration changed before launch.')
    if (!config.enabled) throw new Error(`${config.name} is disabled.`)
    if (!config.command.trim()) throw new Error(`${config.name} needs a command.`)
    const verified = await verifyConfiguredManifest(config)
    const current = this.assertConfigCurrent(id, fingerprint)
    if (mcpSignatureRequired(current) && !verified.ok) throw new Error(verified.error || `${current.name} requires a signed MCP manifest.`)
    if (verified.ok) {
      current.manifestVerifiedAt = Date.now()
      current.manifestDigest = verified.digest
    }
    const state = this.getState(current)
    if (state.runtime.status === 'connected') return state.runtime
    if (state.connectPromise) return state.connectPromise
    const pending = this.connectState(state, fingerprint).finally(() => {
      if (state.connectPromise === pending) state.connectPromise = null
    })
    state.connectPromise = pending
    return pending
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
      decoder: new StringDecoder('utf8'),
      pending: new Map(),
      nextId: 1,
      generation: 0,
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

  private assertConfigCurrent(id: string, fingerprint: string): McpServerConfig {
    const current = this.configs.find((config) => config.id === id)
    if (!current || mcpSpawnFingerprint(current) !== fingerprint) throw new Error('MCP server configuration changed before launch.')
    return current
  }

  private assertLaunchCurrent(state: ClientState, fingerprint: string, generation: number) {
    if (this.clients.get(state.config.id) !== state || state.generation !== generation || state.runtime.status !== 'connecting') {
      throw new Error('MCP server launch was cancelled or its configuration changed.')
    }
    this.assertConfigCurrent(state.config.id, fingerprint)
  }

  private async connectState(state: ClientState, fingerprint: string): Promise<McpServerRuntime> {
    this.stopState(state)
    const generation = state.generation
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
    try {
      const provenance = await commandProvenance(config).catch(() => null)
      this.assertLaunchCurrent(state, fingerprint, generation)
      state.runtime = { ...state.runtime, provenance }
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
      this.assertLaunchCurrent(state, fingerprint, generation)
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
      if (state.generation === generation) {
        state.runtime = { ...state.runtime, status: 'error', pid: null, tools: [], resources: [], prompts: [], lastError: message }
        this.stopState(state)
      }
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
    state.buffer += state.decoder.write(chunk)
    let index = state.buffer.indexOf('\n')
    while (index >= 0) {
      const frame = state.buffer.slice(0, index)
      if (Buffer.byteLength(frame, 'utf8') > MAX_STDOUT_BUFFER_BYTES) { this.rejectOversizedFrame(state); return }
      const raw = frame.trim()
      state.buffer = state.buffer.slice(index + 1)
      if (raw) this.handleMessage(state, raw)
      index = state.buffer.indexOf('\n')
    }
    if (Buffer.byteLength(state.buffer, 'utf8') > MAX_STDOUT_BUFFER_BYTES) this.rejectOversizedFrame(state)
  }

  private rejectOversizedFrame(state: ClientState): void {
    const message = 'MCP server sent an oversized stdout frame.'
    this.appendLog(state, `${state.config.name} exceeded the ${MAX_STDOUT_BUFFER_BYTES} byte limit; disconnecting.`)
    state.runtime = { ...state.runtime, status: 'error', pid: null, lastError: message }
    this.rejectAll(state, new Error(message))
    this.stopState(state)
  }

  private handleMessage(state: ClientState, raw: string) {
    let message: unknown
    try {
      message = JSON.parse(raw)
    } catch {
      this.appendLog(state, `Invalid MCP stdout from ${state.config.name}: ${raw.slice(0, 240)}`)
      return
    }

    if (!isMcpEnvelope(message)) {
      this.appendLog(state, `Invalid MCP message envelope from ${state.config.name}.`)
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
    state.generation++
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
    state.decoder = new StringDecoder('utf8')
  }
}

function serverSupports(state: ClientState, key: 'tools' | 'resources' | 'prompts'): boolean {
  const capabilities = state.runtime.capabilities
  if (!capabilities) return false
  return isRecord(capabilities[key])
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
