import { promises as fs } from 'node:fs'
import path from 'node:path'
import type { AgentEvent, CapabilityTrustLevel } from './types'

// Plugin loading/execution runtime (v1).
//
// A plugin is a directory containing a `vontra-plugin.json` manifest whose `main` entry
// file exports a `register(ctx)` function. register() wires the plugin into the agent
// loop via the PluginContext: new tools (registerTool) plus pre/post tool-call hooks and
// session-lifecycle hooks. The host singleton is loaded during boot, and runAgent forwards
// its typed event stream through it after sending the same events to the renderer.

export interface PluginManifest {
  id: string
  name: string
  version: string
  main: string
  trustLevel: string
  permissions?: string[]
}

export type PreToolCallHookResult = { args?: any; block?: boolean; blockReason?: string } | void

export interface PluginToolExecutionContext {
  cwd: string
  threadId?: string
  signal?: AbortSignal
}

export interface PluginContext {
  registerTool(schema: object, handler: (args: any, context?: PluginToolExecutionContext) => Promise<any>): void
  onPreToolCall(fn: (toolName: string, args: any) => Promise<PreToolCallHookResult>): () => void
  onPostToolCall(fn: (toolName: string, args: any, result: any) => Promise<void>): () => void
  onSessionStart(fn: (threadId: string) => void): () => void
  onSessionEnd(fn: (threadId: string) => void): () => void
  onAgentEvent(fn: (event: AgentEvent) => void | Promise<void>): () => void
}

export interface RegisteredPluginTool {
  pluginId: string
  schema: object
  handler: (args: any, context?: PluginToolExecutionContext) => Promise<any>
}

export interface RegisteredPluginToolSchema {
  pluginId: string
  pluginName: string
  schema: object
}

interface LoadedPlugin {
  manifest: PluginManifest
  dir: string
  tools: Map<string, RegisteredPluginTool>
  preToolCallHooks: Array<(toolName: string, args: any) => Promise<PreToolCallHookResult>>
  postToolCallHooks: Array<(toolName: string, args: any, result: any) => Promise<void>>
  sessionStartHooks: Array<(threadId: string) => void>
  sessionEndHooks: Array<(threadId: string) => void>
  agentEventHooks: Array<(event: AgentEvent) => void | Promise<void>>
}

// Reuse the same trust-level vocabulary the capability registry already discloses
// (electron/agent/capability-registry.ts: 'unknown' | 'checksum' | 'signed') so a loaded
// plugin's manifest-declared trust level lines up with the rest of the capability surface.
const KNOWN_TRUST_LEVELS = new Set<CapabilityTrustLevel>(['unknown', 'checksum', 'signed'])

const REQUIRED_MANIFEST_FIELDS = ['id', 'name', 'version', 'main', 'trustLevel'] as const

function validateManifestShape(value: unknown): { ok: true; manifest: PluginManifest } | { ok: false; error: string } {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return { ok: false, error: 'vontra-plugin.json must contain a JSON object.' }
  }
  const record = value as Record<string, unknown>
  const missing = REQUIRED_MANIFEST_FIELDS.filter((field) => typeof record[field] !== 'string' || !record[field])
  if (missing.length) {
    return { ok: false, error: `vontra-plugin.json is missing required field(s): ${missing.join(', ')}` }
  }
  const trustLevel = String(record.trustLevel)
  if (!KNOWN_TRUST_LEVELS.has(trustLevel as CapabilityTrustLevel)) {
    return { ok: false, error: `vontra-plugin.json has an unknown trustLevel "${trustLevel}" (expected one of: ${[...KNOWN_TRUST_LEVELS].join(', ')})` }
  }
  const permissions = record.permissions
  if (permissions !== undefined && (!Array.isArray(permissions) || !permissions.every((p) => typeof p === 'string'))) {
    return { ok: false, error: 'vontra-plugin.json "permissions" must be an array of strings when present.' }
  }
  return {
    ok: true,
    manifest: {
      id: String(record.id),
      name: String(record.name),
      version: String(record.version),
      main: String(record.main),
      trustLevel,
      permissions: Array.isArray(permissions) ? (permissions as string[]) : undefined,
    },
  }
}

export interface LoadPluginResult {
  ok: boolean
  manifest?: PluginManifest
  error?: string
}

export class PluginRuntime {
  private plugins = new Map<string, LoadedPlugin>()
  private eventTail: Promise<void> = Promise.resolve()

  async loadPlugin(dir: string): Promise<LoadPluginResult> {
    const manifestPath = path.join(dir, 'vontra-plugin.json')
    let manifestRaw: string
    try {
      manifestRaw = await fs.readFile(manifestPath, 'utf8')
    } catch (e: any) {
      return { ok: false, error: `Could not read vontra-plugin.json at ${manifestPath}: ${e?.message ?? e}` }
    }

    let parsed: unknown
    try {
      parsed = JSON.parse(manifestRaw)
    } catch (e: any) {
      return { ok: false, error: `vontra-plugin.json is not valid JSON: ${e?.message ?? e}` }
    }

    const shape = validateManifestShape(parsed)
    if (!shape.ok) return { ok: false, error: shape.error }
    const manifest = shape.manifest

    // The entry file must stay INSIDE the plugin directory: an absolute or traversal
    // `main` would load arbitrary code from anywhere on disk under the plugin's trust.
    const pluginRoot = path.resolve(dir)
    const mainPath = path.resolve(dir, manifest.main)
    if (mainPath !== pluginRoot && !mainPath.startsWith(pluginRoot + path.sep)) {
      return { ok: false, error: `Plugin main entry must stay inside the plugin directory (declared "${manifest.main}").` }
    }
    try {
      await fs.access(mainPath)
    } catch {
      return { ok: false, error: `Plugin main entry file not found: ${mainPath} (declared as "${manifest.main}" in vontra-plugin.json)` }
    }

    // Dynamically load the entry file. This repo's Electron main/preload are bundled to
    // CJS (vite.config.ts rollup output: format 'cjs'), and the plugin's main file lives
    // at a path computed at runtime (not a static specifier a bundler could resolve) — so
    // require() is the right mechanism here, mirroring the existing lazy require()-by-name
    // pattern in electron/agent/ssh-connection.ts (createSsh2Client). Unlike that case this
    // is a require()-by-PATH, which Node resolves directly without any bundler involvement.
    let mod: any
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      mod = require(mainPath)
    } catch (e: any) {
      return { ok: false, error: `Failed to load plugin entry file ${mainPath}: ${e?.message ?? e}` }
    }

    const registerFn = typeof mod === 'function' ? mod : (mod?.register ?? mod?.default?.register ?? (typeof mod?.default === 'function' ? mod.default : undefined))
    if (typeof registerFn !== 'function') {
      return { ok: false, error: `Plugin entry file ${mainPath} does not export a register(ctx) function (default or named export).` }
    }

    const loaded: LoadedPlugin = {
      manifest,
      dir,
      tools: new Map(),
      preToolCallHooks: [],
      postToolCallHooks: [],
      sessionStartHooks: [],
      sessionEndHooks: [],
      agentEventHooks: [],
    }

    const ctx: PluginContext = {
      registerTool(schema, handler) {
        const raw = schema as { name?: unknown; function?: { name?: unknown } } | null | undefined
        // Canonical name: the nested OpenAI shape's function.name wins over the flat
        // legacy `name` field, so the advertised schema name and the executor's lookup
        // key can never diverge.
        const flatName = typeof raw?.name === 'string' && raw.name ? raw.name : ''
        const nestedName = typeof raw?.function?.name === 'string' && raw.function.name ? raw.function.name : ''
        const toolName = nestedName || flatName || `${manifest.id}:tool:${loaded.tools.size}`
        loaded.tools.set(toolName, { pluginId: manifest.id, schema, handler })
      },
      onPreToolCall(fn) {
        loaded.preToolCallHooks.push(fn)
        return () => {
          const idx = loaded.preToolCallHooks.indexOf(fn)
          if (idx >= 0) loaded.preToolCallHooks.splice(idx, 1)
        }
      },
      onPostToolCall(fn) {
        loaded.postToolCallHooks.push(fn)
        return () => {
          const idx = loaded.postToolCallHooks.indexOf(fn)
          if (idx >= 0) loaded.postToolCallHooks.splice(idx, 1)
        }
      },
      onSessionStart(fn) {
        loaded.sessionStartHooks.push(fn)
        return () => {
          const idx = loaded.sessionStartHooks.indexOf(fn)
          if (idx >= 0) loaded.sessionStartHooks.splice(idx, 1)
        }
      },
      onSessionEnd(fn) {
        loaded.sessionEndHooks.push(fn)
        return () => {
          const idx = loaded.sessionEndHooks.indexOf(fn)
          if (idx >= 0) loaded.sessionEndHooks.splice(idx, 1)
        }
      },
      onAgentEvent(fn) {
        loaded.agentEventHooks.push(fn)
        return () => {
          const idx = loaded.agentEventHooks.indexOf(fn)
          if (idx >= 0) loaded.agentEventHooks.splice(idx, 1)
        }
      },
    }

    // A broken/throwing plugin must not crash the host — isolate register() in try/catch
    // and report the failure instead of propagating it.
    try {
      await registerFn(ctx)
    } catch (e: any) {
      return { ok: false, error: `Plugin ${manifest.id} register() threw: ${e?.message ?? e}` }
    }

    this.plugins.set(manifest.id, loaded)
    return { ok: true, manifest }
  }

  unloadPlugin(id: string): void {
    this.plugins.delete(id)
  }

  clear(): void {
    this.plugins.clear()
  }

  listLoaded(): PluginManifest[] {
    return [...this.plugins.values()].map((p) => p.manifest)
  }

  getRegisteredTool(name: string): RegisteredPluginTool | undefined {
    for (const plugin of this.plugins.values()) {
      const tool = plugin.tools.get(name)
      if (tool) return tool
    }
    return undefined
  }

  listRegisteredTools(): RegisteredPluginTool[] {
    const out: RegisteredPluginTool[] = []
    for (const plugin of this.plugins.values()) out.push(...plugin.tools.values())
    return out
  }

  listToolSchemaEntries(): RegisteredPluginToolSchema[] {
    const out: RegisteredPluginToolSchema[] = []
    for (const plugin of this.plugins.values()) {
      for (const [name, tool] of plugin.tools) {
        const raw = tool.schema as any
        if (raw?.type === 'function' && raw?.function?.name) {
          out.push({ pluginId: plugin.manifest.id, pluginName: plugin.manifest.name, schema: raw })
          continue
        }
        out.push({
          pluginId: plugin.manifest.id,
          pluginName: plugin.manifest.name,
          schema: {
            type: 'function',
            function: {
              name,
              description: typeof raw?.description === 'string' ? raw.description : `Plugin tool from ${plugin.manifest.name}.`,
              parameters: raw?.parameters && typeof raw.parameters === 'object'
                ? raw.parameters
                : { type: 'object', properties: {} },
            },
          },
        })
      }
    }
    return out
  }

  listToolSchemas(): object[] {
    return this.listToolSchemaEntries().map((entry) => entry.schema)
  }

  // Runs every loaded plugin's onPreToolCall hooks, in plugin-registration order (and within
  // a plugin, hook-registration order). Args rewritten by one hook are threaded into the next.
  // Any plugin's hook can block the call; once blocked, remaining hooks still run (so multiple
  // plugins can each observe the call) but the block + its reason wins. A throwing hook is
  // caught and ignored — it must never stop other plugins' hooks from running.
  async runPreToolCall(toolName: string, args: any): Promise<{ args: any; blocked: boolean; blockReason?: string }> {
    let currentArgs = args
    let blocked = false
    let blockReason: string | undefined
    for (const plugin of this.plugins.values()) {
      for (const hook of [...plugin.preToolCallHooks]) {
        try {
          const result = await hook(toolName, currentArgs)
          if (!result) continue
          if (result.args !== undefined) currentArgs = result.args
          if (result.block) {
            blocked = true
            blockReason = result.blockReason ?? blockReason
          }
        } catch (e) {
          // Per-plugin isolation: log and continue, never let one plugin's hook break others.
          console.error(`[plugin-runtime] onPreToolCall hook in plugin "${plugin.manifest.id}" threw:`, e)
        }
      }
    }
    return { args: currentArgs, blocked, blockReason }
  }

  // Runs every loaded plugin's onPostToolCall hooks, in registration order. Observation-only
  // (cannot mutate the result in v1). A throwing hook is caught and ignored.
  async runPostToolCall(toolName: string, args: any, result: any): Promise<void> {
    for (const plugin of this.plugins.values()) {
      for (const hook of [...plugin.postToolCallHooks]) {
        try {
          await hook(toolName, args, result)
        } catch (e) {
          console.error(`[plugin-runtime] onPostToolCall hook in plugin "${plugin.manifest.id}" threw:`, e)
        }
      }
    }
  }

  emitSessionStart(threadId: string): void {
    for (const plugin of this.plugins.values()) {
      for (const hook of [...plugin.sessionStartHooks]) {
        try {
          hook(threadId)
        } catch (e) {
          console.error(`[plugin-runtime] onSessionStart hook in plugin "${plugin.manifest.id}" threw:`, e)
        }
      }
    }
  }

  emitSessionEnd(threadId: string): void {
    for (const plugin of this.plugins.values()) {
      for (const hook of [...plugin.sessionEndHooks]) {
        try {
          hook(threadId)
        } catch (e) {
          console.error(`[plugin-runtime] onSessionEnd hook in plugin "${plugin.manifest.id}" threw:`, e)
        }
      }
    }
  }

  queueSessionEnd(threadId: string): Promise<void> {
    const dispatch = this.eventTail.catch(() => {}).then(() => this.emitSessionEnd(threadId))
    this.eventTail = dispatch.catch(() => {})
    return dispatch
  }

  emitAgentEvent(event: AgentEvent): Promise<void> {
    const dispatch = this.eventTail.catch(() => {}).then(async () => {
      for (const plugin of this.plugins.values()) {
        for (const hook of [...plugin.agentEventHooks]) {
          try {
            await hook(event)
          } catch (e) {
            console.error(`[plugin-runtime] onAgentEvent hook in plugin "${plugin.manifest.id}" threw:`, e)
          }
        }
      }
    })
    this.eventTail = dispatch.catch(() => {})
    return dispatch
  }
}

// One process-wide host runtime. Tests and embedders can still pass a private runtime
// through RunDeps, while every normal host run shares the plugins loaded at app boot.
export const hostPluginRuntime = new PluginRuntime()
