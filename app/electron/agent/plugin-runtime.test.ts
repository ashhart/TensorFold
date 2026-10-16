// Run: jiti electron/agent/plugin-runtime.test.ts
import assert from 'node:assert/strict'
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { PluginRuntime, type PluginManifest } from './plugin-runtime.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

// Builds an isolated fixture plugin directory: vontra-plugin.json + an entry .cjs
// file. `entrySource` is the raw body of the entry file (module.exports.register = ...).
// Mirrors the mkdtempSync isolated-fixture pattern used by electron/store-settings.test.ts.
function makePluginDir(opts: {
  manifest?: Partial<PluginManifest> | null
  manifestRaw?: string
  entrySource?: string
  mainFile?: string
  omitEntryFile?: boolean
}): string {
  const dir = mkdtempSync(path.join(os.tmpdir(), 'vontra-plugin-'))
  const mainFile = opts.mainFile ?? 'index.cjs'
  if (opts.manifestRaw !== undefined) {
    writeFileSync(path.join(dir, 'vontra-plugin.json'), opts.manifestRaw, 'utf8')
  } else if (opts.manifest !== null) {
    const manifest: PluginManifest = {
      id: 'fixture-plugin',
      name: 'Fixture Plugin',
      version: '1.0.0',
      main: mainFile,
      trustLevel: 'unknown',
      ...opts.manifest,
    }
    writeFileSync(path.join(dir, 'vontra-plugin.json'), JSON.stringify(manifest, null, 2), 'utf8')
  }
  if (!opts.omitEntryFile) {
    mkdirSync(dir, { recursive: true })
    writeFileSync(path.join(dir, mainFile), opts.entrySource ?? 'module.exports.register = function register() {}\n', 'utf8')
  }
  return dir
}

console.log('plugin-runtime tests:')

// --- loadPlugin: valid fixture plugin registers a tool + pre/post hooks ---
{
  const dir = makePluginDir({
    manifest: { id: 'tool-plugin', name: 'Tool Plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.registerTool({ name: 'plugin_echo', description: 'Echoes input' }, async (args) => ({ echoed: args }))
        ctx.onPreToolCall(async (toolName, args) => { ctx.__preCalls = (ctx.__preCalls || 0) + 1 })
        ctx.onPostToolCall(async (toolName, args, result) => { ctx.__postCalls = (ctx.__postCalls || 0) + 1 })
      }
    `,
  })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)
  assert.equal(result.manifest?.id, 'tool-plugin')
  assert.equal(result.manifest?.name, 'Tool Plugin')
  assert.deepEqual(runtime.listLoaded().map((m) => m.id), ['tool-plugin'])

  // The registered tool is callable through whatever registry the loader stores it in.
  const tool = runtime.getRegisteredTool('plugin_echo')
  assert.ok(tool, 'registered tool is retrievable by name')
  const toolResult = await tool!.handler({ x: 1 })
  assert.deepEqual(toolResult, { echoed: { x: 1 } })

  // The pre/post hooks registered by the plugin actually run.
  const pre = await runtime.runPreToolCall('plugin_echo', { x: 1 })
  assert.equal(pre.blocked, false)
  const post = await runtime.runPostToolCall('plugin_echo', { x: 1 }, { ok: true, output: 'done' })
  assert.equal(post, undefined) // post is fire-and-forget / void by spec

  ok('loadPlugin: valid fixture registers a tool + pre/post hooks, all invocable')
}

// --- loadPlugin: missing manifest file ---
{
  const dir = makePluginDir({ manifest: null })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, false)
  assert.ok(result.error && result.error.length > 0, 'clear error message')
  assert.equal(result.manifest, undefined)
  assert.deepEqual(runtime.listLoaded(), [])
  ok('loadPlugin: missing vontra-plugin.json -> ok:false with a clear error, does not throw')
}

// --- loadPlugin: malformed manifest (invalid JSON) ---
{
  const dir = makePluginDir({ manifestRaw: '{ not valid json ]]]' })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, false)
  assert.ok(result.error && /json|parse/i.test(result.error), `expected a JSON-parse-flavoured error, got: ${result.error}`)
  ok('loadPlugin: malformed (invalid JSON) manifest -> ok:false, clear error, does not throw')
}

// --- loadPlugin: manifest missing required fields ---
{
  const dir = makePluginDir({ manifestRaw: JSON.stringify({ id: 'incomplete' }) })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, false)
  assert.ok(result.error && result.error.length > 0)
  ok('loadPlugin: manifest missing required fields -> ok:false, clear error')
}

// --- loadPlugin: manifest references a main file that does not exist ---
{
  const dir = makePluginDir({ manifest: { id: 'ghost-main' }, omitEntryFile: true })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, false)
  assert.ok(result.error && /main|not found|exist/i.test(result.error), `expected a missing-main-file error, got: ${result.error}`)
  ok('loadPlugin: manifest main file missing on disk -> ok:false, clear error')
}

// --- loadPlugin: register() throws ---
{
  const dir = makePluginDir({
    manifest: { id: 'throwing-plugin' },
    entrySource: `module.exports.register = function register() { throw new Error('boom from plugin') }`,
  })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, false)
  assert.ok(result.error && /boom from plugin/.test(result.error), `expected the plugin's thrown message surfaced, got: ${result.error}`)
  assert.deepEqual(runtime.listLoaded(), [], 'a throwing plugin is not left registered as loaded')
  ok('loadPlugin: register() throws -> ok:false, does not crash the host, not left loaded')
}

// --- loadPlugin: a throwing plugin does not affect another, separately-loaded plugin ---
{
  const badDir = makePluginDir({
    manifest: { id: 'bad-plugin' },
    entrySource: `module.exports.register = function register() { throw new Error('bad plugin exploded') }`,
  })
  const goodDir = makePluginDir({
    manifest: { id: 'good-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.registerTool({ name: 'good_tool' }, async () => 'good result')
      }
    `,
  })
  const runtime = new PluginRuntime()
  const badResult = await runtime.loadPlugin(badDir)
  const goodResult = await runtime.loadPlugin(goodDir)
  assert.equal(badResult.ok, false)
  assert.equal(goodResult.ok, true, goodResult.error)
  assert.deepEqual(runtime.listLoaded().map((m) => m.id), ['good-plugin'])
  assert.ok(runtime.getRegisteredTool('good_tool'), 'the good plugin still registered its tool')
  ok('loadPlugin: a broken plugin does not crash the host or block a subsequent good plugin from loading')
}

// --- runPreToolCall: a hook blocks a call ---
{
  const dir = makePluginDir({
    manifest: { id: 'blocker-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPreToolCall(async (toolName, args) => {
          if (toolName === 'dangerous_tool') return { block: true, blockReason: 'Blocked by policy plugin' }
        })
      }
    `,
  })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)

  const blocked = await runtime.runPreToolCall('dangerous_tool', { x: 1 })
  assert.equal(blocked.blocked, true)
  assert.equal(blocked.blockReason, 'Blocked by policy plugin')

  const allowed = await runtime.runPreToolCall('safe_tool', { x: 1 })
  assert.equal(allowed.blocked, false)
  assert.equal(allowed.blockReason, undefined)
  ok('runPreToolCall: a hook can block a call, returning blocked:true with the reason')
}

// --- runPreToolCall: a hook rewrites args ---
{
  const dir = makePluginDir({
    manifest: { id: 'rewriter-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPreToolCall(async (toolName, args) => {
          return { args: { ...args, rewritten: true } }
        })
      }
    `,
  })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)

  const out = await runtime.runPreToolCall('some_tool', { original: 1 })
  assert.equal(out.blocked, false)
  assert.deepEqual(out.args, { original: 1, rewritten: true })
  ok('runPreToolCall: a hook can rewrite args, returned in the result')
}

// --- multiple plugins' pre hooks all run, in registration order, args threaded through ---
{
  const dirA = makePluginDir({
    manifest: { id: 'order-plugin-a' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPreToolCall(async (toolName, args) => {
          globalThis.__pluginOrderLog = globalThis.__pluginOrderLog || []
          globalThis.__pluginOrderLog.push('a')
          return { args: { ...args, seenByA: true } }
        })
      }
    `,
  })
  const dirB = makePluginDir({
    manifest: { id: 'order-plugin-b' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPreToolCall(async (toolName, args) => {
          globalThis.__pluginOrderLog = globalThis.__pluginOrderLog || []
          globalThis.__pluginOrderLog.push('b')
          return { args: { ...args, seenByB: true } }
        })
      }
    `,
  })
  ;(globalThis as any).__pluginOrderLog = []
  const runtime = new PluginRuntime()
  const resA = await runtime.loadPlugin(dirA)
  const resB = await runtime.loadPlugin(dirB)
  assert.equal(resA.ok, true, resA.error)
  assert.equal(resB.ok, true, resB.error)

  const out = await runtime.runPreToolCall('multi_tool', { start: true })
  assert.deepEqual((globalThis as any).__pluginOrderLog, ['a', 'b'], 'hooks run in registration (load) order')
  assert.deepEqual(out.args, { start: true, seenByA: true, seenByB: true }, 'each hook receives the prior hook\'s rewritten args')
  assert.equal(out.blocked, false)
  ok('runPreToolCall: multiple plugins\' hooks all run for the same event, in registration order')
}

// --- multiple plugins' post hooks all run for the same event ---
{
  const dirA = makePluginDir({
    manifest: { id: 'post-plugin-a' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPostToolCall(async (toolName, args, result) => {
          globalThis.__postLog = globalThis.__postLog || []
          globalThis.__postLog.push('a:' + toolName)
        })
      }
    `,
  })
  const dirB = makePluginDir({
    manifest: { id: 'post-plugin-b' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPostToolCall(async (toolName, args, result) => {
          globalThis.__postLog = globalThis.__postLog || []
          globalThis.__postLog.push('b:' + toolName)
        })
      }
    `,
  })
  ;(globalThis as any).__postLog = []
  const runtime = new PluginRuntime()
  await runtime.loadPlugin(dirA)
  await runtime.loadPlugin(dirB)
  await runtime.runPostToolCall('observed_tool', {}, { ok: true })
  assert.deepEqual((globalThis as any).__postLog, ['a:observed_tool', 'b:observed_tool'], 'both plugins observed the same post-call event, in order')
  ok('runPostToolCall: multiple plugins\' postToolCall hooks all run for the same event, in registration order')
}

// --- a postToolCall hook that throws does not affect other plugins' postToolCall hooks ---
{
  const dirThrows = makePluginDir({
    manifest: { id: 'post-throws-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPostToolCall(async () => { throw new Error('post hook exploded') })
      }
    `,
  })
  const dirObserves = makePluginDir({
    manifest: { id: 'post-observes-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPostToolCall(async (toolName) => {
          globalThis.__postSurvivorLog = globalThis.__postSurvivorLog || []
          globalThis.__postSurvivorLog.push(toolName)
        })
      }
    `,
  })
  ;(globalThis as any).__postSurvivorLog = []
  const runtime = new PluginRuntime()
  await runtime.loadPlugin(dirThrows)
  await runtime.loadPlugin(dirObserves)
  // Must not throw/reject even though one plugin's post hook throws.
  await runtime.runPostToolCall('resilient_tool', {}, { ok: true })
  assert.deepEqual((globalThis as any).__postSurvivorLog, ['resilient_tool'], 'the surviving plugin\'s post hook still ran')
  ok('runPostToolCall: one plugin\'s thrown error does not stop other plugins\' postToolCall hooks from running')
}

// --- a preToolCall hook that throws does not affect other plugins' preToolCall hooks ---
{
  const dirThrows = makePluginDir({
    manifest: { id: 'pre-throws-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPreToolCall(async () => { throw new Error('pre hook exploded') })
      }
    `,
  })
  const dirBlocks = makePluginDir({
    manifest: { id: 'pre-blocks-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onPreToolCall(async () => ({ block: true, blockReason: 'still blocked despite sibling failure' }))
      }
    `,
  })
  const runtime = new PluginRuntime()
  await runtime.loadPlugin(dirThrows)
  await runtime.loadPlugin(dirBlocks)
  const out = await runtime.runPreToolCall('still_runs_tool', {})
  assert.equal(out.blocked, true)
  assert.equal(out.blockReason, 'still blocked despite sibling failure')
  ok('runPreToolCall: one plugin\'s thrown error does not stop other plugins\' preToolCall hooks from running')
}

// --- unloadPlugin removes that plugin's hooks from subsequent runs ---
{
  const dir = makePluginDir({
    manifest: { id: 'unload-me' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.registerTool({ name: 'unload_tool' }, async () => 'still here?')
        ctx.onPreToolCall(async () => ({ block: true, blockReason: 'blocked while loaded' }))
        ctx.onPostToolCall(async (toolName) => {
          globalThis.__unloadPostLog = globalThis.__unloadPostLog || []
          globalThis.__unloadPostLog.push(toolName)
        })
      }
    `,
  })
  ;(globalThis as any).__unloadPostLog = []
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)
  assert.deepEqual(runtime.listLoaded().map((m) => m.id), ['unload-me'])

  // Before unload: the hook blocks, the tool is registered, post hook observes.
  const before = await runtime.runPreToolCall('any_tool', {})
  assert.equal(before.blocked, true)
  assert.ok(runtime.getRegisteredTool('unload_tool'))
  await runtime.runPostToolCall('any_tool', {}, { ok: true })
  assert.deepEqual((globalThis as any).__unloadPostLog, ['any_tool'])

  runtime.unloadPlugin('unload-me')
  assert.deepEqual(runtime.listLoaded(), [], 'unloaded plugin no longer reported as loaded')

  // After unload: the hook no longer blocks, the tool is gone, post hook no longer observes.
  const after = await runtime.runPreToolCall('any_tool', {})
  assert.equal(after.blocked, false, 'pre-call hook from the unloaded plugin no longer runs')
  assert.equal(runtime.getRegisteredTool('unload_tool'), undefined, 'tool from the unloaded plugin is gone')
  await runtime.runPostToolCall('any_tool', {}, { ok: true })
  assert.deepEqual((globalThis as any).__unloadPostLog, ['any_tool'], 'post hook from the unloaded plugin no longer fires (log unchanged)')

  ok('unloadPlugin: removes that plugin\'s registered tools + hooks from subsequent runs')
}

// --- onSessionStart / onSessionEnd hooks fire on emitSessionStart / emitSessionEnd ---
{
  const dir = makePluginDir({
    manifest: { id: 'session-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.onSessionStart((threadId) => { globalThis.__sessionLog = globalThis.__sessionLog || []; globalThis.__sessionLog.push('start:' + threadId) })
        ctx.onSessionEnd((threadId) => { globalThis.__sessionLog = globalThis.__sessionLog || []; globalThis.__sessionLog.push('end:' + threadId) })
      }
    `,
  })
  ;(globalThis as any).__sessionLog = []
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)
  runtime.emitSessionStart('thread-123')
  runtime.emitSessionEnd('thread-123')
  assert.deepEqual((globalThis as any).__sessionLog, ['start:thread-123', 'end:thread-123'])
  ok('emitSessionStart / emitSessionEnd: invoke the plugin\'s registered session hooks')
}

// --- onPreToolCall / onPostToolCall / onSessionStart / onSessionEnd return an unsubscribe fn ---
{
  const dir = makePluginDir({
    manifest: { id: 'unsubscribe-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        const unsub = ctx.onPreToolCall(async () => ({ block: true, blockReason: 'should stop after unsubscribe' }))
        ctx.__unsub = unsub
        // Unsubscribe immediately inside register() itself to prove the returned fn works.
        unsub()
      }
    `,
  })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)
  const out = await runtime.runPreToolCall('any_tool', {})
  assert.equal(out.blocked, false, 'a hook that unsubscribed itself during register() never fires')
  ok('onPreToolCall returns a working unsubscribe function')
}

// --- registered tools become model schemas and typed host events stay ordered ---
{
  const dir = makePluginDir({
    manifest: { id: 'event-plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.registerTool({ name: 'plugin_lookup', description: 'Look up a plugin value.', parameters: { type: 'object', properties: {} } }, async () => 'ok')
        ctx.onAgentEvent(async (event) => {
          globalThis.__agentEventLog = globalThis.__agentEventLog || []
          globalThis.__agentEventLog.push(event.type + ':' + (event.phase || ''))
        })
        ctx.onSessionEnd(() => globalThis.__agentEventLog.push('session:end'))
      }
    `,
  })
  ;(globalThis as any).__agentEventLog = []
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)
  assert.equal((runtime.listToolSchemas()[0] as any)?.function?.name, 'plugin_lookup')
  await Promise.all([
    runtime.emitAgentEvent({ runId: 'run-1', threadId: 'thread-1', type: 'lifecycle', phase: 'run_start', ts: 1 }),
    runtime.emitAgentEvent({ runId: 'run-1', threadId: 'thread-1', type: 'status', status: 'thinking' }),
  ])
  await runtime.queueSessionEnd('thread-1')
  assert.deepEqual((globalThis as any).__agentEventLog, ['lifecycle:run_start', 'status:', 'session:end'])
  runtime.clear()
  assert.equal(runtime.listToolSchemas().length, 0)
  ok('registered tool schemas and ordered AgentEvent hooks are wired through the runtime')
}

// --- a nested OpenAI-shape tool schema registers under its advertised function name ---
// registerTool historically keyed tools by flat `schema.name`; the nested shape left the
// map key as `plugin-id:tool:N` while listToolSchemaEntries advertised function.name, so
// the model could call a tool the executor could never route back to the plugin handler.
{
  const dir = makePluginDir({
    manifest: { id: 'nested-tool-plugin', name: 'Nested Tool Plugin' },
    entrySource: `
      module.exports.register = function register(ctx) {
        ctx.registerTool({ type: 'function', function: { name: 'deep_scan', description: 'Scan deeply.', parameters: { type: 'object', properties: {} } } }, async () => 'scanned')
      }
    `,
  })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, true, result.error)
  const entries = runtime.listToolSchemaEntries()
  assert.equal(entries.length, 1)
  assert.equal((entries[0].schema as any).function?.name, 'deep_scan')
  assert.ok(runtime.getRegisteredTool('deep_scan'), 'the advertised function name must resolve to the plugin handler')
  ok('nested OpenAI-shape tools register under their advertised function name')
}

// --- a manifest main path may never escape the plugin directory ---
// `main: "/tmp/evil.cjs"` (absolute) or `../shared/evil.cjs` (traversal) would load
// arbitrary code from disk under the plugin's granted trust.
{
  const dir = makePluginDir({
    manifest: { id: 'escape-plugin', name: 'Escape Plugin' },
    manifestRaw: JSON.stringify({ id: 'escape-plugin', name: 'Escape Plugin', version: '1.0.0', main: '/tmp/vontra-plugin-escape-marker.cjs', trustLevel: 'unknown' }),
    omitEntryFile: true,
  })
  const runtime = new PluginRuntime()
  const result = await runtime.loadPlugin(dir)
  assert.equal(result.ok, false)
  assert.match(result.error!, /must stay inside the plugin directory/)
  ok('absolute main paths are refused')
}

console.log(`\n${passed} passed`)
