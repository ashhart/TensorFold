import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import ts from 'typescript'
import * as mcpClient from '../../../agent/mcp-client'
import * as security from '../../../agent/security'
import type { McpServerConfig } from '../../../agent/types'
import * as confirmation from '../sensitive-confirmation'

const source = await fs.readFile(new URL('./mcp-ipc.ts', import.meta.url), 'utf8')
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS } }).outputText
const base = mcpClient.createMcpServer({ id: 'synthetic-mcp', name: 'Synthetic MCP', command: '/bin/echo', args: ['safe'], env: { ROUTE: 'safe' } })

function fixture(initial: McpServerConfig = base) {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const store = { mcpServers: [structuredClone(initial)] }
  const launches: McpServerConfig[] = []
  const requests: confirmation.NativeConfirmationRequest[] = []
  let role = 'owner'
  let confirm: confirmation.ConfirmSensitiveAction | undefined = async () => false
  const host = new mcpClient.McpHost()
  host.setConfigs(store.mcpServers)
  ;(host as any).connectState = async (state: any) => {
    launches.push(structuredClone(state.config))
    state.runtime.status = 'connected'
    return state.runtime
  }
  const dependencies: Record<string, unknown> = {
    '../../../agent/mcp-client': mcpClient,
    '../../../agent/security': security,
    '../../state': { store },
    '../audit': { auditEvent: async () => {}, requireSecurityRole: (required: string) => {
      if (role === 'viewer' || (required === 'owner' && role !== 'owner')) throw new Error(`${required} role required`)
    } },
    '../ipc': { handle: (name: string, fn: (...args: any[]) => Promise<any>) => handlers.set(name, fn) },
    '../persistence': { persistMcpServers: () => {} },
    '../sensitive-confirmation': confirmation,
  }
  const exports: Record<string, any> = {}
  new Function('require', 'exports', compiled)((name: string) => {
    assert.ok(name in dependencies, `Unbound actual host dependency ${name}`)
    return dependencies[name]
  }, exports)
  const register = () => exports.registerMcpIpc({ mcpHost: host, confirmSensitiveAction: confirm && (async (input: confirmation.NativeConfirmationRequest) => {
    requests.push(input)
    return confirm!(input)
  }) })
  register()
  return {
    host, launches, requests,
    fingerprint: exports.mcpSpawnFingerprint as (config: McpServerConfig) => string,
    role(value: string) { role = value },
    confirm(value: confirmation.ConfirmSensitiveAction | undefined) { confirm = value; register() },
    update(next: McpServerConfig) { store.mcpServers = [structuredClone(next)]; host.setConfigs(store.mcpServers) },
    invoke(channel: string, ...args: unknown[]) { return handlers.get(channel)!({}, ...args) },
  }
}

let failed = 0
async function test(name: string, run: () => Promise<void> | void) {
  try { await run(); console.log(`PASS ${name}`) }
  catch (error) { failed++; console.error(`FAIL ${name}`, error) }
}

await test('viewer cannot launch and operator cannot rewrite server configuration', async () => {
  const f = fixture()
  f.role('viewer')
  await assert.rejects(f.invoke('mcp:connect', base.id), /operator role/)
  await assert.rejects(f.invoke('mcp:refresh', base.id), /operator role/)
  f.role('operator')
  await assert.rejects(f.invoke('mcp:saveServer', { id: base.id, command: '/bin/false' }), /owner role/)
  assert.equal(f.launches.length + f.requests.length, 0)
})

for (const all of [false, true]) await test(`refresh ${all ? 'all servers' : 'one server'} cannot bypass rejected native launch confirmation`, async () => {
  const f = fixture()
  f.role('operator')
  await f.invoke('mcp:refresh', ...(all ? [] : [base.id])).catch(() => null)
  assert.equal(f.launches.length, 0, 'refresh must not start a rejected process')
  assert.equal(f.requests.length, 1)
})

await test('missing native confirmation wiring fails closed', async () => {
  const f = fixture()
  f.confirm(undefined)
  await assert.rejects(f.invoke('mcp:connect', base.id), /confirmation is unavailable/)
  let toolCalls = 0
  f.host.toolDefinitions = () => [{ serverId: base.id, serverName: base.name, approvalMode: 'ask', name: 'synthetic', callName: 'synthetic', inputSchema: {}, readOnlyHint: false }]
  f.host.callTool = async () => { toolCalls++; return { ok: true, output: 'synthetic' } }
  await assert.rejects(f.invoke('mcp:testTool', { callName: 'synthetic', args: {} }), /confirmation is unavailable/)
  assert.equal(toolCalls, 0)
  assert.equal(f.launches.length, 0)
})

await test('configuration changed during native approval cannot replace the approved launch', async () => {
  const f = fixture()
  f.confirm(async () => { f.update({ ...base, args: ['changed-during-dialog'] }); return true })
  await assert.rejects(f.invoke('mcp:connect', base.id), /changed|no longer matches/i)
  assert.equal(f.launches.length, 0)
})

const gateway = { image: `example.invalid/test@sha256:${'a'.repeat(64)}`, workdir: '/workspace', network: 'none' as const, readOnlyRootfs: true, envAllowlist: ['ROUTE'], mounts: [{ source: '/tmp', target: '/workspace', readOnly: true }] }
const docker = { ...base, launchMode: 'docker_gateway' as const, dockerGateway: gateway }
const changes: Array<[string, McpServerConfig, McpServerConfig]> = [
  ['environment value', base, { ...base, env: { ROUTE: 'changed' } }],
  ['Docker image', docker, { ...docker, dockerGateway: { ...gateway, image: `example.invalid/test@sha256:${'b'.repeat(64)}` } }],
  ['Docker mount source', docker, { ...docker, dockerGateway: { ...gateway, mounts: [{ source: '/var/tmp', target: '/workspace', readOnly: true }] } }],
  ['Docker mount access', docker, { ...docker, dockerGateway: { ...gateway, mounts: [{ ...gateway.mounts[0], readOnly: false }] } }],
  ['Docker network', docker, { ...docker, dockerGateway: { ...gateway, network: 'bridge' } }],
  ['Docker root filesystem', docker, { ...docker, dockerGateway: { ...gateway, readOnlyRootfs: false } }],
  ['Docker environment allowlist', docker, { ...docker, dockerGateway: { ...gateway, envAllowlist: [] } }],
  ['Docker working directory', docker, { ...docker, dockerGateway: { ...gateway, workdir: '/changed' } }],
]
for (const [name, initial, changed] of changes) await test(`${name} changes require a fresh native launch confirmation`, async () => {
  const f = fixture(initial)
  let approved = true
  f.confirm(async () => approved)
  await f.invoke('mcp:connect', initial.id)
  await f.host.disconnect(initial.id)
  f.update(changed)
  approved = false
  await assert.rejects(f.invoke('mcp:connect', initial.id), /cancelled by the local user/)
  assert.equal(f.launches.length, 1)
  assert.equal(f.requests.length, 2)
})

await test('unchanged spawn permits a confirmed reconnect and ordinary metadata refresh', async () => {
  const f = fixture()
  f.confirm(async () => true)
  await f.invoke('mcp:connect', base.id)
  await f.invoke('mcp:refresh', base.id)
  await f.host.disconnect(base.id)
  await f.invoke('mcp:connect', base.id)
  assert.equal(f.launches.length, 2)
  assert.equal(f.requests.length, 1)
})

if (failed) process.exitCode = 1
