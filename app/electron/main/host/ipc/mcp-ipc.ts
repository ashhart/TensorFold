import { createMcpServer, McpHost, normalizeServerConfig } from '../../../agent/mcp-client'
import type { McpServerConfig } from '../../../agent/types'
import { auditPreview, mergeMcpServerSecrets } from '../../../agent/security'
import { store } from '../../state'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { persistMcpServers } from '../persistence'
import { requireNativeConfirmation, type ConfirmSensitiveAction } from '../sensitive-confirmation'

export interface McpIpcDeps {
  mcpHost: McpHost
  /** When provided, first/changed server connects and non-read-only test calls need a native dialog. */
  confirmSensitiveAction?: ConfirmSensitiveAction
}

/** Spawn identity of a server config: what an attacker would need to change to redirect the spawn. */
export function mcpSpawnFingerprint(config: McpServerConfig): string {
  return JSON.stringify({
    command: config.command,
    args: config.args ?? [],
    cwd: config.cwd ?? '',
    envKeys: Object.keys(config.env ?? {}).sort(),
    launchMode: config.launchMode ?? 'direct',
  })
}

/** Session-scoped record of the spawn configs the local user has confirmed. */
const confirmedSpawnFingerprints = new Map<string, string>()

export function registerMcpIpc(deps: McpIpcDeps) {
  const { mcpHost, confirmSensitiveAction } = deps

handle('mcp:snapshot', async () => mcpHost.snapshot())
handle('mcp:saveServer', async (_e, input: Partial<McpServerConfig>) => {
  requireSecurityRole('owner', 'save MCP server configuration')
  const existing = input.id ? store.mcpServers.find((server) => server.id === input.id) : null
  const normalized = existing
    ? normalizeServerConfig({ ...existing, ...input, updatedAt: Date.now() })
    : createMcpServer(input)
  const next = mergeMcpServerSecrets(normalized, existing)
  store.mcpServers = existing
    ? store.mcpServers.map((server) => server.id === next.id ? next : server)
    : [next, ...store.mcpServers]
  await auditEvent('mcp.save_server', { serverId: next.id, name: auditPreview(next.name), command: auditPreview(next.command), launchMode: next.launchMode ?? 'direct', trusted: next.approvalMode === 'trusted' })
  persistMcpServers()
  mcpHost.setConfigs(store.mcpServers)
  if (!next.enabled) await mcpHost.disconnect(next.id).catch(() => null)
  return mcpHost.snapshot()
})
handle('mcp:deleteServer', async (_e, id: string) => {
  requireSecurityRole('owner', 'delete MCP server configuration')
  await mcpHost.disconnect(id).catch(() => null)
  store.mcpServers = store.mcpServers.filter((server) => server.id !== id)
  await auditEvent('mcp.delete_server', { serverId: id })
  persistMcpServers()
  mcpHost.setConfigs(store.mcpServers)
  return mcpHost.snapshot()
})
handle('mcp:connect', async (_e, id: string) => {
  requireSecurityRole('operator', 'connect MCP servers')
  // A renderer can rewrite server configs wholesale; the spawn (command/args/cwd) must
  // never change silently. The first connect of each distinct spawn identity this
  // session — and every change after it — needs a native confirmation.
  const config = store.mcpServers.find((server) => server.id === id)
  if (config && confirmSensitiveAction) {
    const fingerprint = mcpSpawnFingerprint(config)
    if (confirmedSpawnFingerprints.get(id) !== fingerprint) {
      await requireNativeConfirmation(confirmSensitiveAction, {
        title: 'Start MCP server',
        message: `Start the MCP server "${config.name}"?`,
        detail: `It will launch: ${config.command} ${(config.args ?? []).join(' ')}${config.cwd ? `\nWorking directory: ${config.cwd}` : ''}${config.launchMode === 'docker_gateway' ? '\nLaunch mode: Docker gateway' : ''}`,
        confirmLabel: 'Start server',
      })
      confirmedSpawnFingerprints.set(id, fingerprint)
    }
  }
  await mcpHost.connect(id)
  await auditEvent('mcp.connect', { serverId: id })
  return mcpHost.snapshot()
})
handle('mcp:disconnect', async (_e, id: string) => {
  requireSecurityRole('operator', 'disconnect MCP servers')
  await mcpHost.disconnect(id)
  await auditEvent('mcp.disconnect', { serverId: id })
  return mcpHost.snapshot()
})
handle('mcp:refresh', async (_e, id?: string) => {
  requireSecurityRole('operator', 'refresh MCP servers')
  if (id) await mcpHost.refresh(id)
  else await Promise.all(store.mcpServers.filter((server) => server.enabled).map((server) => mcpHost.refresh(server.id).catch(() => null)))
  return mcpHost.snapshot()
})
handle('mcp:testTool', async (_e, { callName, args }: { callName: string; args?: Record<string, any> }) => {
  requireSecurityRole('operator', 'call MCP tools')
  // Test calls bypass the agent loop's approval gate, so anything not provably
  // read-only needs the same native confirmation the other sensitive actions use.
  const tool = mcpHost.toolDefinitions().find((candidate) => candidate.callName === callName)
  if (confirmSensitiveAction && tool?.readOnlyHint !== true) {
    await requireNativeConfirmation(confirmSensitiveAction, {
      title: 'Run MCP tool',
      message: `Run the MCP tool "${callName}" for real?`,
      detail: 'This executes the tool outside the agent approval flow. Only continue for tools you trust.',
      confirmLabel: 'Run tool',
    })
  }
  await auditEvent('mcp.test_tool', { callName, args: auditPreview(JSON.stringify(args ?? {})) })
  return mcpHost.callTool(callName, args ?? {})
})

}
