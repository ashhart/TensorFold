import { createMcpServer, McpHost, normalizeServerConfig, mcpSpawnFingerprint } from '../../../agent/mcp-client'
import type { McpServerConfig } from '../../../agent/types'
import { auditPreview, mergeMcpServerSecrets } from '../../../agent/security'
import { store } from '../../state'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { persistMcpServers } from '../persistence'
import { requireNativeConfirmation, type ConfirmSensitiveAction } from '../sensitive-confirmation'
export { mcpSpawnFingerprint } from '../../../agent/mcp-client'

export interface McpIpcDeps {
  mcpHost: McpHost
  // Native confirmation for first or changed launches and non-read-only test calls.
  confirmSensitiveAction?: ConfirmSensitiveAction
}

export function registerMcpIpc(deps: McpIpcDeps) {
  const { mcpHost, confirmSensitiveAction } = deps
  const confirmedSpawnFingerprints = new Map<string, string>()

  async function connectConfirmed(id: string) {
    const stored = store.mcpServers.find((server) => server.id === id)
    if (!stored) throw new Error('MCP server not found.')
    const config = normalizeServerConfig(stored)
    const fingerprint = mcpSpawnFingerprint(config)
    if (confirmedSpawnFingerprints.get(id) !== fingerprint) {
      await requireNativeConfirmation(confirmSensitiveAction, {
        title: 'Start MCP server',
        message: `Start the MCP server "${config.name}"?`,
        detail: launchDetails(config),
        confirmLabel: 'Start server',
      })
    }
    const current = store.mcpServers.find((server) => server.id === id)
    if (!current || mcpSpawnFingerprint(normalizeServerConfig(current)) !== fingerprint) {
      throw new Error('MCP server configuration changed during confirmation. Review it and try again.')
    }
    await mcpHost.connect(id, fingerprint)
    confirmedSpawnFingerprints.set(id, fingerprint)
  }

  async function refreshConfirmed(id: string) {
    const server = mcpHost.snapshot().servers.find((candidate) => candidate.id === id)
    if (server?.runtime.status === 'connected') return mcpHost.refresh(id)
    await connectConfirmed(id)
  }

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
  await connectConfirmed(id)
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
  if (id) await refreshConfirmed(id)
  else for (const server of store.mcpServers.filter((candidate) => candidate.enabled)) await refreshConfirmed(server.id).catch(() => null)
  return mcpHost.snapshot()
})
handle('mcp:testTool', async (_e, { callName, args }: { callName: string; args?: Record<string, any> }) => {
  requireSecurityRole('operator', 'call MCP tools')
  const tool = mcpHost.toolDefinitions().find((candidate) => candidate.callName === callName)
  if (tool?.readOnlyHint !== true) {
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

function launchDetails(config: McpServerConfig): string {
  const details = [
    `Command: ${JSON.stringify(config.command)}`,
    `Arguments: ${JSON.stringify(config.args)}`,
    `Working directory: ${JSON.stringify(config.cwd || process.cwd())}`,
    `Environment keys: ${Object.keys(config.env ?? {}).sort().join(', ') || '(none)'}`,
  ]
  if (config.launchMode === 'docker_gateway') details.push(`Docker gateway: ${JSON.stringify(config.dockerGateway)}`)
  if (config.manifestPath) details.push(`Signed manifest: ${JSON.stringify(config.manifestPath)}`)
  return details.join('\n')
}
