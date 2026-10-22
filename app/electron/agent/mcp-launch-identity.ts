import { createHash } from 'node:crypto'
import type { McpServerConfig } from './types'

export function mcpSpawnFingerprint(config: McpServerConfig): string {
  const gateway = config.dockerGateway
  const identity = {
    command: config.command,
    args: config.args ?? [],
    cwd: config.cwd || '',
    env: Object.entries(config.env ?? {}).sort(([a], [b]) => a.localeCompare(b)),
    launchMode: config.launchMode ?? 'direct',
    dockerGateway: gateway ? {
      image: gateway.image,
      workdir: gateway.workdir || '',
      network: gateway.network ?? 'none',
      readOnlyRootfs: gateway.readOnlyRootfs !== false,
      envAllowlist: [...(gateway.envAllowlist ?? [])].sort(),
      mounts: (gateway.mounts ?? []).map((mount) => ({ source: mount.source, target: mount.target, readOnly: mount.readOnly !== false })),
    } : null,
    enabled: config.enabled,
    autoConnect: config.autoConnect,
    approvalMode: config.approvalMode,
    manifestPath: config.manifestPath || '',
    manifestSignature: config.manifestSignature || '',
    manifestPublicKey: config.manifestPublicKey || '',
  }
  return createHash('sha256').update(JSON.stringify(identity)).digest('hex')
}
