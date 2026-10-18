import type { ClusterHost, ClusterPlanInput, ClusterProfile, ClusterWorkspaceSnapshot } from '../../../shared/cluster-workspace'

export const newClusterCommand = (): string => crypto.randomUUID()
export function commonProfiles(hosts: readonly ClusterHost[]): ClusterProfile[] {
  if (!hosts.length || hosts.some(host => host.state !== 'ready' || !host.nodeId || !host.snapshot?.engine.available)) return []
  return (hosts[0].snapshot?.engine.profiles ?? []).filter(profile => hosts.every(host => host.snapshot?.engine.profiles.some(candidate => candidate.id === profile.id
    && candidate.engineVersion === profile.engineVersion && candidate.mode === profile.mode && candidate.ranks === hosts.length)))
}
export function clusterMembersMatch(input: ClusterPlanInput, snapshot: ClusterWorkspaceSnapshot): boolean {
  return input.members.every(member => snapshot.hosts.some(host => host.id === member.hostId && host.revision === member.hostRevision && host.nodeId === member.nodeId && host.state === 'ready'))
}
export function discoveryEndpoint(host: string, port: number): string {
  try {
    if (!host || host.startsWith('-') || /[\s/@?#=]/.test(host) || !Number.isInteger(port) || port < 1 || port > 65535) return ''
    return new URL(`https://${host.includes(':') && !host.startsWith('[') ? `[${host}]` : host}:${port}`).href
  } catch { return '' }
}
export const shortDigest = (value: string) => value.length > 24 ? `${value.slice(0, 12)}…${value.slice(-8)}` : value
export function clusterPhaseLabel(value: string, tx: (source: string) => string): string {
  switch (value) {
    case 'queued': return tx('Queued')
    case 'running': return tx('Running')
    case 'reconciling': return tx('Reconciling')
    case 'needs-review': return tx('Needs review')
    case 'completed': return tx('Completed')
    case 'failed': return tx('Failed')
    case 'cancelled': return tx('Cancelled')
    case 'preparing': return tx('Preparing')
    case 'starting': return tx('Starting')
    case 'ready': return tx('Ready')
    case 'stopping': return tx('Stopping')
    case 'stopped': return tx('Stopped')
    default: return value
  }
}
