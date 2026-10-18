import { createHash, randomUUID } from 'node:crypto'
import type { ClusterNodeSnapshot, ClusterPlan, ClusterPlanInput, ClusterProfile } from '../../../../shared/cluster-workspace'
import { validateClusterPlanInput } from '../../../../shared/cluster-validation'
import type { StoredClusterHost } from './store'
import type { ClusterNodeClient } from './client'

export const clusterDigest = (value: unknown): string => createHash('sha256').update(JSON.stringify(value)).digest('hex')
export function memberAdmissionInput(plan: ClusterPlanInput, member: ClusterPlanInput['members'][number], host: StoredClusterHost) {
  return { expectedNodeId: member.nodeId, model: plan.model, profileId: plan.profileId, generation: plan.generation, role: host.role, transport: plan.transport, rank: member.rank, worldSize: plan.members.length, links: member.link ? [member.link] : [] }
}
export async function buildClusterPlan(raw: ClusterPlanInput, options: {
  hosts: StoredClusterHost[]; client(host: StoredClusterHost): Promise<ClusterNodeClient>; now?: () => number
}): Promise<ClusterPlan> {
  const input = validateClusterPlanInput(raw), now = (options.now ?? Date.now)()
  const blockers: string[] = [], warnings: string[] = [], profiles: ClusterProfile[] = [], observed: ClusterNodeSnapshot[] = []
  const admission: ClusterPlan['admission'] = []
  let stageAllowed = true
  for (const member of input.members) {
    const host = options.hosts.find(host => host.id === member.hostId)
    if (!host || host.revision !== member.hostRevision || host.nodeId !== member.nodeId || host.state !== 'ready' || !host.token) { blockers.push('Inspect each selected machine after its last configuration change.'); stageAllowed = false; continue }
    const client = await options.client(host), snapshot = await client.status()
    observed.push(snapshot)
    const profile = snapshot.engine.profiles.find(row => row.id === input.profileId)
    if (!snapshot.engine.available || !profile || profile.ranks !== input.members.length || !profile.transports.includes(input.transport) || profile.platform !== snapshot.platform || profile.arch !== snapshot.arch || profile.engineVersion !== snapshot.engine.version) {
      blockers.push(`${host.label}: the engine has not reported an implemented profile for this model topology and transport.`); stageAllowed = false; continue
    }
    profiles.push(profile)
    if (input.members.length > 1 && !member.link) { blockers.push(`${host.label}: choose an explicit peer link before staging.`); stageAllowed = false; continue }
    if (input.transport !== 'tcp') {
      const link = snapshot.links.find(link => link.transport === input.transport && link.available && link.verified && link.proof?.matched && input.members.some(other => other.nodeId === link.proof?.peerNodeId && other.rank !== member.rank))
      if (!link || now - link.checkedAt > 60_000 || link.checkedAt > now + 5_000 || member.link && !link.devices.includes(member.link.device)) { blockers.push(`${host.label}: verify the selected data link on both machines; adapter discovery alone is insufficient.`); stageAllowed = false; continue }
    }
    const checked = await client.admission(memberAdmissionInput(input, member, host))
    if (checked.nodeId !== member.nodeId || checked.profile?.id !== profile.id) { blockers.push(`${host.label}: model admission no longer matches the reviewed engine.`); stageAllowed = false; continue }
    admission.push({ hostId: host.id, allowed: checked.allowed, requiredBytes: checked.requiredBytes, budgetBytes: checked.budgetBytes, reason: checked.reason })
    if (!checked.allowed) {
      blockers.push(`${host.label}: ${checked.reason || 'The engine refused model admission.'}`)
      if (checked.cached || checked.requiredBytes !== null && checked.requiredBytes > checked.budgetBytes) stageAllowed = false
    }
    if (!checked.cached) warnings.push(`${host.label}: stage the pinned checkpoint and review a new plan before loading.`)
    if (checked.requiredBytes === null) warnings.push(`${host.label}: the engine has not reported complete memory requirements; a load is not admitted.`)
  }
  if (profiles.length !== input.members.length || admission.length !== input.members.length) stageAllowed = false
  if (profiles.some(profile => profile.mode !== profiles[0].mode || profile.poolsWeights !== profiles[0].poolsWeights || profile.engineVersion !== profiles[0].engineVersion)) { blockers.push('Every rank must report the same implemented execution profile and engine version.'); stageAllowed = false }
  if (input.members.length > 1) {
    for (const member of input.members) {
      const peer = input.members.find(row => row.rank === member.link?.peer)
      if (!peer?.link || peer.link.peer !== member.rank || peer.link.name !== member.link?.name || (member.link.peerPort ?? member.link.port) !== peer.link.port || (peer.link.peerPort ?? peer.link.port) !== member.link.port) { blockers.push('Peer ranks, connection names and local/peer ports must be complementary.'); stageAllowed = false; break }
    }
  }
  if (observed.length > 1 && new Set(observed.map(node => node.platform)).size > 1) warnings.push('This plan crosses operating systems; only explicitly reported mixed-host engine support can admit it.')
  if (profiles[0]?.mode === 'replicated-speed-up') warnings.push('Every machine keeps a full model replica; this profile does not pool model-weight capacity.')
  if (admission.some(item => !item.allowed)) warnings.push('A staged checkpoint still needs fresh native admission immediately before loading.')
  const data = { ...input, blockers: [...new Set(blockers)], warnings: [...new Set(warnings)], poolsWeights: profiles.length === input.members.length && profiles.every(profile => profile.poolsWeights), stageAllowed, admission }
  return { ...data, id: randomUUID(), digest: clusterDigest(data), createdAt: now, expiresAt: now + 15 * 60_000 }
}
