import { useState } from 'react'
import type { ClusterDeployment, ClusterHost, ClusterJob, ClusterPlan, ClusterWorkspaceSnapshot } from '../../../shared/cluster-workspace'
import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { api } from '../../lib/ipc'
import { formatBytes } from '../../lib/engine'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Button } from '../ui'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import type { ClusterClient } from './client'
import { clusterMembersMatch, clusterPhaseLabel, newClusterCommand, shortDigest } from './display'

export function ClusterHostCard({ host, busy, onInspect, onEdit, onRemove }: { host: ClusterHost; busy: boolean; onInspect(): void; onEdit(): void; onRemove(): void }) {
  const tx = useAuthoredText(), [confirm, setConfirm] = useState(false)
  return <article className="space-y-3 rounded-xl border border-line bg-card p-3">
    <div className="flex flex-wrap items-start justify-between gap-2"><div className="min-w-0"><h3 className="break-words text-[14px] font-medium text-ink">{host.label}</h3><p className="mt-1 text-[11px] text-ink-3">{host.role === 'headless' ? tx('Headless') : tx('Workstation')} · {host.state === 'ready' ? tx('Management verified') : host.state === 'identity-changed' ? tx('Host key changed') : tx('Not connected')}</p></div><div className="flex gap-1"><Button size="xs" disabled={busy} onClick={onInspect}>{tx('Inspect')}</Button><Button size="xs" disabled={busy} onClick={onEdit}>{tx('Edit')}</Button><Button size="xs" variant="ghost" disabled={busy} onClick={() => setConfirm(true)}>{tx('Remove')}</Button></div></div>
    <code className="block break-all text-[11px] text-ink-3">{host.connection.kind === 'endpoint' ? host.connection.url : host.connection.sshHostId}</code>
    {confirm && <div role="group" className="flex flex-wrap items-center justify-between gap-2 rounded border border-line p-2 text-[12px] text-ink-2"><span>{tx('Remove this host from the workspace? Active work must be stopped first.')}</span><div className="flex gap-1"><Button size="xs" disabled={busy} onClick={() => setConfirm(false)}>{tx('Cancel')}</Button><Button size="xs" variant="danger" disabled={busy} onClick={() => { if (!busy) { setConfirm(false); onRemove() } }}>{tx('Remove')}</Button></div></div>}
    {host.lastError && <p className="text-[12px] text-warn"><OriginalDiagnostic>{host.lastError}</OriginalDiagnostic></p>}
    {host.snapshot && <div className="space-y-2 text-[12px] text-ink-2"><p>{host.snapshot.platform} {host.snapshot.arch} · {host.snapshot.engine.version ?? tx('Unavailable')}</p><p>{tx('Free memory')}: {formatBytes(host.snapshot.freeMemoryBytes)} / {formatBytes(host.snapshot.totalMemoryBytes)}</p><time className="block text-[11px] text-ink-3" dateTime={new Date(host.snapshot.observedAt).toISOString()}>{new Date(host.snapshot.observedAt).toLocaleString(currentIntlLocale())}</time>{host.snapshot.engine.reason && <p className="text-warn"><OriginalDiagnostic>{host.snapshot.engine.reason}</OriginalDiagnostic></p>}
      <details><summary className="cursor-pointer text-[12px] focus-visible:outline-2 focus-visible:outline-accent">{tx('Capabilities and links')}</summary><div className="mt-2 space-y-2">
        {host.snapshot.engine.profiles.length ? host.snapshot.engine.profiles.map(profile => <p key={profile.id}>{profile.title} · {profile.ranks} · {profile.poolsWeights ? tx('This profile divides model weights across members.') : tx('Replicated weights')}</p>) : <p>{tx('Unavailable')}</p>}
        {host.snapshot.links.map((link, index) => <div key={index} className="rounded border border-line p-2"><p>{link.transport} · {link.verified && link.proof?.matched ? tx('Byte verification passed') : link.available ? tx('Detected') : tx('Unavailable')}</p><p className="break-words text-[11px] text-ink-3">{link.devices.join(', ')}</p><OriginalDiagnostic>{link.message}</OriginalDiagnostic></div>)}
      </div></details>
    </div>}
  </article>
}

export function ClusterPlanCard({ plan, snapshot, client, busy }: { plan: ClusterPlan; snapshot: ClusterWorkspaceSnapshot; client: ClusterClient; busy: boolean }) {
  const tx = useAuthoredText()
  const current = clusterMembersMatch(plan, snapshot) && plan.expiresAt > Date.now(), occupied = snapshot.jobs.some(job => ['queued','running','reconciling','needs-review'].includes(job.phase) && job.hostIds.some(id => plan.members.some(member => member.hostId === id))) || snapshot.deployments.some(deployment => ['preparing','starting','ready','stopping','needs-review'].includes(deployment.state) && deployment.plan.members.some(member => plan.members.some(candidate => candidate.hostId === member.hostId))), blocked = !current || occupied || !!plan.blockers.length, stageBlocked = !current || occupied || !plan.stageAllowed
  const run = (kind: 'stage' | 'activate') => {
    if (busy || (kind === 'stage' ? stageBlocked : blocked) || plan.expiresAt <= Date.now() || !clusterMembersMatch(plan, client.snapshot().snapshot ?? snapshot)) return
    return client.execute(kind === 'stage' ? tx('Stage model') : tx('Start deployment'), () => kind === 'stage'
      ? api.clusterEnsureModel({ planId: plan.id, digest: plan.digest, commandId: newClusterCommand() })
      : api.clusterActivate({ planId: plan.id, digest: plan.digest, commandId: newClusterCommand() }))
  }
  return <article className="space-y-2 rounded-xl border border-line bg-card p-3">
    <h3 className="break-words text-[14px] font-medium text-ink">{plan.name}</h3><code className="block break-all text-[11px] text-ink-3">{plan.model.repo} @ {plan.model.revision}</code><p className="text-[12px] text-ink-2">{plan.profileId} · {plan.transport} · {plan.generation.contextTokens} / {plan.generation.maxOutputTokens} · {plan.generation.temperature}</p>
    <p className="text-[11px] text-ink-3">{tx('Plan digest')}: <code>{shortDigest(plan.digest)}</code></p>
    <p className="text-[11px] text-ink-3">{plan.poolsWeights ? tx('This profile divides model weights across members.') : tx('Replicated weights')}</p>
    {plan.members.map(member => <p key={member.hostId} className="break-words text-[12px] text-ink-2">{member.rank} · {snapshot.hosts.find(host => host.id === member.hostId)?.label ?? member.hostId} · {member.nodeId}{member.link ? ` · ${member.link.name} / ${member.link.device} → ${member.link.peer}` : ''}</p>)}
    {plan.admission.map(row => <div key={row.hostId} className="text-[12px] text-ink-2">{snapshot.hosts.find(host => host.id === row.hostId)?.label ?? row.hostId}: {row.requiredBytes === null ? tx('Unknown') : formatBytes(row.requiredBytes)} / {formatBytes(row.budgetBytes)}<OriginalDiagnostic>{row.reason}</OriginalDiagnostic></div>)}
    {[...plan.blockers, ...plan.warnings].map((reason, index) => <p key={index} className="text-[12px] text-warn"><OriginalDiagnostic>{reason}</OriginalDiagnostic></p>)}
    {!current && <p role="status" className="text-[12px] text-warn">{tx('This review expired or changed. Review again before applying.')}</p>}
    <div className="flex flex-wrap gap-2"><Button size="sm" disabled={busy || stageBlocked} onClick={() => void run('stage')}>{tx('Stage model')}</Button><Button size="sm" variant="primary" disabled={busy || blocked} onClick={() => void run('activate')}>{tx('Start deployment')}</Button></div>
  </article>
}

export function ClusterJobCard({ job, client, busy, hosts }: { job: ClusterJob; client: ClusterClient; busy: boolean; hosts: readonly ClusterHost[] }) {
  const tx = useAuthoredText(), active = ['queued','running','reconciling'].includes(job.phase)
  return <article className="space-y-2 rounded-lg border border-line bg-card p-3"><div className="flex flex-wrap items-center justify-between gap-2"><span className="text-[12px] font-medium text-ink">{job.kind === 'inspect' ? tx('Inspect') : job.kind === 'ensure-model' ? tx('Stage model') : job.kind === 'activate' ? tx('Start deployment') : job.kind === 'stop' ? tx('Stop') : tx('Apply reviewed setup')} · {clusterPhaseLabel(job.phase, tx)}</span>{active && <Button size="xs" disabled={busy || job.cancelRequested} onClick={() => void client.execute(tx('Cancel'), () => api.clusterCancelJob({ jobId: job.id }))}>{job.cancelRequested ? tx('Cancelling…') : tx('Cancel')}</Button>}</div><p className="break-words text-[11px] text-ink-3">{job.hostIds.map(id => hosts.find(host => host.id === id)?.label ?? id).join(', ')}</p><p role="status" className="break-words text-[12px] text-ink-2"><OriginalDiagnostic>{`${job.step}: ${job.message}`}</OriginalDiagnostic></p>{job.phase === 'needs-review' && <><p className="text-[11px] text-warn">{tx('The last acknowledgement is uncertain. Reconcile before retrying; a restart never replays this command.')}</p><Button size="xs" disabled={busy} onClick={() => void client.execute(tx('Reconcile'), () => api.clusterReconcile({ jobId: job.id }))}>{tx('Reconcile')}</Button></>}</article>
}

export function ClusterDeploymentCard({ deployment, client, busy }: { deployment: ClusterDeployment; client: ClusterClient; busy: boolean }) {
  const tx = useAuthoredText(), ready = deployment.state === 'ready' && deployment.endpoint?.ready
  const [copyKey, setCopyKey] = useState(false)
  const copy = (includeKey: boolean) => client.execute(tx('Copy'), async () => {
    if (!ready) return
    const value = includeKey ? await api.clusterEndpointCredentials({ deploymentId: deployment.id }) : { baseUrl: deployment.endpoint!.baseUrl, apiKey: '', model: deployment.endpoint!.model }
    const result = await api.writeClipboard(includeKey ? value.apiKey : value.baseUrl)
    if (result?.ok === false) throw new Error('Clipboard access was refused.')
    setCopyKey(false)
  })
  return <article className="space-y-2 rounded-xl border border-line bg-card p-3"><h3 className="break-words text-[14px] font-medium text-ink">{deployment.name}</h3><p className="text-[12px] text-ink-2">{clusterPhaseLabel(deployment.state, tx)}</p>{deployment.error && <p className="text-[12px] text-warn"><OriginalDiagnostic>{deployment.error}</OriginalDiagnostic></p>}
    {deployment.endpoint && <div className="space-y-2"><code className="block break-all text-[11px] text-ink-2">{deployment.endpoint.baseUrl}</code><p className="text-[11px] text-ink-3">{tx('This is a remote provider; phone, Google and bot restrictions still apply.')}</p>{ready && <div className="flex flex-wrap gap-2"><Button size="xs" disabled={busy} onClick={() => void copy(false)}>{tx('Copy endpoint')}</Button>{deployment.endpoint.hasKey && <Button size="xs" disabled={busy} onClick={() => setCopyKey(true)}>{tx('Copy API key')}</Button>}<Button size="xs" disabled={busy} onClick={() => void client.execute(tx('Use as provider'), () => api.clusterUseEndpoint({ deploymentId: deployment.id }))}>{tx('Use as provider')}</Button></div>}</div>}
    {copyKey && <div role="group" className="flex flex-wrap items-center justify-between gap-2 rounded border border-line p-2 text-[12px] text-ink-2"><span>{tx('Copy API key')}</span><div className="flex gap-1"><Button size="xs" disabled={busy} onClick={() => setCopyKey(false)}>{tx('Cancel')}</Button><Button size="xs" disabled={busy || !ready} onClick={() => void copy(true)}>{tx('Copy')}</Button></div></div>}
    <div className="flex flex-wrap gap-2">{deployment.state !== 'stopped' && <Button size="sm" disabled={busy} onClick={() => void client.execute(tx('Reconcile'), () => api.clusterReconcile({ deploymentId: deployment.id }))}>{tx('Reconcile')}</Button>}{['ready','starting','preparing','needs-review','failed'].includes(deployment.state) && <Button size="sm" variant="danger" disabled={busy} onClick={() => void client.execute(tx('Stop'), () => api.clusterStop({ deploymentId: deployment.id, commandId: newClusterCommand() }))}>{tx('Stop')}</Button>}</div>
  </article>
}
