import { useEffect, useMemo, useState } from 'react'
import type { ClusterDiscoveryCandidate, ClusterHost } from '../../../shared/cluster-workspace'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Button } from '../ui'
import { createClusterClient } from './client'
import { discoveryEndpoint, newClusterCommand } from './display'
import { ClusterHostEditor } from './ClusterHostEditor'
import { ClusterPlanEditor } from './ClusterPlanEditor'
import { ClusterSshSetup } from './ClusterSshSetup'
import { ClusterDeploymentCard, ClusterHostCard, ClusterJobCard, ClusterPlanCard } from './ClusterCards'

export function ClusterWorkspace({ onCloseBlockedChange }: { onCloseBlockedChange?(blocked: boolean): void }) {
  const tx = useAuthoredText(), sshHosts = useStore(state => state.sshHosts)
  const client = useMemo(() => createClusterClient(api), [])
  const [state, setState] = useState(client.snapshot), [editor, setEditor] = useState<{ host?: ClusterHost; url?: string; kind?: 'endpoint' | 'ssh'; port?: number; sshHostId?: string } | null>(null), [planning, setPlanning] = useState(false)
  const [jobPage, setJobPage] = useState(0)
  const [setupEditing, setSetupEditing] = useState(false)
  const [discovered, setDiscovered] = useState<ClusterDiscoveryCandidate[]>([]), [searched, setSearched] = useState(false)
  useEffect(() => { client.activateView(); const remove = client.subscribe(setState); if (api.platform === 'web-remote') return () => { remove(); client.dispose() }; void client.refresh(); const timer = setInterval(() => void client.refresh(true), 3000); return () => { clearInterval(timer); remove(); client.dispose() } }, [client])
  const busy = api.platform !== 'web-remote' && (!!state.pending || state.loading), closeBlocked = busy || !!editor || planning || setupEditing
  useEffect(() => { onCloseBlockedChange?.(closeBlocked); return () => onCloseBlockedChange?.(false) }, [closeBlocked, onCloseBlockedChange])
  const snapshot = state.snapshot
  const terminal = ['completed','failed','cancelled']
  const jobs = snapshot ? [...snapshot.jobs].sort((left, right) => Number(terminal.includes(left.phase)) - Number(terminal.includes(right.phase)) || right.updatedAt - left.updatedAt) : []
  const page = Math.min(jobPage, Math.max(0, Math.ceil(jobs.length / 30) - 1))
  const deployments = snapshot ? [...snapshot.deployments].sort((left, right) => Number(left.state === 'stopped') - Number(right.state === 'stopped') || right.updatedAt - left.updatedAt) : []
  return <div data-testid="cluster-workspace" className="space-y-5">
    <p className="text-[13px] leading-relaxed text-ink-2">{tx('Connect trusted TensorFold hosts, inspect real capabilities, then review a model deployment. Discovery never grants trust.')}</p>
    {api.platform === 'web-remote' ? <p role="status" className="text-[13px] text-warn">{tx('Cluster management is available in the local desktop app.')}</p> : <>
      <div className="flex flex-wrap gap-2"><Button size="sm" disabled={busy || !!editor || planning || setupEditing} onClick={() => setEditor({})}>{tx('Add host')}</Button><Button size="sm" disabled={busy} onClick={async () => { const result = await client.execute(tx('Discover hosts'), () => api.clusterDiscover()); if (result) { setDiscovered(result); setSearched(true) } }}>{tx('Discover hosts')}</Button><Button size="sm" variant="ghost" disabled={busy} onClick={() => void client.refresh()}>{tx('Refresh')}</Button></div>
      {state.pending && <p role="status" aria-live="polite" className="text-[12px] text-ink-2">{state.pending}</p>}{state.error && <p role="alert" className="rounded-lg border border-danger/30 bg-card p-3 text-[12px] text-danger"><OriginalDiagnostic>{state.error}</OriginalDiagnostic></p>}
      {editor && <ClusterHostEditor host={editor.host} initialUrl={editor.url} initialKind={editor.kind} initialPort={editor.port} initialSshHostId={editor.sshHostId} sshHosts={sshHosts} busy={busy} onCancel={() => setEditor(null)} onSave={async input => !!await client.execute(tx('Save'), () => api.clusterSaveHost(input))} />}
      {searched && !discovered.length && <p role="status" className="text-[12px] text-ink-3">{tx('No matches.')}</p>}
      {!!discovered.length && <section className="space-y-2 rounded-xl border border-line bg-card p-3"><h3 className="text-[13px] font-medium text-ink">{tx('Nearby suggestions')}</h3><p className="text-[11px] text-ink-3">{tx('These hosts are untrusted. Review the address and credentials before adding one.')}</p>{discovered.slice(0, 100).map(candidate => <div key={candidate.id} className="flex flex-wrap items-center justify-between gap-2 border-t border-line py-2"><div className="min-w-0"><p className="break-words text-[12px] text-ink-2">{candidate.label}</p><code className="break-all text-[11px] text-ink-3">{candidate.host}:{candidate.port}</code></div><Button size="xs" disabled={busy || !!editor || planning || setupEditing || candidate.expiresAt <= Date.now()} onClick={() => { if (candidate.expiresAt <= Date.now()) return; if (candidate.via === 'ssh') { const match = sshHosts.find(host => host.hostName.replace(/\.$/, '').toLowerCase() === candidate.host.replace(/\.$/, '').toLowerCase()); setEditor({ kind: 'ssh', port: candidate.port, sshHostId: match?.id }) } else setEditor({ kind: 'endpoint', url: discoveryEndpoint(candidate.host, candidate.port) }) }}>{tx('Review')}</Button></div>)}</section>}
      <ClusterSshSetup hosts={sshHosts} client={client} busy={busy || !!editor || planning} onEditingChange={setSetupEditing} />
      {!snapshot && <p role="status" className="text-[13px] text-ink-3">{state.loading ? tx('Loading…') : tx('Unavailable')}</p>}
      {snapshot && <>
        <section className="space-y-3"><h3 className="text-[14px] font-medium text-ink">{tx('Hosts')}</h3>{!snapshot.hosts.length && <p className="rounded-lg border border-dashed border-line p-4 text-[12px] text-ink-3">{tx('Add a host to inspect its engine, models and links. No remote process starts automatically.')}</p>}{snapshot.hosts.slice(0, 100).map(host => <ClusterHostCard key={host.id} host={host} busy={busy || !!editor || planning || setupEditing} onEdit={() => setEditor({ host })} onInspect={() => void client.execute(tx('Inspect'), () => api.clusterInspectHost({ id: host.id, expectedRevision: host.revision, commandId: newClusterCommand() }))} onRemove={() => void client.execute(tx('Remove'), () => api.clusterRemoveHost({ id: host.id, expectedRevision: host.revision }))} />)}</section>
        <section className="space-y-3"><div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-[14px] font-medium text-ink">{tx('Reviewed plans')}</h3><Button size="sm" disabled={busy || !!editor || planning || setupEditing || !snapshot.hosts.length} onClick={() => setPlanning(true)}>{tx('New plan')}</Button></div>{planning && <ClusterPlanEditor snapshot={snapshot} busy={busy} onCancel={() => setPlanning(false)} onPlan={async input => { const plan = await client.execute(tx('Review plan'), () => api.clusterPlan(input)); if (!plan) return false; await client.refresh(); return true }} />}{snapshot.plans.slice(-20).reverse().map(plan => <ClusterPlanCard key={plan.id} plan={plan} snapshot={snapshot} client={client} busy={busy || !!editor || planning || setupEditing} />)}</section>
        {!!snapshot.deployments.length && <section className="space-y-3"><h3 className="text-[14px] font-medium text-ink">{tx('Deployments')}</h3>{deployments.filter((deployment, index) => deployment.state !== 'stopped' || index < 20).map(deployment => <ClusterDeploymentCard key={deployment.id} deployment={deployment} client={client} busy={busy || !!editor || planning || setupEditing} />)}</section>}
        {!!snapshot.jobs.length && <section className="space-y-2"><h3 className="text-[14px] font-medium text-ink">{tx('Recent operations')}</h3>{jobs.slice(page * 30, (page + 1) * 30).map(job => <ClusterJobCard key={job.id} job={job} hosts={snapshot.hosts} client={client} busy={busy} />)}{jobs.length > 30 && <div className="flex flex-wrap items-center justify-between gap-2 text-[12px] text-ink-2"><span>{tx(' · Page {page} of {pages}', { page: page + 1, pages: Math.ceil(jobs.length / 30) })}</span><div className="flex gap-2"><Button size="xs" disabled={page === 0} onClick={() => setJobPage(page - 1)}>{tx('Previous entries')}</Button><Button size="xs" disabled={(page + 1) * 30 >= jobs.length} onClick={() => setJobPage(page + 1)}>{tx('Next entries')}</Button></div></div>}</section>}
      </>}
    </>}
  </div>
}
