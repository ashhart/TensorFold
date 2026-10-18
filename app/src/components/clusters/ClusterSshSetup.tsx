import { useEffect, useRef, useState } from 'react'
import type { ClusterBootstrapPreview, ClusterSshKey, ClusterWorkspaceApi } from '../../../shared/cluster-workspace'
import type { SshHost } from '../../lib/types'
import { api } from '../../lib/ipc'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Button, Input } from '../ui'
import { ClusterSshHostForm } from './ClusterSshHostForm'
import type { ClusterClient } from './client'
import { newClusterCommand } from './display'

export function ClusterSshSetup({ hosts, client, busy, onEditingChange }: { hosts: readonly SshHost[]; client: ClusterClient; busy: boolean; onEditingChange?(editing: boolean): void }) {
  const tx = useAuthoredText()
  const [adding, setAdding] = useState(false)
  useEffect(() => () => onEditingChange?.(false), [onEditingChange])
  const [sshHostId, setSshHostId] = useState(''), [port, setPort] = useState(8766)
  const [key, setKey] = useState<ClusterSshKey | null>(null), [preview, setPreview] = useState<ClusterBootstrapPreview | null>(null)
  const current = useRef({ sshHostId, port }); current.current = { sshHostId, port }
  const inspectKey = (action: 'inspect' | 'create') => client.execute(tx('SSH key'), () => api.clusterSshKey({ action })).then(result => { if (result) setKey(result) })
  const matches = !!preview && preview.sshHostId === sshHostId && preview.expiresAt > Date.now()
  const copyKey = () => client.execute(tx('Copy public key'), async () => {
    if (!key?.publicKey) return
    const result = await api.writeClipboard(key.publicKey); if (result?.ok === false) throw new Error('Clipboard access was refused.')
  })
  const select = 'min-w-0 rounded-lg border border-line bg-card px-2 py-2 text-[13px] text-ink'
  return <details className="rounded-xl border border-line bg-card p-3">
    <summary className="cursor-pointer text-[13px] font-medium text-ink focus-visible:outline-2 focus-visible:outline-accent">{tx('SSH and headless setup')}</summary>
    <div className="mt-3 space-y-3">
      <p className="text-[12px] leading-relaxed text-ink-2">{tx('Use a saved SSH connection and compare its host fingerprint. Installation is a separate reviewed action; drivers and system security stay under your control.')}</p>
      <div className="flex flex-wrap gap-2"><Button size="xs" disabled={busy || adding} onClick={() => { setAdding(true); onEditingChange?.(true) }}>{tx('Add SSH host…')}</Button><Button size="xs" disabled={busy} onClick={() => void inspectKey('inspect')}>{tx('Inspect')}</Button><Button size="xs" disabled={busy} onClick={() => void inspectKey('create')}>{tx('Create SSH key')}</Button>{key?.publicKey && <Button size="xs" disabled={busy} onClick={() => void copyKey()}>{tx('Copy public key')}</Button>}</div>
      {adding && <ClusterSshHostForm client={client} busy={busy} onDone={id => { setAdding(false); onEditingChange?.(false); if (id) { setSshHostId(id); setPreview(null) } }} />}
      {key && <div className="space-y-1 text-[12px] text-ink-2">{key.fingerprint && <code className="block break-all select-text">{key.fingerprint}</code>}<OriginalDiagnostic>{key.message}</OriginalDiagnostic></div>}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2"><label className="grid min-w-0 gap-1 text-[12px] text-ink-2">{tx('Saved SSH host')}<select aria-label={tx('Saved SSH host')} value={sshHostId} className={select} disabled={busy} onChange={event => { setSshHostId(event.target.value); setPreview(null) }}><option value="">{tx('Choose host')}</option>{hosts.map(host => <option key={host.id} value={host.id}>{host.label}</option>)}</select></label><label className="grid gap-1 text-[12px] text-ink-2">{tx('Management port')}<Input aria-label={tx('Management port')} type="number" min={1} max={65535} value={port} disabled={busy || adding} onChange={event => { setPort(Number(event.target.value)); setPreview(null) }} /></label></div>
      <Button size="sm" disabled={busy || adding || !sshHostId} onClick={async () => {
        const owner = { sshHostId, port }
        const result = await client.execute(tx('Review setup'), () => api.clusterBootstrapPreview(owner))
        if (result && current.current.sshHostId === owner.sshHostId && current.current.port === owner.port) setPreview(result)
      }}>{tx('Review setup')}</Button>
      {preview && <section aria-label={tx('Review setup')} className="space-y-2 rounded-lg border border-line bg-canvas p-3">
        <p className="text-[12px] text-ink-2">{preview.platform}</p><ol className="list-inside list-decimal space-y-2 text-[12px] leading-relaxed text-ink-2">{preview.steps.map((step, index) => <li key={index} className="break-words">{step}</li>)}</ol>
        {preview.blockers.map((reason, index) => <p key={index} className="text-[12px] text-warn"><OriginalDiagnostic>{reason}</OriginalDiagnostic></p>)}
        {!matches && <p role="status" className="text-[12px] text-warn">{tx('This review expired or changed. Review again before applying.')}</p>}
        <Button size="sm" variant="primary" disabled={busy || !matches || !!preview.blockers.length} onClick={async () => {
          if (!matches || preview.expiresAt <= Date.now()) return
          const result = await client.execute(tx('Apply reviewed setup'), () => api.clusterBootstrapApply({ previewId: preview.id, digest: preview.digest, commandId: newClusterCommand() }))
          if (result) setPreview(null)
        }}>{tx('Apply reviewed setup')}</Button>
      </section>}
    </div>
  </details>
}
