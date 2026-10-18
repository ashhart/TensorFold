import { useRef, useState } from 'react'
import type { ClusterHost, SaveClusterHost } from '../../../shared/cluster-workspace'
import type { SshHost } from '../../lib/types'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Button, Input } from '../ui'

export function ClusterHostEditor({ host, initialUrl = '', initialKind = 'endpoint', initialPort = 8766, initialSshHostId = '', sshHosts, busy, onSave, onCancel }: {
  host?: ClusterHost; initialUrl?: string; initialKind?: 'endpoint' | 'ssh'; initialPort?: number; initialSshHostId?: string; sshHosts: readonly SshHost[]; busy: boolean
  onSave(input: SaveClusterHost): Promise<boolean>; onCancel(): void
}) {
  const tx = useAuthoredText(), submitting = useRef(false)
  const [label, setLabel] = useState(host?.label ?? ''), [role, setRole] = useState<ClusterHost['role']>(host?.role ?? 'workstation')
  const [kind, setKind] = useState<'endpoint' | 'ssh'>(host?.connection.kind ?? initialKind)
  const [url, setUrl] = useState(host?.connection.kind === 'endpoint' ? host.connection.url : initialUrl)
  const [sshHostId, setSshHostId] = useState(host?.connection.kind === 'ssh' ? host.connection.sshHostId : initialSshHostId)
  const [port, setPort] = useState(host?.connection.kind === 'ssh' ? host.connection.port : initialPort)
  const [token, setToken] = useState('')
  const field = 'grid min-w-0 gap-1 text-[12px] text-ink-2'
  const select = 'min-w-0 rounded-lg border border-line bg-card px-2 py-2 text-[13px] text-ink'
  return <form aria-label={tx('Add host')} className="space-y-3 rounded-xl border border-accent/30 bg-card p-4" onSubmit={async event => {
    event.preventDefault(); if (busy || submitting.current || !label.trim() || kind === 'ssh' && !sshHostId) return
    submitting.current = true
    try { if (await onSave({ ...(host ? { id: host.id } : {}), expectedRevision: host?.revision ?? 0, label: label.trim(), role,
      connection: kind === 'endpoint' ? { kind, url: url.trim() } : { kind, sshHostId, port }, ...(token ? { token } : {}) })) { setToken(''); onCancel() } }
    finally { submitting.current = false }
  }}>
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2"><label className={field}>{tx('Name')}<Input aria-label={tx('Name')} autoFocus value={label} maxLength={120} required disabled={busy} onChange={event => setLabel(event.target.value)} /></label><label className={field}>{tx('Connection')}<select aria-label={tx('Connection')} value={kind} disabled={busy} className={select} onChange={event => { setKind(event.target.value as 'endpoint' | 'ssh'); setToken('') }}><option value="endpoint">{tx('HTTPS endpoint')}</option><option value="ssh">{tx('Saved SSH host')}</option></select></label></div>
    {kind === 'endpoint' ? <><label className={field}>{tx('Management URL')}<Input aria-label={tx('Management URL')} type="url" value={url} required disabled={busy} onChange={event => setUrl(event.target.value)} /></label></> : <div className="grid grid-cols-1 gap-3 sm:grid-cols-2"><label className={field}>{tx('Saved SSH host')}<select aria-label={tx('Saved SSH host')} value={sshHostId} required disabled={busy} className={select} onChange={event => setSshHostId(event.target.value)}><option value="">{tx('Choose host')}</option>{sshHosts.map(ssh => <option key={ssh.id} value={ssh.id}>{ssh.label}</option>)}</select></label><label className={field}>{tx('Management port')}<Input aria-label={tx('Management port')} type="number" min={1} max={65535} value={port} disabled={busy} onChange={event => setPort(Number(event.target.value))} /></label></div>}
    <label className={field}>{tx('Management token')}<Input aria-label={tx('Management token')} type="password" autoComplete="off" value={token} maxLength={4096} disabled={busy} onChange={event => setToken(event.target.value)} /></label>
    <label className={field}>{tx('Memory role')}<select aria-label={tx('Memory role')} value={role} disabled={busy} className={select} onChange={event => setRole(event.target.value as ClusterHost['role'])}><option value="workstation">{tx('Workstation')}</option><option value="headless">{tx('Headless')}</option></select></label>
    <p className="text-[11px] leading-relaxed text-ink-3">{tx('Saving does not install, trust or start a remote service. Inspect next; a blank token keeps credentials only for the same connection.')}</p>
    <div className="flex flex-wrap justify-end gap-2"><Button size="sm" disabled={busy} onClick={() => { if (!busy && !submitting.current) { setToken(''); onCancel() } }}>{tx('Cancel')}</Button><Button size="sm" variant="primary" type="submit" disabled={busy || !label.trim() || kind === 'ssh' && !sshHostId}>{tx('Save')}</Button></div>
  </form>
}
