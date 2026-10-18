import { useRef, useState } from 'react'
import type { SshHost } from '../../lib/types'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Button, Input } from '../ui'
import type { ClusterClient } from './client'

/** Saves connection metadata only; trust and authentication remain with the existing SSH broker. */
export function ClusterSshHostForm({ client, busy, onDone }: { client: ClusterClient; busy: boolean; onDone(hostId?: string): void }) {
  const tx = useAuthoredText(), lock = useRef(false)
  const [label, setLabel] = useState(''), [hostName, setHostName] = useState(''), [user, setUser] = useState(''), [port, setPort] = useState(22), [identity, setIdentity] = useState('')
  const field = 'grid min-w-0 gap-1 text-[12px] text-ink-2'
  return <form aria-label={tx('Add SSH host…')} className="space-y-3 rounded-lg border border-accent/30 bg-canvas p-3" onSubmit={async event => {
    event.preventDefault(); if (busy || lock.current || !hostName.trim() || !label.trim()) return
    lock.current = true
    try {
      const now = Date.now(), host: SshHost = { id: crypto.randomUUID(), label: label.trim(), hostName: hostName.trim(), ...(user.trim() ? { user: user.trim() } : {}), port, identityFiles: identity.trim() ? [identity.trim()] : [], origin: 'manual', createdAt: now, updatedAt: now }
      const result = await client.execute(tx('Save'), () => api.saveSshHost(host))
      if (result) { useStore.setState({ sshHosts: result }); onDone(host.id) }
    } finally { lock.current = false }
  }}>
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2"><label className={field}>{tx('Name')}<Input aria-label={tx('Name')} autoFocus value={label} required maxLength={120} disabled={busy} onChange={event => setLabel(event.target.value)} /></label><label className={field}>{tx('Hostname')}<Input aria-label={tx('Hostname')} value={hostName} required maxLength={256} disabled={busy} onChange={event => setHostName(event.target.value)} /></label><label className={field}>{tx('Username')}<Input aria-label={tx('Username')} value={user} maxLength={64} disabled={busy} onChange={event => setUser(event.target.value)} /></label><label className={field}>{tx('Port')}<Input aria-label={tx('Port')} type="number" min={1} max={65535} value={port} disabled={busy} onChange={event => setPort(Number(event.target.value))} /></label></div>
    <label className={field}>{tx('Identity file')}<Input aria-label={tx('Identity file')} value={identity} maxLength={4096} disabled={busy} onChange={event => setIdentity(event.target.value)} /></label>
    <div className="flex flex-wrap justify-end gap-2"><Button size="sm" disabled={busy} onClick={() => { if (!busy && !lock.current) onDone() }}>{tx('Cancel')}</Button><Button size="sm" variant="primary" type="submit" disabled={busy || !label.trim() || !hostName.trim()}>{tx('Save')}</Button></div>
  </form>
}
