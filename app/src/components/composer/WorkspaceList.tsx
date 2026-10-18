import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import { MenuItem } from '../ui'
import { Icon } from '../../lib/icons'
import type { SshHost, SshConnectionState, WorkspaceLocation } from '../../lib/types'
import { SshStatusDot } from './SshStatusDot'

// Where a task runs: this Mac, or any SSH host (mirrors Claude's host picker).
export function WorkspaceList({ workspace, hosts, statusMap, onLocal, onHost, onAddHost, close }: {
  workspace: WorkspaceLocation | null
  hosts: SshHost[]
  statusMap: Record<string, SshConnectionState>
  onLocal: () => void
  onHost: (host: SshHost, path: string) => void
  onAddHost: () => void
  close: () => void
}) {
  const tx = useAuthoredText()
  const [pending, setPending] = useState<SshHost | null>(null)
  const [remotePath, setRemotePath] = useState('')
  const use = (host: SshHost) => { onHost(host, remotePath.trim()); setPending(null); close() }
  return (
    <div className="max-h-[360px] overflow-y-auto scroll-thin">
      <MenuItem active={!workspace} onClick={() => { onLocal(); setPending(null); close() }}>
        <span className="flex items-center gap-2"><Icon name="monitor" size={14} className="text-ink-3" />{tx("Work locally")}</span>
      </MenuItem>
      {hosts.length > 0 && <div className="my-1 border-t border-line" />}
      {hosts.map((h) => (
        <MenuItem key={h.id} active={workspace?.hostId === h.id} onClick={() => { setPending(h); setRemotePath(h.lastRemoteDir || '') }}>
          <span className="flex flex-1 items-center gap-2">
            <Icon name="globe" size={14} className="text-ink-3" />
            <span className="flex-1 truncate">{h.label}</span>
            <span className="truncate text-[13px] text-ink-4">{h.user ? `${h.user}@` : ''}{h.hostName}</span>
            <SshStatusDot phase={statusMap[h.id]?.phase} />
          </span>
        </MenuItem>
      ))}
      {pending && (
        <div className="mt-1 border-t border-line px-2.5 pb-2 pt-2">
          <div className="mb-1 text-[13px] text-ink-3">{tx("Remote directory on ")}<span className="text-ink-2">{pending.label}</span></div>
          <input
            autoFocus
            value={remotePath}
            onChange={(e) => setRemotePath(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter' && remotePath.startsWith('/')) use(pending) }}
            placeholder="/home/you/project"
            className="w-full rounded-md border border-line bg-canvas px-2 py-1.5 text-[13px] text-ink outline-none focus:border-line-strong"
          />
          <button
            disabled={!remotePath.startsWith('/')}
            onClick={() => use(pending)}
            className="mt-1.5 w-full rounded-md bg-accent px-2 py-1.5 text-[13px] font-medium text-accent-ink transition-colors hover:bg-accent-strong disabled:bg-ink-5 disabled:text-ink-4"
          >{tx("Connect and use")}</button>
          <div className="mt-1 text-[13px] leading-4 text-ink-4">{tx("Absolute path. You'll be asked for any passphrase or unknown host key.")}</div>
        </div>
      )}
      <div className="my-1 border-t border-line" />
      <MenuItem onClick={() => { onAddHost(); close() }}>
        <span className="flex items-center gap-2 text-ink-2"><Icon name="plus" size={14} className="text-ink-3" />{tx("Add SSH host…")}</span>
      </MenuItem>
    </div>
  )
}
