import { useEffect, useState } from 'react'
import { Icon } from '../../lib/icons'
import { api } from '../../lib/ipc'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import type { Room } from '../../lib/types'

type WorktreeRoom = Pick<Room, 'id' | 'state' | 'workspaceCwd' | 'workspace' | 'orchestratorHostId'>

export function RoomWorktree({ room }: { room: WorktreeRoom }) {
  const tx = useAuthoredText()
  const cwd = room.workspaceCwd?.trim() || null
  const kind = room.workspace?.kind
  const hostId = room.workspace?.hostId
  const origin = room.orchestratorHostId
  const scope = JSON.stringify([room.id, cwd, kind, hostId, origin])
  const [branch, setBranch] = useState<{ scope: string; name: string | null } | null>(null)

  useEffect(() => {
    if (!cwd || origin) return
    let current = true
    const workspace = kind ? { kind, hostId, path: cwd } : null
    api.getBranch(cwd, workspace)
      .then((name) => { if (current) setBranch({ scope, name: name?.trim() || null }) })
      .catch(() => { if (current) setBranch({ scope, name: null }) })
    return () => { current = false }
  }, [cwd, kind, hostId, origin, scope, room.state])

  if (!cwd) return null
  const name = branch?.scope === scope ? branch.name : null
  return (
    <div className="mt-2 min-w-0 rounded-lg border border-line bg-card px-2 py-1.5">
      <div className="flex items-center gap-1.5 text-[12px] font-medium text-ink-3">
        <Icon name="branch" size={12} />
        <span>{tx('Workspace')}</span>
      </div>
      {name && <div dir="ltr" title={name} className="mt-1 select-text break-all font-mono text-[12px] text-ink-2">{name}</div>}
      <div dir="ltr" title={cwd} className="mt-1 select-text break-all font-mono text-[11px] text-ink-3">{cwd}</div>
    </div>
  )
}
