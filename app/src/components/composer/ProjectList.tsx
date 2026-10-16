import { useState } from 'react'
import { MenuItem } from '../ui'
import { Icon } from '../../lib/icons'
import type { Project } from '../../lib/types'

// Project choice for the composer's overflow menu: search, pick, create, or none.
export function ProjectList({ value, projects, onSelect, onAdd, close }: { value: string | null; projects: Project[]; onSelect: (id: string | null) => void; onAdd: (kind: 'scratch' | 'folder') => Promise<Project | null>; close: () => void }) {
  const [q, setQ] = useState('')
  const filtered = projects.filter((p) => !q || p.name.toLowerCase().includes(q.toLowerCase()))
  return (
    <div>
      <div className="mb-1 flex items-center gap-1.5 rounded-md bg-card-2 px-2 py-1.5">
        <Icon name="search" size={14} className="text-ink-3" />
        <input autoFocus value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search projects" className="w-full bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-3" />
      </div>
      <div className="max-h-[200px] overflow-y-auto scroll-thin">
        {filtered.map((p) => <MenuItem key={p.id} icon="folder" active={p.id === value} onClick={() => { onSelect(p.id); close() }}>{p.name}</MenuItem>)}
        {filtered.length === 0 && <div className="px-2.5 py-1.5 text-[13px] text-ink-3">No projects yet</div>}
      </div>
      <div className="my-1 border-t border-line" />
      <MenuItem icon="plus" onClick={async () => { close(); const p = await onAdd('scratch'); if (p) onSelect(p.id) }}>New project</MenuItem>
      <MenuItem icon="folder" onClick={async () => { close(); const p = await onAdd('folder'); if (p) onSelect(p.id) }}>Use existing folder…</MenuItem>
      <MenuItem icon="x" onClick={() => { onSelect(null); close() }}>Don't work in a project</MenuItem>
    </div>
  )
}
