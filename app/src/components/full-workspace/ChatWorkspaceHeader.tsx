import { useMemo, useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import { fullWorkspaceActivity } from '../../lib/full-workspace-activity'
import { Icon } from '../../lib/icons'
import type { Thread } from '../../lib/types'
import { Button, Menu, MenuItem } from '../ui'
import { ThreadTrustBadge } from '../ThreadTrustBadge'
import { CortheonBadge } from '../CortheonBadge'
import { FullLayoutSwitch } from './FullLayoutSwitch'

export function ChatWorkspaceHeader({ title, thread }: { title: string; thread: Thread | null | undefined }) {
  const threads = useStore((state) => state.threads)
  const status = useStore((state) => state.status)
  const approvals = useStore((state) => state.approvals)
  const jobs = useStore((state) => state.jobs)
  const rooms = useStore((state) => state.rooms)
  const roomStatus = useStore((state) => state.roomStatus)
  const automations = useStore((state) => state.automations)
  const setView = useStore((state) => state.setView)
  const [creating, setCreating] = useState(false)
  const lock = useRef(false)
  const recent = useMemo(() => threads.filter((item) => item.kind === 'chat' && !item.archivedAt).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 20), [threads])
  const activity = useMemo(() => fullWorkspaceActivity({ threads, status, approvals, jobs, rooms, roomStatus, automations }), [threads, status, approvals, jobs, rooms, roomStatus, automations])
  const needsYou = activity.filter((item) => item.needsYou).length
  const newChat = async () => {
    if (lock.current) return
    lock.current = true; setCreating(true)
    try { await useStore.getState().newChat() }
    catch { useStore.getState().setNotice({ kind: 'error', message: 'Could not open a chat. Your current work is still here; try New chat again.' }) }
    finally { lock.current = false; setCreating(false) }
  }
  return <header aria-label="Full chat navigation" className="drag relative z-10 shrink-0 border-b border-line/60 bg-canvas">
    <div className="flex min-h-11 items-center gap-2 pl-[96px] pr-3">
      <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{title || 'TensorFold'}</span>
      <FullLayoutSwitch />
    </div>
    <nav aria-label="Chat workspace" className="no-drag flex flex-wrap items-center gap-1.5 px-3 pb-2">
      <Button size="sm" variant="primary" icon="plus" disabled={creating} onClick={() => void newChat()}>{creating ? 'Opening…' : 'New chat'}</Button>
      <Menu width={300} trigger={(open) => <button type="button" aria-expanded={open} aria-label="Recent chats" className="flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] text-ink hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent"><Icon name="message" size={14} />Chats</button>}>
        {(close) => <><div className="px-2.5 py-2 text-[12px] text-ink-2">Recent chats</div>{recent.length ? recent.map((item) => <MenuItem key={item.id} icon="message" active={thread?.id === item.id} onClick={() => { setView({ name: 'thread', id: item.id }); close() }}>{item.title || 'Untitled chat'}{item.incognito ? ' · Not saved' : ''}</MenuItem>) : <p role="status" className="px-2.5 py-3 text-[13px] text-ink-2">No chats yet. Start one with New chat.</p>}<MenuItem icon="grid" onClick={() => { useStore.getState().setFullWorkspaceLayout('workspace'); close() }}>All conversations in workspace</MenuItem></>}
      </Menu>
      <Button size="sm" variant="ghost" icon="cpu" onClick={() => setView({ name: 'models' })}>Models</Button>
      <Button size="sm" variant="ghost" icon="spark" onClick={() => setView({ name: 'bots' })}>Bots</Button>
      <Button size="sm" variant="ghost" icon="settings" onClick={() => setView({ name: 'settings', section: 'app' })}>Settings</Button>
      <Menu width={320} align="end" trigger={(open) => <button type="button" aria-expanded={open} aria-label={`Work activity: ${activity.length} active items, ${needsYou} need attention`} className="flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] text-ink hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent"><Icon name="bell" size={14} /><span>Work{activity.length ? ` ${activity.length}` : ''}</span>{needsYou > 0 && <span className="h-1.5 w-1.5 rounded-full bg-warn" />}</button>}>
        {(close) => <><div className="px-2.5 py-2 text-[12px] text-ink-2">Recorded work activity</div>{activity.length ? activity.slice(0, 20).map((item) => <MenuItem key={item.key} icon={item.needsYou ? 'alert' : 'layers'} onClick={() => { setView(item.view); close() }}><span className="flex min-w-0 items-center justify-between gap-2"><span className="truncate">{item.title}</span><span className="shrink-0 text-[12px] text-ink-2">{item.state}</span></span></MenuItem>) : <p role="status" className="px-2.5 py-3 text-[13px] text-ink-2">No active work recorded. Models and API servers can still be loaded.</p>}<MenuItem icon="bell" onClick={() => { setView({ name: 'commandCentre' }); close() }}>Open full activity</MenuItem></>}
      </Menu>
      {thread && <ThreadTrustBadge thread={thread} />}
      {thread && <CortheonBadge thread={thread} />}
    </nav>
  </header>
}
