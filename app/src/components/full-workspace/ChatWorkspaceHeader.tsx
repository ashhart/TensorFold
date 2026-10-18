import { statusText } from '../../lib/preferences/display'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useUiText } from '../../lib/preferences/hooks'
import { requestConversationSearch } from '../search/search-events'
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
  const tx = useAuthoredText()
  const text = useUiText()
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
    catch { useStore.getState().setNotice({ kind: 'error', message: tx("Could not open a chat. Your current work is still here; try New chat again.") }) }
    finally { lock.current = false; setCreating(false) }
  }
  return <header aria-label={tx("Full chat navigation")} className="drag relative z-10 shrink-0 border-b border-line/60 bg-canvas">
    <div className="flex min-h-11 items-center gap-2 pl-[96px] pr-3">
      <span className="min-w-0 flex-1 truncate text-[13px] font-medium text-ink">{title || "TensorFold"}</span>
      <FullLayoutSwitch />
    </div>
    <nav aria-label={tx("Chat workspace")} className="no-drag flex flex-wrap items-center gap-1.5 px-3 pb-2">
      <Button size="sm" variant="primary" icon="plus" disabled={creating} onClick={() => void newChat()}>{creating ? tx("Opening…") : text('common.newChat')}</Button>
      <Menu width={300} trigger={(open) => <button type="button" aria-expanded={open} aria-label={tx("Recent chats")} className="flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] text-ink hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent"><Icon name="message" size={14} />{tx("Chats")}</button>}>
        {(close) => <><div className="px-2.5 py-2 text-[12px] text-ink-2">{tx("Recent chats")}</div>{recent.length ? recent.map((item) => <MenuItem key={item.id} icon="message" active={thread?.id === item.id} onClick={() => { setView({ name: 'thread', id: item.id }); close() }}>{item.title || tx("Untitled chat")}{item.incognito ? tx(" · Not saved") : ''}</MenuItem>) : <p role="status" className="px-2.5 py-3 text-[13px] text-ink-2">{tx("No chats yet. Start one with New chat.")}</p>}<MenuItem icon="grid" onClick={() => { useStore.getState().setFullWorkspaceLayout('workspace'); close() }}>{tx("All conversations in workspace")}</MenuItem></>}
      </Menu>
      <Button size="sm" variant="ghost" icon="search" onClick={() => requestConversationSearch({ scope: 'all' })}>{text('common.search')}</Button>
      <Button size="sm" variant="ghost" icon="cpu" onClick={() => setView({ name: 'models' })}>{tx("Models")}</Button>
      <Button size="sm" variant="ghost" icon="spark" onClick={() => setView({ name: 'bots' })}>{tx("Bots")}</Button>
      <Button size="sm" variant="ghost" icon="settings" onClick={() => setView({ name: 'settings', section: 'app' })}>{text('common.settings')}</Button>
      <Menu width={320} align="end" trigger={(open) => <button type="button" aria-expanded={open} aria-label={tx("Work activity: {value0} active items, {value1} need attention", { value0: activity.length, value1: needsYou })} className="flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] text-ink hover:bg-surface-hover focus-visible:outline-2 focus-visible:outline-accent"><Icon name="bell" size={14} /><span>{tx("Work")}{activity.length ? ` ${activity.length}` : ''}</span>{needsYou > 0 && <span className="h-1.5 w-1.5 rounded-full bg-warn" />}</button>}>
        {(close) => <><div className="px-2.5 py-2 text-[12px] text-ink-2">{tx("Recorded work activity")}</div>{activity.length ? activity.slice(0, 20).map((item) => <MenuItem key={item.key} icon={item.needsYou ? 'alert' : 'layers'} onClick={() => { setView(item.view); close() }}><span className="flex min-w-0 items-center justify-between gap-2"><span className="truncate">{item.title}</span><span className="shrink-0 text-[12px] text-ink-2">{statusText(item.state, tx)}</span></span></MenuItem>) : <p role="status" className="px-2.5 py-3 text-[13px] text-ink-2">{tx("No active work recorded. Models and API servers can still be loaded.")}</p>}<MenuItem icon="bell" onClick={() => { setView({ name: 'commandCentre' }); close() }}>{tx("Open full activity")}</MenuItem></>}
      </Menu>
      {thread && <ThreadTrustBadge thread={thread} />}
      {thread && <CortheonBadge thread={thread} />}
    </nav>
  </header>
}
