// The Chat's list of conversations: pinned ones first, then grouped by date, by folder or by model; each with a
// "⋯" menu to pin, rename, move to a folder or delete. What is done with the open conversation (check the speed-up,
// replay, compare, export) sits above it, in the conversation itself.
import { useEffect, useRef, useState } from 'react'
import { api, type ConversationRow, type ConversationFolder } from '../api'
import type { Ctx } from '../ctx'
import { Ask, AskName } from '../ask'
import { ModelLogo } from '../brandLogo'

export type Grouping = 'recent' | 'folder' | 'model'
const GROUP_KEY = 'tfs-chat-grouping'
const DAY = 86400

/** "Today", "Yesterday", "This week", "This month", "Earlier", by the day a conversation was last written in. */
function dateGroup(t: number, now = Date.now() / 1000): string {
  const start = new Date(); start.setHours(0, 0, 0, 0)
  const today = start.getTime() / 1000
  if (t >= today) return 'Today'
  if (t >= today - DAY) return 'Yesterday'
  if (t >= today - 6 * DAY) return 'This week'
  if (t >= now - 30 * DAY) return 'This month'
  return 'Earlier'
}

function groupsOf(convs: ConversationRow[], by: Grouping, folders: ConversationFolder[]): [string, ConversationRow[]][] {
  const out = new Map<string, ConversationRow[]>()
  if (by === 'folder') for (const f of folders) out.set(f.name, [])          // a folder made and still empty shows too
  for (const c of convs) {
    const key = by === 'recent' ? dateGroup(c.updated) : by === 'folder' ? c.folder ?? '' : c.model ?? ''
    out.set(key, [...(out.get(key) ?? []), c])
  }
  const entries = [...out.entries()]
  if (by === 'folder' || by === 'model') {
    // named groups in order, the unnamed one ("No folder", "No reply yet") last
    entries.sort(([a], [b]) => (a === '' ? 1 : b === '' ? -1 : a.localeCompare(b)))
  }
  return entries
}

/** The conversations that ask the same questions as `id`: its original, the original's other replays, and theirs. */
export function sameQuestions(convs: ConversationRow[], id: string): ConversationRow[] {
  const by = new Map(convs.map((c) => [c.id, c]))
  const root = (c: ConversationRow) => { let r = c; for (let i = 0; i < 20 && r.replay_of && by.has(r.replay_of); i++) r = by.get(r.replay_of)!; return r.id }
  const me = by.get(id)
  if (!me) return []
  const mine = root(me)
  return convs.filter((c) => c.id !== id && root(c) === mine)
}

const short = (model: string | null | undefined) => model ? model.split('/').pop()! : null

/** What a conversation's replies were written with, in words: the model, and for a replay the speed-up or the
 *  engine it was sent to. Its question is the same as the others', so it is left out. */
export function madeWith(c: ConversationRow): string {
  const t = c.target
  if (!c.replay_of || !t) return `the original, ${short(c.model) ?? 'no reply yet'}`
  if (t.kind === 'address') return `${short(t.model) ?? 'a model'} on ${t.label || t.url}`
  return `${short(t.model ?? c.model) ?? 'the served model'}, ${t.drafts === false ? 'without' : 'with'} the speed-up`
}

/** Names for a list of conversations, with the time each was made when two would read the same. */
export function namesOf(rows: ConversationRow[]): Map<string, string> {
  const base = rows.map(madeWith)
  const at = (t: number) => new Date(t * 1000).toLocaleString([], { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
  return new Map(rows.map((c, i) => [c.id, base.filter((b) => b === base[i]).length > 1 ? `${base[i]} (${at(c.created)})` : base[i]]))
}

export function ConversationList({ convs, more, loadMore, convId, q, setQ, open, refresh, onDeleted, toast }: {
  more?: boolean; loadMore?: () => void
  convs: ConversationRow[]; convId: string | null; q: string; setQ: (q: string) => void
  open: (id: string) => void; refresh: () => void; onDeleted: (id: string) => void; toast: Ctx['toast']
}) {
  const [by, setBy] = useState<Grouping>(() => { try { return (localStorage.getItem(GROUP_KEY) as Grouping) || 'recent' } catch { return 'recent' } })
  const [folders, setFolders] = useState<ConversationFolder[]>([])
  const [menu, setMenu] = useState<string | null>(null)                    // the conversation whose menu is open
  const [ask, setAsk] = useState<{ kind: 'rename' | 'delete' | 'newfolder'; id: string } | { kind: 'folder-new' } | { kind: 'folder-rename' | 'folder-delete'; name: string } | null>(null)
  const loadFolders = () => api.conversationFolders().then((r) => setFolders(r.folders)).catch(() => {})
  useEffect(() => { loadFolders() }, [convs])
  useEffect(() => { try { localStorage.setItem(GROUP_KEY, by) } catch { /* no storage */ } }, [by])
  const box = useRef<HTMLDivElement>(null)
  useEffect(() => {                                                       // a click elsewhere or Escape closes the menu
    if (!menu) return
    const off = (e: MouseEvent) => { if (!box.current?.contains(e.target as Node)) setMenu(null) }
    const esc = (e: KeyboardEvent) => { if (e.key === 'Escape') setMenu(null) }
    addEventListener('mousedown', off); addEventListener('keydown', esc)
    return () => { removeEventListener('mousedown', off); removeEventListener('keydown', esc) }
  }, [menu])

  const meta = async (id: string, patch: { pinned?: boolean; folder?: string | null; title?: string | null }, said: string) => {
    setMenu(null); setAsk(null)
    try { await api.conversationSave(id, patch as never); toast(said); refresh(); loadFolders() } catch (e) { toast(String(e), 'bad') }
  }
  const remove = async (id: string) => {
    setAsk(null)
    try { await api.conversationDelete(id); onDeleted(id); refresh(); toast('Conversation deleted') } catch (e) { toast(String(e), 'bad') }
  }
  const folderAct = async (what: () => Promise<unknown>, said: string) => {
    setAsk(null)
    try { await what(); toast(said); loadFolders(); refresh() } catch (e) { toast(String(e), 'bad') }
  }

  const pinned = convs.filter((c) => c.pinned)
  const rest = q ? convs : convs.filter((c) => !c.pinned)                  // a search lists every match in its group
  const label = (key: string) => key || (by === 'folder' ? 'No folder' : by === 'model' ? 'No reply yet' : key)

  const row = (c: ConversationRow) => (
    <li key={c.id} className={c.id === convId ? 'active' : undefined}>
      <button type="button" className="open" onClick={() => open(c.id)} title={c.title}>
        <b>{c.pinned && <span className="pin" aria-label="Pinned">●</span>}{c.title}</b>
        <span className="small muted">{c.match ?? [by !== 'model' && c.model, by !== 'folder' && c.folder && `in ${c.folder}`, `${c.messages} messages`].filter(Boolean).join(' · ')}</span>
      </button>
      <button type="button" className="more" aria-label={`Actions for ${c.title}`} aria-expanded={menu === c.id} onClick={() => setMenu(menu === c.id ? null : c.id)}>⋯</button>
      {menu === c.id && <div className="convmenu" role="menu" ref={box}>
        <button role="menuitem" onClick={() => meta(c.id, { pinned: !c.pinned }, c.pinned ? 'Unpinned' : 'Pinned: it stays at the top')}>{c.pinned ? 'Unpin' : 'Pin to the top'}</button>
        <button role="menuitem" onClick={() => { setMenu(null); setAsk({ kind: 'rename', id: c.id }) }}>Rename</button>
        <label className="moveto">Move to
          <select className="sel sm" value={c.folder ?? ''} onChange={(e) => { const v = e.target.value; if (v === '\u0000new') { setMenu(null); setAsk({ kind: 'newfolder', id: c.id }) } else meta(c.id, { folder: v || null }, v ? `Moved to ${v}` : 'Out of its folder') }} aria-label="Move to a folder">
            <option value="">no folder</option>
            {folders.map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}
            <option value={'\u0000new'}>a new folder…</option>
          </select></label>
        <button role="menuitem" className="danger-text" onClick={() => { setMenu(null); setAsk({ kind: 'delete', id: c.id }) }}>Delete…</button>
      </div>}
      {ask && 'id' in ask && ask.id === c.id && ask.kind === 'rename' && <AskName text="New name" initial={c.title} ok="Rename" onOk={(t) => meta(c.id, { title: t }, 'Renamed')} onCancel={() => setAsk(null)} />}
      {ask && 'id' in ask && ask.id === c.id && ask.kind === 'newfolder' && <AskName text="Name of the new folder" initial="" ok="Make it and move" onOk={async (n) => { try { await api.conversationFolderMake(n) } catch (e) { toast(String(e), 'bad'); return } meta(c.id, { folder: n }, `Moved to ${n}`) }} onCancel={() => setAsk(null)} />}
      {ask && 'id' in ask && ask.id === c.id && ask.kind === 'delete' && <Ask danger yes="Delete it" onYes={() => remove(c.id)} onNo={() => setAsk(null)} text={<>Delete “{c.title}”? This cannot be undone.</>} />}
    </li>
  )

  return (
    <section className="convs" aria-label="Conversations">
      <input className="input" type="search" placeholder="Search conversations" value={q} onChange={(e) => setQ(e.target.value)} aria-label="Search conversations" />
      <div className="convs-head">
        <label className="small muted">Show <select className="sel sm" value={by} onChange={(e) => setBy(e.target.value as Grouping)} aria-label="Group conversations">
          <option value="recent">by date</option><option value="folder">by folder</option><option value="model">by model</option>
        </select></label>
        {by === 'folder' && <button type="button" className="btn sm quiet" onClick={() => setAsk({ kind: 'folder-new' })}>New folder</button>}
      </div>
      {ask?.kind === 'folder-new' && <AskName text="Name of the folder" initial="" ok="Make it" onOk={(n) => folderAct(() => api.conversationFolderMake(n), `Folder ${n} made`)} onCancel={() => setAsk(null)} />}
      {!q && pinned.length > 0 && <div className="convgroup"><div className="ghead">Pinned</div><ul>{pinned.map(row)}</ul></div>}
      {groupsOf(rest, by, q ? [] : folders).map(([key, list]) => (
        <div key={key || '-'} className="convgroup">
          <div className="ghead"><span>{by === 'model' && key && <ModelLogo names={[key]} size="xs" />}{label(key)}</span>
            {by === 'folder' && key && <span className="gacts">
              <button type="button" className="btn sm quiet" onClick={() => setAsk({ kind: 'folder-rename', name: key })} aria-label={`Rename the folder ${key}`}>rename</button>
              <button type="button" className="btn sm quiet" onClick={() => setAsk({ kind: 'folder-delete', name: key })} aria-label={`Delete the folder ${key}`}>delete</button></span>}</div>
          {ask?.kind === 'folder-rename' && ask.name === key && <AskName text="New name of the folder" initial={key} ok="Rename" onOk={(n) => folderAct(() => api.conversationFolderRename(key, n), 'Folder renamed')} onCancel={() => setAsk(null)} />}
          {ask?.kind === 'folder-delete' && ask.name === key && <Ask danger yes="Delete the folder" onYes={() => folderAct(() => api.conversationFolderRemove(key), 'Folder deleted; its conversations are kept')} onNo={() => setAsk(null)} text={<>Delete the folder “{key}”? Its {list.length} {list.length === 1 ? 'conversation stays' : 'conversations stay'}, in no folder.</>} />}
          <ul>{list.length ? list.map(row) : <li className="small muted empty">Empty. “⋯” on a conversation, then Move to.</li>}</ul>
        </div>
      ))}
      {!convs.length && <p className="small muted">{q ? 'No conversation matches.' : 'Conversations are kept here.'}</p>}
      {more && loadMore && <div className="actions"><button className="btn sm quiet" onClick={loadMore}>Show older conversations</button></div>}
    </section>
  )
}
