import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useMemo, useState } from 'react'
import clsx from 'clsx'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import { FileGlyph } from '../lib/fileicon'
import type { GitFileChange, GitStatus, WorkspaceLocation } from '../lib/types'

interface Node { name: string; path: string; dir: boolean; children: Node[]; status?: GitFileChange['status'] }

const STATUS_TINT: Record<string, string> = { A: 'text-positive', M: 'text-warn', D: 'text-danger', R: 'text-accent', '?': 'text-positive' }
const STATUS_CH: Record<string, string> = { get A() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "A") }, get M() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "M") }, get D() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "D") }, get R() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "R") }, get '?'() { return authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "A") } }

function buildTree(files: string[], statusByPath: Map<string, GitFileChange['status']>): Node[] {
  const root: Node[] = []
  const dirs = new Map<string, Node>()
  for (const f of files) {
    const parts = f.split('/')
    let level = root, cur = ''
    for (let i = 0; i < parts.length; i++) {
      cur = cur ? `${cur}/${parts[i]}` : parts[i]
      if (i === parts.length - 1) {
        level.push({ name: parts[i], path: f, dir: false, children: [], status: statusByPath.get(f) })
      } else {
        let node = dirs.get(cur)
        if (!node) { node = { name: parts[i], path: cur, dir: true, children: [] }; dirs.set(cur, node); level.push(node) }
        level = node.children
      }
    }
  }
  const sort = (nodes: Node[]) => { nodes.sort((a, b) => (a.dir === b.dir ? a.name.localeCompare(b.name) : a.dir ? -1 : 1)); nodes.forEach((n) => n.dir && sort(n.children)) }
  sort(root)
  return root
}

function Row({ node, depth, defaultOpen }: { node: Node; depth: number; defaultOpen: boolean }) {
  const tx = useAuthoredText()
  const [open, setOpen] = useState(depth < 1 || defaultOpen)
  const addFileMention = useStore((s) => s.addFileMention)
  const pad = { paddingLeft: 8 + depth * 12 }

  if (node.dir) {
    return (
      <div>
        <button onClick={() => setOpen((o) => !o)} style={pad} className="no-drag group flex w-full items-center gap-1.5 rounded-md py-1 pr-2 text-left text-[13px] text-ink-2 transition-colors hover:bg-ink-5">
          <Icon name={open ? 'chevronDown' : 'chevronRight'} size={12} className="shrink-0 text-ink-4" />
          <Icon name="folder" size={13} className="shrink-0 text-ink-3" />
          <span className="min-w-0 flex-1 truncate">{node.name}</span>
        </button>
        {open && node.children.map((c) => <Row key={c.path} node={c} depth={depth + 1} defaultOpen={defaultOpen} />)}
      </div>
    )
  }
  return (
    <div
      draggable
      onDragStart={(e) => e.dataTransfer.setData('text/plain', `@${node.path} `)}
      onClick={() => addFileMention(node.path)}
      title={tx("{value0} — click or drag into the composer", { value0: node.path })}
      style={pad}
      className="no-drag group flex cursor-pointer items-center gap-1.5 rounded-md py-1 pr-2 text-[13px] text-ink-2 transition-colors hover:bg-ink-5 hover:text-ink"
    >
      <span className="w-3 shrink-0" />
      <FileGlyph path={node.path} size={14} />
      <span className="min-w-0 flex-1 truncate">{node.name}</span>
      {node.status && <span className={clsx('shrink-0 font-mono text-[12px]', STATUS_TINT[node.status])}>{STATUS_CH[node.status]}</span>}
    </div>
  )
}

export function FileExplorer({ cwd, workspace = null }: { cwd: string; workspace?: WorkspaceLocation | null }) {
  const tx = useAuthoredText()
  const [files, setFiles] = useState<string[]>([])
  const [git, setGit] = useState<GitStatus | null>(null)
  const [query, setQuery] = useState('')
  const [changedOnly, setChangedOnly] = useState(false)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!cwd) { setFiles([]); setGit(null); return }
    let alive = true
    setLoading(true)
    Promise.all([api.listFiles(cwd, workspace), api.gitStatus(cwd, workspace).catch(() => null)])
      .then(([f, g]) => { if (alive) { setFiles(f); setGit(g) } })
      .finally(() => { if (alive) setLoading(false) })
    return () => { alive = false }
  }, [cwd, workspace])

  const statusByPath = useMemo(() => {
    const m = new Map<string, GitFileChange['status']>()
    for (const f of git?.files ?? []) m.set(f.path, f.status)
    return m
  }, [git, tx])

  const tree = useMemo(() => {
    const q = query.trim().toLowerCase()
    let src = changedOnly ? (git?.files.map((f) => f.path) ?? []) : files
    if (q) src = src.filter((f) => f.toLowerCase().includes(q))
    return buildTree(src, statusByPath)
  }, [files, git, query, changedOnly, statusByPath, tx])

  const changedCount = git?.files.length ?? 0

  if (!cwd) return <div className="px-4 py-6 text-center text-[13px] text-ink-3">{tx("Open a task or project with a folder to browse its files.")}</div>

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="space-y-1.5 px-3 pb-2">
        <div className="flex items-center gap-2 rounded-lg bg-ink-5/60 px-2.5 py-1.5">
          <Icon name="search" size={14} className="shrink-0 text-ink-3" />
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder={tx("Filter files")} className="w-full bg-transparent text-[13px] text-ink outline-none placeholder:text-ink-3" />
          {query && <button onClick={() => setQuery('')} className="text-ink-3 hover:text-ink"><Icon name="x" size={13} /></button>}
        </div>
        <button
          onClick={() => setChangedOnly((v) => !v)}
          className={clsx('flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-[13px] transition-colors', changedOnly ? 'bg-accent-soft text-accent' : 'text-ink-2 hover:bg-ink-5')}
        >
          <Icon name="branch" size={13} className={changedOnly ? 'text-accent' : 'text-ink-3'} />
          {tx("Changed files only")}<span className={clsx('ml-auto rounded px-1.5 text-[12px] tabular-nums', changedOnly ? 'bg-accent/15' : 'bg-ink-5 text-ink-3')}>{changedCount}</span>
        </button>
      </div>
      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-2 pb-3">
        {loading && files.length === 0 ? (
          <div className="px-3 py-4 text-[13px] text-ink-3">{tx("Loading…")}</div>
        ) : tree.length === 0 ? (
          <div className="px-3 py-4 text-[13px] text-ink-3">{changedOnly ? tx("No changes.") : tx("No files.")}</div>
        ) : (
          tree.map((n) => <Row key={n.path} node={n} depth={0} defaultOpen={changedOnly} />)
        )}
      </div>
    </div>
  )
}
