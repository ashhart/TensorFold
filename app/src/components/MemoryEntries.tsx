import { currentIntlLocale } from '../lib/preferences/current-locale'
import { OriginalDiagnostic } from './OriginalDiagnostic'
import { authoredText } from '../lib/preferences/language-packs'
import { useStore as useLocaleStore } from '../lib/store'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import type { ProjectMemoryCandidate, ProjectMemoryCandidateDiff, ProjectMemoryEntry, ProjectMemoryKind, ProjectMemoryConfidence, ProjectMemoryPrivacyScope } from '../lib/types'

const KINDS: ProjectMemoryKind[] = ['architecture', 'convention', 'api', 'decision', 'gotcha', 'summary', 'person', 'preference']
const CONF: ProjectMemoryConfidence[] = ['low', 'medium', 'high']
const SCOPES: ProjectMemoryPrivacyScope[] = ['project', 'personal', 'team']
const BUTTON = 'rounded-md border border-line px-2 py-1 text-[12px] text-ink-3 hover:border-line-strong hover:text-ink focus-visible:outline focus-visible:outline-accent disabled:opacity-40'
type Owner = { id: string; live: boolean }
type MemoryDraft = { owner: Owner; projectId: string; id?: string; kind: ProjectMemoryKind; text: string; confidence: ProjectMemoryConfidence; privacyScope: ProjectMemoryPrivacyScope }
type Snapshot = { owner: Owner; entries: ProjectMemoryEntry[]; trash: ProjectMemoryEntry[]; candidates: ProjectMemoryCandidate[]; diffs: Record<string, ProjectMemoryCandidateDiff> }
const message = (error: unknown) => error instanceof Error ? error.message : authoredText(useLocaleStore.getState().settings?.uiPreferences?.locale, "The memory change could not be completed.")

// Durable typed facts, separate from the user-authored freeform project charter.
export function MemoryEntries({ projectId }: { projectId: string }) {
  const tx = useAuthoredText()
  const ownerRef = useRef<Owner>({ id: projectId, live: true })
  if (ownerRef.current.id !== projectId) ownerRef.current = { id: projectId, live: true }
  const owner = ownerRef.current
  const sequence = useRef(0)
  const actionRef = useRef<{ owner: Owner; key: string } | null>(null)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [draftState, setDraft] = useState<MemoryDraft | null>(null)
  const [tabState, setTab] = useState<{ owner: Owner; value: 'active' | 'trash' } | null>(null)
  const [pending, setPending] = useState<{ owner: Owner; key: string } | null>(null)
  const [error, setError] = useState<{ owner: Owner; text: string } | null>(null)
  const [notice, setNotice] = useState<{ owner: Owner; text: string } | null>(null)
  const isCurrent = (scope: Owner) => scope.live && ownerRef.current === scope
  const remote = api.platform === 'web-remote'
  const data = snapshot?.owner === owner ? snapshot : null
  const draft = draftState?.owner === owner ? draftState : null
  const tab = tabState?.owner === owner ? tabState.value : 'active'
  const busy = pending?.owner === owner

  async function reload(scope = owner): Promise<boolean> {
    if (!isCurrent(scope) || remote) return false
    const request = ++sequence.current
    try {
      const [active, trash, candidates] = await Promise.all([api.getProjectMemory(scope.id), api.getProjectMemoryTrash(scope.id), api.getProjectMemoryCandidates(scope.id)])
      const visible = candidates.filter((item) => item.projectId === scope.id)
      const diffResults = await Promise.allSettled(visible.map((item) => api.memoryCandidateDiff(item.id)))
      if (!isCurrent(scope) || request !== sequence.current) return false
      const diffs: Record<string, ProjectMemoryCandidateDiff> = {}
      diffResults.forEach((result, index) => { if (result.status === 'fulfilled') diffs[visible[index].id] = result.value })
      setSnapshot({ owner: scope, entries: active.filter((item) => item.projectId === scope.id && item.trashedAt == null), trash: trash.filter((item) => item.projectId === scope.id && item.trashedAt != null), candidates: visible, diffs })
      setError(diffResults.some((result) => result.status === 'rejected') ? { owner: scope, text: 'Some candidate previews could not be loaded; retry before approving them.' } : null)
      return true
    } catch (failure) {
      if (isCurrent(scope) && request === sequence.current) setError({ owner: scope, text: message(failure) })
      return false
    }
  }
  useEffect(() => {
    owner.live = true
    void reload(owner)
    return () => { owner.live = false }
  }, [projectId])

  async function run(key: string, change: () => Promise<unknown>, success: string, after?: () => void): Promise<void> {
    const scope = owner
    if (remote || !isCurrent(scope) || actionRef.current?.owner === scope) return
    const token = { owner: scope, key }
    actionRef.current = token; setPending(token); setError(null); setNotice(null)
    try {
      await change()
      if (!isCurrent(scope)) return
      setSnapshot(null)
      after?.()
      setNotice({ owner: scope, text: success })
      await reload(scope)
    } catch (failure) {
      if (isCurrent(scope)) setError({ owner: scope, text: message(failure) })
    } finally {
      if (actionRef.current === token) actionRef.current = null
      if (isCurrent(scope)) setPending((value) => value === token ? null : value)
    }
  }
  const lifecycle = (entry: ProjectMemoryEntry) => ({ projectId, entryId: entry.id, expectedRevision: entry.stateRevision ?? 1 })
  function trash(entry: ProjectMemoryEntry) {
    void run(`trash:${entry.id}`, () => api.trashProjectMemory(lifecycle(entry)), 'Moved to Trash; you can restore it below.', () => { if (draft?.id === entry.id) setDraft(null) })
  }
  function forget(entry: ProjectMemoryEntry) {
    if (!confirm(tx("Forget this memory permanently? Its saved text, previous versions and linked review records will be deleted. This cannot be undone; earlier chats and learned model weights are unchanged."))) return
    void run(`forget:${entry.id}`, () => api.deleteProjectMemory(entry.id, { projectId, expectedRevision: entry.stateRevision ?? 1 }), 'Memory permanently forgotten.')
  }
  function rollback(entry: ProjectMemoryEntry) {
    const what = entry.revisions?.length ? 'restore its previous version' : 'remove this entry because its creation was the last change'
    if (!confirm(tx("Roll back this memory? This will {value0}. The linked approval is marked rolled back.", { value0: what }))) return
    void run(`rollback:${entry.id}`, () => api.rollbackMemoryEntry(entry.id), 'Memory rolled back.', () => { if (draft?.id === entry.id) setDraft(null) })
  }
  function save() {
    if (!draft?.text.trim()) return
    const input = { id: draft.id, projectId, kind: draft.kind, text: draft.text.trim(), confidence: draft.confidence, privacyScope: draft.privacyScope, ...(draft.id ? {} : { sourceType: 'manual' as const }) }
    void run('save', () => api.saveProjectMemory(input), 'Memory saved.', () => setDraft(null))
  }
  if (remote) return <div className="mt-4 rounded-lg border border-line p-3 text-[13px] text-ink-3">{tx("Manage project memory in the desktop app on its owning host.")}</div>

  return (
    <section className="mt-4" aria-label={tx("Project memory")} aria-busy={busy}>
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-[13px] font-medium text-ink-2">{tx("Project memory ")}<span className="font-normal text-ink-4">{tx("· facts agents reuse")}</span></div>
        {!draft && <button disabled={busy || !data} onClick={() => { setTab({ owner, value: 'active' }); setDraft({ owner, projectId, kind: 'convention', text: '', confidence: 'medium', privacyScope: 'project' }) }} className={BUTTON}><span className="flex items-center gap-1"><Icon name="plus" size={12} />{tx("Add memory")}</span></button>}
      </div>
      <div className="my-2 flex gap-1" role="tablist" aria-label={tx("Memory state")}>
        <button role="tab" aria-selected={tab === 'active'} onClick={() => setTab({ owner, value: 'active' })} className={`${BUTTON} ${tab === 'active' ? 'bg-card-2 text-ink' : ''}`}>{tx("Active")}{data ? ` (${data.entries.length})` : ''}</button>
        <button role="tab" aria-selected={tab === 'trash'} onClick={() => setTab({ owner, value: 'trash' })} className={`${BUTTON} ${tab === 'trash' ? 'bg-card-2 text-ink' : ''}`}>{tx("Trash")}{data ? ` (${data.trash.length})` : ''}</button>
      </div>
      <p className="mb-2 text-[12px] leading-relaxed text-ink-4">{tx("Trash excludes a fact from future project runs; you can restore it at any time. Earlier chats, context already in use and learned model weights are unchanged.")}</p>
      {error?.owner === owner && <div role="alert" className="mb-2 rounded-lg border border-danger/35 bg-card px-2.5 py-2 text-[13px] text-danger">{<OriginalDiagnostic>{error.text}</OriginalDiagnostic>}<button disabled={busy} onClick={() => void reload()} className={`${BUTTON} ml-2`}>{tx("Retry")}</button></div>}
      {notice?.owner === owner && <p role="status" className="mb-2 text-[12px] text-positive">{notice.text}</p>}
      {!data && error?.owner !== owner && <p role="status" className="text-[13px] text-ink-4">{tx("Loading memory…")}</p>}
      {data && tab === 'active' && data.candidates.length > 0 && <div className="mb-2 space-y-1 rounded-lg border border-warn/35 bg-card p-2.5">
        <div className="mb-1 flex items-center gap-1.5 text-[13px] font-medium text-ink-3"><Icon name="shield" size={12} />{tx("Review suggested facts")}</div>
        {data.candidates.map((candidate) => {
          const diff = data.diffs[candidate.id], change = diff?.changes.find((item) => item.field === 'text')
          return <div key={candidate.id} className="rounded-md border border-line bg-card-2 p-2">
            {diff?.action === 'update' && change && <p className="text-[13px] text-ink-4 line-through decoration-danger/50">{change.before}</p>}
            <p className="whitespace-pre-wrap break-words text-[13px] text-ink-2">{candidate.text}</p>
            {diff?.action === 'update' && diff.changes.some((item) => item.field !== 'text') && <p className="text-[12px] text-warn">{tx("Also changes ")}{diff.changes.filter((item) => item.field !== 'text').map((item) => `${item.field}: ${item.before ?? '—'} → ${item.after ?? '—'}`).join(', ')}</p>}
            <p className="mt-1 break-words text-[12px] text-ink-4">{diff ? diff.action === 'update' ? tx("Updates existing memory") : tx("New memory") : tx("Preview unavailable")} · {candidate.privacyScope ?? tx("project")} {tx("scope")}{candidate.sourceRef ? ` · ${candidate.sourceRef}` : ''}{candidate.sourceThreadId ? tx(" · turn {value0}", { value0: candidate.sourceTurn ?? '?' }) : ''}{candidate.reason ? ` · ${candidate.reason}` : ''}</p>
            <div className="mt-2 flex gap-2">
              <button disabled={busy || !diff} onClick={() => void run(`approve:${candidate.id}`, () => api.approveProjectMemoryCandidate(candidate.id), 'Suggested fact approved.')} className={`${BUTTON} text-positive`}>{tx("Approve")}</button>
              <button disabled={busy} onClick={() => void run(`reject:${candidate.id}`, () => api.rejectProjectMemoryCandidate(candidate.id), 'Suggested fact rejected.')} className={BUTTON}>{tx("Reject")}</button>
            </div>
          </div>
        })}
      </div>}
      <div className="space-y-2" role="tabpanel" aria-label={tab === 'active' ? tx("Active memory") : tx("Trashed memory")}>
        {(tab === 'active' ? data?.entries : data?.trash)?.map((entry) => <article key={entry.id} className="rounded-lg border border-line bg-card p-2.5" aria-label={tx("Memory: {value0}", { value0: entry.kind })}>
          <p className="whitespace-pre-wrap break-words text-[13px] leading-relaxed text-ink-2">{entry.text}</p>
          <p className="mt-1 break-words text-[12px] text-ink-4">{entry.kind} · {entry.confidence} {tx("confidence · ")}{entry.privacyScope ?? tx("project")} {tx("scope · ")}{entry.sourceType}{entry.sourceRef ? ` · ${entry.sourceRef}` : ''}{entry.updatedAt ? tx(" · updated {value0}", { value0: new Date(entry.updatedAt).toLocaleDateString(currentIntlLocale()) }) : ''}{entry.revisions?.length ? tx(" · {value0} previous versions", { value0: entry.revisions.length }) : ''}{tab === 'trash' && typeof entry.trashedAt === 'number' && Number.isFinite(entry.trashedAt) ? tx(" · trashed {value0}", { value0: new Date(entry.trashedAt).toLocaleDateString(currentIntlLocale()) }) : ''}</p>
          <div className="mt-2 flex flex-wrap gap-2">
            {tab === 'active' ? <>
              <button disabled={busy || draft?.id === entry.id} onClick={() => setDraft({ owner, projectId, id: entry.id, kind: entry.kind, text: entry.text, confidence: entry.confidence, privacyScope: entry.privacyScope ?? 'project' })} className={BUTTON}>{tx("Edit")}</button>
              <button disabled={busy} onClick={() => rollback(entry)} className={BUTTON}>{tx("Roll back")}</button>
              <button disabled={busy} onClick={() => trash(entry)} className={BUTTON}>{tx("Move to Trash")}</button>
            </> : <>
              <button disabled={busy} onClick={() => void run(`restore:${entry.id}`, () => api.restoreProjectMemory(lifecycle(entry)), 'Memory restored.')} className={`${BUTTON} text-positive`}>{tx("Restore")}</button>
              <button disabled={busy} onClick={() => forget(entry)} className={`${BUTTON} hover:text-danger`}>{tx("Forget permanently")}</button>
            </>}
          </div>
        </article>)}
        {data && (tab === 'active' ? data.entries : data.trash).length === 0 && <div className="rounded-lg border border-dashed border-line p-2.5 text-[13px] text-ink-4">{tab === 'active' ? tx("No active facts yet; add a convention, decision or something useful to remember.") : tx("Trash is empty.")}</div>}
      </div>
      {draft && tab === 'active' && <div className="mt-2 space-y-2 rounded-lg border border-accent/40 bg-card p-2.5">
        <div className="text-[13px] font-medium text-ink-3">{draft.id ? tx("Edit memory entry") : tx("Add memory entry")}</div>
        <div className="flex flex-wrap gap-2">
          <select disabled={busy} aria-label={tx("Memory kind")} value={draft.kind} onChange={(event) => setDraft({ ...draft, kind: event.target.value as ProjectMemoryKind })} className={BUTTON}>{KINDS.map((kind) => <option key={kind} value={kind}>{kind}</option>)}</select>
          <select disabled={busy} aria-label={tx("Memory confidence")} value={draft.confidence} onChange={(event) => setDraft({ ...draft, confidence: event.target.value as ProjectMemoryConfidence })} className={BUTTON}>{CONF.map((confidence) => <option key={confidence} value={confidence}>{confidence} {tx("confidence")}</option>)}</select>
          <select disabled={busy} aria-label={tx("Memory scope")} value={draft.privacyScope} onChange={(event) => setDraft({ ...draft, privacyScope: event.target.value as ProjectMemoryPrivacyScope })} className={BUTTON}>{SCOPES.map((scope) => <option key={scope} value={scope}>{scope} {tx("scope")}</option>)}</select>
        </div>
        <textarea autoFocus disabled={busy} aria-label={tx("Memory text")} value={draft.text} onChange={(event) => setDraft({ ...draft, text: event.target.value })} rows={3} placeholder={tx("A useful convention or decision for this project…")} className="scroll-thin w-full resize-y rounded-md border border-line bg-card-2 p-2 text-[13px] text-ink outline-none focus:border-accent" />
        <div className="flex justify-end gap-2"><button disabled={busy} onClick={() => setDraft(null)} className={BUTTON}>{tx("Cancel")}</button><button disabled={busy || !draft.text.trim()} onClick={save} className={`${BUTTON} bg-accent text-accent-ink`}>{pending?.key === 'save' && busy ? tx("Saving…") : tx("Save")}</button></div>
      </div>}
    </section>
  )
}
