import { useEffect, useState } from 'react'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import type { ProjectMemoryCandidate, ProjectMemoryCandidateDiff, ProjectMemoryEntry, ProjectMemoryKind, ProjectMemoryConfidence, ProjectMemoryPrivacyScope } from '../lib/types'

const KINDS: ProjectMemoryKind[] = ['architecture', 'convention', 'api', 'decision', 'gotcha', 'summary', 'person', 'preference']
const CONF: ProjectMemoryConfidence[] = ['low', 'medium', 'high']
const SCOPES: ProjectMemoryPrivacyScope[] = ['project', 'personal', 'team']
const CONF_TINT: Record<ProjectMemoryConfidence, string> = { low: 'text-ink-4', medium: 'text-ink-3', high: 'text-positive' }

type MemoryDraft = {
  id?: string
  kind: ProjectMemoryKind
  text: string
  confidence: ProjectMemoryConfidence
  privacyScope: ProjectMemoryPrivacyScope
}

// Typed, source-attributed project memory (build.md P1 / Pillar 6). Durable facts an
// agent reuses across runs — distinct from the freeform "Shared memory" charter blob.
export function MemoryEntries({ projectId }: { projectId: string }) {
  const [entries, setEntries] = useState<ProjectMemoryEntry[]>([])
  const [candidates, setCandidates] = useState<ProjectMemoryCandidate[]>([])
  const [diffs, setDiffs] = useState<Record<string, ProjectMemoryCandidateDiff>>({})
  const [draft, setDraft] = useState<MemoryDraft | null>(null)

  const reload = () => {
    api.getProjectMemory(projectId).then(setEntries).catch(() => {})
    api.getProjectMemoryCandidates(projectId).then((list) => {
      setCandidates(list)
      // Informed review (VST-009): show what each approval would DO — especially the
      // before/after when a candidate overwrites an existing entry.
      for (const candidate of list) {
        api.memoryCandidateDiff(candidate.id)
          .then((diff) => setDiffs((prev) => ({ ...prev, [candidate.id]: diff })))
          .catch(() => {})
      }
    }).catch(() => {})
  }
  useEffect(reload, [projectId])

  const rollback = async (entry: ProjectMemoryEntry) => {
    const hasRevisions = (entry.revisions?.length ?? 0) > 0
    const what = hasRevisions ? 'restore its previous version' : 'remove it (its creation was the last change)'
    if (!confirm(`Roll back this memory? This will ${what}. The linked approval is marked rolled back.`)) return
    await api.rollbackMemoryEntry(entry.id).catch(() => {})
    reload()
  }

  const save = async () => {
    if (!draft?.text.trim()) return
    await api.saveProjectMemory({
      id: draft.id,
      projectId,
      kind: draft.kind,
      text: draft.text.trim(),
      confidence: draft.confidence,
      privacyScope: draft.privacyScope,
      ...(draft.id ? {} : { sourceType: 'manual' as const }),
    })
    setDraft(null); reload()
  }
  const editEntry = (entry: ProjectMemoryEntry) => {
    setDraft({ id: entry.id, kind: entry.kind, text: entry.text, confidence: entry.confidence, privacyScope: entry.privacyScope ?? 'project' })
  }

  return (
    <div className="mt-4">
      <div className="mb-1.5 flex items-center justify-between">
        <div className="text-[13px] font-medium text-ink-2">Memory entries <span className="text-ink-4">· typed facts agents reuse</span></div>
        {!draft && (
          <button onClick={() => setDraft({ kind: 'convention', text: '', confidence: 'medium', privacyScope: 'project' })} className="flex items-center gap-1 rounded-md border border-line px-2 py-1 text-[13px] text-ink-2 transition-colors hover:border-line-strong hover:text-ink">
            <Icon name="plus" size={12} />Add
          </button>
        )}
      </div>

      {candidates.length > 0 && (
        <div className="mb-2 space-y-1 rounded-lg border border-warn/35 bg-card px-2.5 py-2">
          <div className="mb-1 flex items-center gap-1.5 text-[13px] font-medium text-ink-3">
            <Icon name="shield" size={12} /> Memory candidates
            <span className="text-ink-4">· approve before durable write</span>
          </div>
          {candidates.map((candidate) => {
            const diff = diffs[candidate.id]
            const textChange = diff?.action === 'update' ? diff.changes.find((c) => c.field === 'text') : null
            return (
              <div key={candidate.id} className="flex items-start gap-2 rounded-md border border-line bg-card-2 px-2 py-1.5">
                <span className="mt-0.5 shrink-0 font-mono text-[11px] uppercase tracking-wide text-ink-4">{candidate.kind}</span>
                <span className="min-w-0 flex-1">
                  {/* An UPDATE shows the exact overwrite (before struck through); a create shows the new text. */}
                  {textChange ? (
                    <>
                      <span className="block text-[13px] text-ink-4 line-through decoration-danger/50">{textChange.before}</span>
                      <span className="block text-[13px] text-ink-2">{textChange.after}</span>
                    </>
                  ) : (
                    <span className="block text-[13px] text-ink-2">{candidate.text}</span>
                  )}
                  {diff?.action === 'update' && diff.changes.filter((c) => c.field !== 'text').length > 0 && (
                    <span className="mt-0.5 block text-[12px] text-warn">
                      also changes {diff.changes.filter((c) => c.field !== 'text').map((c) => `${c.field}: ${c.before ?? '—'} → ${c.after ?? '—'}`).join(', ')}
                    </span>
                  )}
                  <span className="mt-0.5 block truncate text-[12px] text-ink-4">
                    {diff ? (diff.action === 'update' ? 'updates existing memory' : 'new memory') : candidate.sourceType}
                    {' - '}scope:{candidate.privacyScope ?? 'project'}
                    {candidate.sourceThreadId ? ` - from turn ${candidate.sourceTurn ?? '?'}` : ''}
                    {candidate.sourceRef ? ` - ${candidate.sourceRef}` : ''}{candidate.reason ? ` - ${candidate.reason}` : ''}
                  </span>
                </span>
                <button onClick={() => api.approveProjectMemoryCandidate(candidate.id).then(reload)} className="shrink-0 rounded-md bg-positive/15 px-2 py-1 text-[12px] font-medium text-positive hover:bg-positive/25">Approve</button>
                <button onClick={() => api.rejectProjectMemoryCandidate(candidate.id).then(reload)} className="shrink-0 rounded-md border border-line px-2 py-1 text-[12px] text-ink-3 hover:text-danger">Reject</button>
              </div>
            )
          })}
        </div>
      )}

      <div className="space-y-1">
        {entries.map((m) => (
          <div key={m.id} className="group flex items-start gap-2 rounded-lg border border-line bg-card px-2.5 py-1.5">
            <span className="mt-0.5 shrink-0 font-mono text-[11px] uppercase tracking-wide text-ink-4">{m.kind}</span>
            <span className="min-w-0 flex-1">
              <span className="block text-[13px] text-ink-2">{m.text}</span>
              <span className="mt-0.5 block truncate text-[12px] text-ink-4">
                {m.sourceType} - scope:{m.privacyScope ?? 'project'}{m.sourceRef ? ` - ${m.sourceRef}` : ''}{m.updatedAt ? ` - updated ${new Date(m.updatedAt).toLocaleDateString()}` : ''}
              </span>
            </span>
            <span className={`shrink-0 text-[11px] ${CONF_TINT[m.confidence]}`} title={`confidence: ${m.confidence}${m.sourceRef ? ` · src ${m.sourceRef}` : ''}`}>{m.confidence}</span>
            {(m.revisions?.length ?? 0) > 0 && (
              <span className="shrink-0 rounded bg-ink-5 px-1 text-[11px] tabular-nums text-ink-4" title={`${m.revisions!.length} previous version${m.revisions!.length === 1 ? '' : 's'} kept for rollback`}>{m.revisions!.length} rev</span>
            )}
            <button
              onClick={() => void rollback(m)}
              title={(m.revisions?.length ?? 0) > 0 ? 'Roll back to the previous version' : 'Roll back — removes this entry (its creation was the last change)'}
              className="shrink-0 text-ink-4 opacity-0 transition-all hover:text-warn group-hover:opacity-100"
              aria-label="Roll back memory"
            >
              <Icon name="refresh" size={13} />
            </button>
            <button onClick={() => editEntry(m)} disabled={draft?.id === m.id} className="shrink-0 text-ink-4 opacity-0 transition-all hover:text-ink group-hover:opacity-100 disabled:opacity-40" aria-label="Edit memory"><Icon name="edit" size={13} /></button>
            <button onClick={() => api.deleteProjectMemory(m.id).then(reload)} className="shrink-0 text-ink-4 opacity-0 transition-all hover:text-danger group-hover:opacity-100" aria-label="Delete memory"><Icon name="trash" size={13} /></button>
          </div>
        ))}
        {entries.length === 0 && !draft && <div className="rounded-lg border border-dashed border-line px-2.5 py-2 text-[13px] text-ink-4">No typed memory yet — add architecture, conventions, APIs, decisions, or gotchas.</div>}
      </div>

      {draft && (
        <div className="mt-2 space-y-2 rounded-lg border border-accent/40 bg-card p-2.5">
          <div className="text-[13px] font-medium text-ink-3">{draft.id ? 'Edit memory entry' : 'Add memory entry'}</div>
          <div className="flex gap-2">
            <select value={draft.kind} onChange={(e) => setDraft({ ...draft, kind: e.target.value as ProjectMemoryKind })} className="rounded-md border border-line bg-card-2 px-2 py-1 text-[13px] text-ink outline-none">
              {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
            </select>
            <select value={draft.confidence} onChange={(e) => setDraft({ ...draft, confidence: e.target.value as ProjectMemoryConfidence })} className="rounded-md border border-line bg-card-2 px-2 py-1 text-[13px] text-ink outline-none">
              {CONF.map((c) => <option key={c} value={c}>{c} confidence</option>)}
            </select>
            <select value={draft.privacyScope} onChange={(e) => setDraft({ ...draft, privacyScope: e.target.value as ProjectMemoryPrivacyScope })} className="rounded-md border border-line bg-card-2 px-2 py-1 text-[13px] text-ink outline-none">
              {SCOPES.map((scope) => <option key={scope} value={scope}>{scope} scope</option>)}
            </select>
          </div>
          <textarea autoFocus value={draft.text} onChange={(e) => setDraft({ ...draft, text: e.target.value })} rows={2} placeholder="e.g. All IPC goes through the handle() wrapper which injects the sender trust check." className="scroll-thin w-full resize-none rounded-md border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink outline-none focus:border-accent" />
          <div className="flex justify-end gap-2">
            <button onClick={() => setDraft(null)} className="rounded-md px-2 py-1 text-[13px] text-ink-3 hover:text-ink">Cancel</button>
            <button onClick={save} className="rounded-md bg-accent px-2.5 py-1 text-[13px] text-accent-ink hover:bg-accent-strong">Save</button>
          </div>
        </div>
      )}
    </div>
  )
}
