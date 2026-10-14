import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import { Button, Input, Toggle } from './ui'
import type { Skill, Command, SkillSuggestion, SkillSuggestionDiff } from '../lib/types'

// Settings management for $ skills and / custom commands.
export function SkillsCommands() {
  const [skills, setSkills] = useState<Skill[]>([])
  const [commands, setCommands] = useState<Command[]>([])
  const [suggestions, setSuggestions] = useState<SkillSuggestion[]>([])
  const [editSkill, setEditSkill] = useState<Partial<Skill> | null>(null)
  const [editCmd, setEditCmd] = useState<Partial<Command> | null>(null)

  const [diffs, setDiffs] = useState<Record<string, SkillSuggestionDiff>>({})
  const reload = () => { api.getSkills().then(setSkills); api.getCommands().then(setCommands); api.getSuggestions().then(setSuggestions) }
  // What accepting each pending suggestion would DO (VST-010) — diffs for update proposals.
  useEffect(() => {
    for (const s of suggestions.filter((x) => x.status === 'pending' && x.targetSkillId)) {
      if (diffs[s.id]) continue
      api.suggestionDiff(s.id).then((diff) => setDiffs((prev) => ({ ...prev, [s.id]: diff }))).catch(() => {})
    }
  }, [suggestions])
  useEffect(reload, [])
  // Live-update when the agent files a new suggestion.
  useEffect(() => api.onSuggestionsUpdate(setSuggestions), [])

  const saveSkill = async () => { if (!editSkill) return; await api.saveSkill(editSkill); setEditSkill(null); reload() }
  const saveCmd = async () => { if (!editCmd) return; await api.saveCommand(editCmd); setEditCmd(null); reload() }
  const pending = suggestions.filter((s) => s.status === 'pending')
  const acceptSuggestion = async (id: string) => { await api.acceptSuggestion(id); reload() }
  const dismissSuggestion = async (id: string) => { await api.dismissSuggestion(id); reload() }

  return (
    <div className="space-y-6">
      {/* Suggestions: things the assistant noticed are worth saving. You decide. */}
      {pending.length > 0 && (
        <div>
          <div className="mb-2">
            <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><Icon name="spark" size={14} className="text-accent" />Suggestions</div>
            <div className="text-[13px] text-ink-3">Reusable workflows TensorFold noticed. Nothing is saved until you accept.</div>
          </div>
          <div className="space-y-1.5">
            {pending.map((s) => (
              <div key={s.id} className="rounded-xl border border-line bg-card p-3 shadow-card">
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0 rounded-md bg-ink-5 px-1.5 py-0.5 text-[11px] font-semibold uppercase tracking-wide text-ink-3">{s.targetSkillId ? 'update' : s.kind}</span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-[13px] font-medium text-ink">{s.kind === 'command' ? '/' : ''}{s.name}</div>
                    {s.description && <div className="truncate text-[13px] text-ink-2">{s.description}</div>}
                    {s.rationale && <div className="mt-0.5 text-[13px] leading-snug text-ink-3">{s.rationale}</div>}
                    {/* Informed review (VST-010): an in-place improvement shows exactly what
                        accepting overwrites — old body struck through, new below. */}
                    {diffs[s.id]?.action === 'update' && diffs[s.id].changes.filter((c) => c.field === 'body').map((c) => (
                      <div key="body-diff" className="mt-1.5 rounded-md border border-line bg-card-2 p-2">
                        <div className="mb-0.5 text-[11px] font-semibold uppercase tracking-wide text-warn">replaces the current instructions</div>
                        <div className="max-h-16 overflow-y-auto scroll-thin whitespace-pre-wrap text-[12px] leading-snug text-ink-4 line-through decoration-danger/50">{c.before}</div>
                        <div className="mt-1 max-h-24 overflow-y-auto scroll-thin whitespace-pre-wrap text-[13px] leading-snug text-ink-2">{c.after}</div>
                      </div>
                    ))}
                  </div>
                </div>
                <div className="mt-2 flex items-center gap-1.5">
                  <Button variant="primary" size="sm" icon="check" onClick={() => acceptSuggestion(s.id)}>Save {s.kind}</Button>
                  <button onClick={() => { setEditSkill(null); s.kind === 'command' ? setEditCmd({ name: s.name, description: s.description, prompt: s.body, argHint: s.argHint }) : setEditSkill({ name: s.name, description: s.description, body: s.body, enabled: true }); dismissSuggestion(s.id) }} className="rounded-lg border border-line px-2.5 py-1 text-[13px] font-medium text-ink-2 transition-colors hover:bg-surface-hover">Edit first</button>
                  <button onClick={() => dismissSuggestion(s.id)} className="rounded-lg px-2 py-1 text-[13px] text-ink-3 transition-colors hover:text-danger">Dismiss</button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
      {/* Skills */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><span className="font-mono text-accent">$</span>Skills</div>
            <div className="text-[13px] text-ink-3">Reusable instructions you invoke with <span className="font-mono">$</span> in the composer.</div>
          </div>
          <Button variant="default" size="sm" icon="plus" onClick={() => setEditSkill({ name: '', description: '', body: '', enabled: true })}>New skill</Button>
        </div>
        <div className="space-y-1.5">
          {skills.map((s) => (
            <div key={s.id} className="group flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><span className="font-mono text-ink-3">${s.name}</span></div>
                <div className="truncate text-[13px] text-ink-3">{s.description || s.body.slice(0, 80)}</div>
              </div>
              <Toggle label={`Enable ${s.name} skill`} checked={s.enabled} onChange={(v) => api.saveSkill({ id: s.id, enabled: v }).then(reload)} />
              <button onClick={() => setEditSkill({ id: s.id, name: s.name, description: s.description, body: s.body, enabled: s.enabled, trigger: s.trigger })} className="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"><Icon name="edit" size={14} /></button>
              <button onClick={() => api.deleteSkill(s.id).then(reload)} className="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 opacity-0 transition-all hover:text-danger group-hover:opacity-100"><Icon name="trash" size={14} /></button>
            </div>
          ))}
          {skills.length === 0 && !editSkill && <div className="rounded-lg border border-dashed border-line px-3 py-4 text-center text-[13px] text-ink-3">No skills yet.</div>}
        </div>
        {editSkill && (
          <Editor onCancel={() => setEditSkill(null)} onSave={saveSkill} title={editSkill.id ? 'Edit skill' : 'New skill'}>
            <Input value={editSkill.name || ''} onChange={(e) => setEditSkill({ ...editSkill, name: e.target.value })} placeholder="Name (e.g. code-review)" />
            <Input value={editSkill.description || ''} onChange={(e) => setEditSkill({ ...editSkill, description: e.target.value })} placeholder="Short description" />
            <Input value={editSkill.trigger || ''} onChange={(e) => setEditSkill({ ...editSkill, trigger: e.target.value })} placeholder="Use when… (helps the agent pick this skill at the right moment)" />
            <textarea value={editSkill.body || ''} onChange={(e) => setEditSkill({ ...editSkill, body: e.target.value })} rows={4} placeholder="Instructions. Markdown headings ## References, ## Scripts, ## Templates, ## Examples, ## Tests become structured sections." className="scroll-thin w-full resize-none rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink outline-none focus:border-accent" />
          </Editor>
        )}
      </div>

      {/* Commands */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <div>
            <div className="flex items-center gap-1.5 text-[13px] font-medium text-ink"><span className="font-mono text-accent">/</span>Custom commands</div>
            <div className="text-[13px] text-ink-3">Saved prompts you run with <span className="font-mono">/name</span> in the composer.</div>
          </div>
          <Button variant="default" size="sm" icon="plus" onClick={() => setEditCmd({ name: '', description: '', prompt: '' })}>New command</Button>
        </div>
        <div className="space-y-1.5">
          {commands.map((c) => (
            <div key={c.id} className="group flex items-center gap-3 rounded-lg border border-line bg-card px-3 py-2">
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-medium text-ink"><span className="font-mono text-ink-3">/{c.name}</span>{c.argHint && <span className="ml-1 font-mono text-[12px] text-ink-4">{c.argHint}</span>}</div>
                <div className="truncate text-[13px] text-ink-3">{c.description || c.prompt.slice(0, 80)}</div>
              </div>
              <button onClick={() => setEditCmd({ id: c.id, name: c.name, description: c.description, prompt: c.prompt, argHint: c.argHint })} className="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"><Icon name="edit" size={14} /></button>
              <button onClick={() => api.deleteCommand(c.id).then(reload)} className="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 opacity-0 transition-all hover:text-danger group-hover:opacity-100"><Icon name="trash" size={14} /></button>
            </div>
          ))}
          {commands.length === 0 && !editCmd && <div className="rounded-lg border border-dashed border-line px-3 py-4 text-center text-[13px] text-ink-3">No custom commands yet.</div>}
        </div>
        {editCmd && (
          <Editor onCancel={() => setEditCmd(null)} onSave={saveCmd} title={editCmd.id ? 'Edit command' : 'New command'}>
            <div className="flex gap-2">
              <Input value={editCmd.name || ''} onChange={(e) => setEditCmd({ ...editCmd, name: e.target.value.replace(/^\/+/, '') })} placeholder="name (used as /name)" />
              <Input value={editCmd.argHint || ''} onChange={(e) => setEditCmd({ ...editCmd, argHint: e.target.value })} placeholder="arg hint (optional)" />
            </div>
            <Input value={editCmd.description || ''} onChange={(e) => setEditCmd({ ...editCmd, description: e.target.value })} placeholder="Short description" />
            <textarea value={editCmd.prompt || ''} onChange={(e) => setEditCmd({ ...editCmd, prompt: e.target.value })} rows={3} placeholder="The prompt to send (your typed args are appended)…" className="scroll-thin w-full resize-none rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink outline-none focus:border-accent" />
          </Editor>
        )}
      </div>
    </div>
  )
}

function Editor({ title, children, onSave, onCancel }: { title: string; children: React.ReactNode; onSave: () => void; onCancel: () => void }) {
  return (
    <div className={clsx('mt-2 space-y-2 rounded-xl border border-accent/40 bg-card p-3 shadow-card')}>
      <div className="text-[13px] font-medium text-ink-2">{title}</div>
      {children}
      <div className="flex justify-end gap-2">
        <Button variant="ghost" size="sm" onClick={onCancel}>Cancel</Button>
        <Button variant="primary" size="sm" icon="check" onClick={onSave}>Save</Button>
      </div>
    </div>
  )
}
