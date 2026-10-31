// The user's own suites on the Bench screen: written here or imported from a file, measured like the built-in ones.
import { useRef, useState } from 'react'
import { api, download, type CustomPrompt, type CustomSuite, type CustomSuiteInput } from '../api'
import type { Ctx } from '../ctx'

const BLANK: CustomPrompt = { user: '', system: null, max_tokens: 300 }

function Editor({ initial, onDone, toast }: { initial: CustomSuite | null; onDone: (saved: boolean) => void; toast: Ctx['toast'] }) {
  const [name, setName] = useState(initial?.name ?? '')
  const [about, setAbout] = useState(initial?.about ?? '')
  const [prompts, setPrompts] = useState<CustomPrompt[]>(initial?.prompts.map((p) => ({ ...p })) ?? [{ ...BLANK }])
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const set = (i: number, patch: Partial<CustomPrompt>) => setPrompts((ps) => ps.map((p, j) => (j === i ? { ...p, ...patch } : p)))
  const save = async () => {
    setSaving(true); setErr(null)
    try {
      await api.suiteSave({ name, about, prompts: prompts.map((p) => ({ ...p, system: p.system?.trim() || null })) }, initial?.id)
      toast(initial ? 'Suite saved; its digest changes if a prompt did' : 'Suite added'); onDone(true)
    } catch (e) { setErr(String(e)) } finally { setSaving(false) }
  }
  return (
    <div className="suite-edit">
      {err && <div className="notice bad">{err}</div>}
      <label className="small"><span>Name</span><input className="input" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} placeholder="Customer support replies" /></label>
      <label className="small"><span>What it measures <span className="muted">(optional)</span></span><input className="input" value={about} maxLength={500} onChange={(e) => setAbout(e.target.value)} /></label>
      {prompts.map((p, i) => (
        <fieldset key={i} className="suite-prompt">
          <legend>Prompt {i + 1}</legend>
          {!!p.history?.length && <details open>
            <summary className="small">Earlier in the conversation ({p.history.length} {p.history.length === 1 ? 'turn' : 'turns'})</summary>
            {p.history.map((t, k) => (
              <div key={k} className="suite-turn">
                <span className="small muted">{t.role === 'user' ? 'User' : 'Assistant'}</span>
                <textarea className="input" rows={2} value={t.content} aria-label={`Prompt ${i + 1}, earlier turn ${k + 1} (${t.role})`}
                  onChange={(e) => set(i, { history: p.history!.map((h, m) => (m === k ? { ...h, content: e.target.value } : h)) })} />
              </div>
            ))}
            <button type="button" className="btn sm quiet" onClick={() => set(i, { history: [] })}>Remove the earlier turns</button>
          </details>}
          {!!p.history?.length && <span className="small muted">User, the message it answers</span>}
          <textarea className="input" rows={4} value={p.user} onChange={(e) => set(i, { user: e.target.value })} placeholder="The message, as a user would send it" aria-label={`Prompt ${i + 1}`} />
          {!p.history?.length && <button type="button" className="btn sm quiet suite-addturn" onClick={() => set(i, { history: [{ role: 'user', content: '' }, { role: 'assistant', content: '' }] })}>Add earlier turns (a conversation already going)</button>}
          <details open={!!p.system}>
            <summary className="small">System message</summary>
            <textarea className="input" rows={2} value={p.system ?? ''} onChange={(e) => set(i, { system: e.target.value })} aria-label={`System message of prompt ${i + 1}`} />
          </details>
          <div className="suite-row small">
            <label>Reply up to <input className="input num" type="number" min={16} max={4096} value={p.max_tokens} onChange={(e) => set(i, { max_tokens: Number(e.target.value) })} /> tokens</label>
            {prompts.length > 1 && <button type="button" className="btn sm quiet" onClick={() => setPrompts((ps) => ps.filter((_, j) => j !== i))}>Remove</button>}
          </div>
        </fieldset>
      ))}
      <div className="actions">
        {prompts.length < 20 && <button type="button" className="btn sm" onClick={() => setPrompts((ps) => [...ps, { ...BLANK }])}>Add a prompt</button>}
        <span style={{ flex: 1 }} />
        <button type="button" className="btn sm quiet" onClick={() => onDone(false)}>Cancel</button>
        <button type="button" className="btn sm primary" aria-busy={saving} disabled={saving || !name.trim() || prompts.some((p) => !p.user.trim() || p.history?.some((t) => !t.content.trim()))} onClick={save}>{saving ? 'Saving…' : 'Save the suite'}</button>
      </div>
    </div>
  )
}

export default function CustomSuites({ suites, builtIn, onChange, toast }: { suites: CustomSuite[]; builtIn: { id: string; name: string }[]; onChange: () => void; toast: Ctx['toast'] }) {
  const [editing, setEditing] = useState<CustomSuite | 'new' | null>(null)
  const [asking, setAsking] = useState<string | null>(null)
  const file = useRef<HTMLInputElement>(null)
  const importFile = async (f: File) => {
    try {
      const doc = JSON.parse(await f.text()) as CustomSuiteInput
      await api.suiteSave(doc); toast(`Suite "${doc.name}" imported`); onChange()
    } catch (e) { toast(e instanceof SyntaxError ? 'This file is not JSON' : String(e), 'bad') }
  }
  // a built-in suite copied in and opened at once, to change its prompts; the built-in one stays as it is
  const copy = async (id: string) => {
    if (!id) return
    try { const made = await api.suiteCopy(id); toast(`"${made.name}" added: change its prompts, then save`); onChange(); setEditing(made) } catch (e) { toast(String(e), 'bad') }
  }
  const remove = async (id: string) => {
    setAsking(null)
    try { await api.suiteDelete(id); toast('Suite deleted; runs made with it keep its prompts'); onChange() } catch (e) { toast(String(e), 'bad') }
  }
  return (
    <div className="card">
      <div className="report-head">
        <div><b>Your suites</b><p className="small muted">Prompts of your own, measured in a usage profile, a load sweep or a KL divergence with the same protocol and exactness check. The built-in suites stay fixed so that runs compare; copy one here to change its prompts. A suite's digest covers what is sent, so runs compare only on the same prompts.</p></div>
        {!editing && <div className="actions">
          <select className="sel sm" value="" aria-label="Copy a built-in suite" onChange={(e) => copy(e.target.value)}>
            <option value="">Copy a built-in suite…</option>
            {builtIn.map((b) => <option key={b.id} value={b.id}>{b.name}</option>)}
          </select>
          <button className="btn sm" onClick={() => file.current?.click()}>Import</button>
          <button className="btn sm primary" onClick={() => setEditing('new')}>New suite</button>
          <input ref={file} type="file" accept="application/json,.json" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) importFile(f); e.target.value = '' }} />
        </div>}
      </div>
      {editing && <Editor initial={editing === 'new' ? null : editing} toast={toast} onDone={(saved) => { setEditing(null); if (saved) onChange() }} />}
      {!editing && (suites.length === 0
        ? <p className="small muted">None yet. Write one here, or import a file exported from another TensorFold Web.</p>
        : <div className="tablewrap"><table className="tbl">
          <thead><tr><th>Name</th><th>prompts</th><th>digest</th><th /></tr></thead>
          <tbody>{suites.map((s) => (
            <tr key={s.id}>
              <td>{s.name}{s.about ? <div className="small muted">{s.about}</div> : null}</td><td>{s.prompts.length}</td><td><code>{s.digest}</code></td>
              <td className="actions">{asking === s.id
                ? <><span className="small">Delete it?</span><button className="btn sm danger" onClick={() => remove(s.id)}>Delete</button><button className="btn sm quiet" onClick={() => setAsking(null)}>Keep</button></>
                : <><button className="btn sm quiet" onClick={() => setEditing(s)}>Edit</button>
                  <button className="btn sm quiet" onClick={() => download(`/api/bench/suites/${s.id}/export`, `${s.id}.json`).catch((e) => toast(`Not exported: ${e}`, 'bad'))}>Export</button>
                  <button className="btn sm quiet" onClick={() => setAsking(s.id)}>Delete</button></>}</td>
            </tr>
          ))}</tbody>
        </table></div>)}
    </div>
  )
}
