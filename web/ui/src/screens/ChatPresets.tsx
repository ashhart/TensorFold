// Chat presets: a system prompt and the sampling, chosen per conversation. The built-in ones ship with the page;
// a person's own are kept on TensorFold Web, so every browser and device that opens it sees the same list.
import { useEffect, useState, type ReactNode } from 'react'
import { api, ApiError, type ChatPreset } from '../api'
import type { Ctx } from '../ctx'
import { Ask, AskName } from '../ask'

export type Preset = ChatPreset

export const PRESETS: Preset[] = [
  { name: 'Model defaults', system: '', temperature: null, max_tokens: null },
  { name: 'Precise', system: 'Answer exactly and briefly. Say when you are not sure.', temperature: 0, max_tokens: null },
  { name: 'Creative', system: '', temperature: 1.0, max_tokens: null },
  { name: 'Code reviewer', system: 'You review code. Point out bugs first, then risks, then style; quote the lines you mean.', temperature: 0.2, max_tokens: 4096 },
  { name: 'Short answers', system: 'Answer in three sentences or fewer.', temperature: null, max_tokens: 512 },
]

const OLD_KEY = 'tfs-chat-presets'                           // where this browser kept them before TensorFold Web did
const SAMPLING = ['temperature', 'max_tokens', 'top_p', 'top_k', 'min_p', 'seed'] as const

/** The request fields a preset sets; a field left empty is the model's own. */
export function sampling(p: Preset): Record<string, unknown> {
  const out: Record<string, unknown> = {}
  for (const k of SAMPLING) if (p[k] != null) out[k] = p[k]
  const stop = p.stop?.filter(Boolean)
  if (stop?.length) out.stop = stop
  return out
}

const same = (a: Preset, b: Preset) => a.system === b.system && (a.stop ?? []).join('\n') === (b.stop ?? []).join('\n') && SAMPLING.every((k) => (a[k] ?? null) === (b[k] ?? null))

/** The presets kept on TensorFold Web; those this browser held before are moved there once, then forgotten here. */
export function usePresets(toast: Ctx['toast']) {
  const [mine, setMine] = useState<Preset[]>([])
  const reload = () => api.chatPresets().then((r) => setMine(r.presets)).catch(() => {})
  useEffect(() => {
    (async () => {
      let local: Preset[] = []
      try { local = JSON.parse(localStorage.getItem(OLD_KEY) ?? '[]') as Preset[] } catch { /* none */ }
      if (local.length) {
        let kept = true
        const there = await api.chatPresets().then((r) => r.presets).catch(() => null)
        if (there) {
          for (const p of local) {
            const twin = there.find((t) => t.name === p.name)
            if (twin && same(twin, p)) continue
            // a different preset of the same name on TensorFold Web keeps its name; this one is marked as this browser's
            try { await api.chatPresetSave({ ...p, stop: p.stop?.filter(Boolean), name: twin ? `${p.name} (this browser)`.slice(0, 60) : p.name }) } catch (e) { if (!(e instanceof ApiError && e.status === 409)) kept = false }
          }
          if (kept) { try { localStorage.removeItem(OLD_KEY) } catch { /* no storage */ } toast(`Your presets are kept on TensorFold Web now (${local.length})`) }
        }
      }
      reload()
    })()
  }, [])
  return { mine, reload }
}

const num = (v: string, int = false) => v === '' ? null : int ? Math.round(Number(v)) : Number(v)

/** A setting: its name and control on one line, a sentence saying what it does under them. */
export function Field({ label, help, children }: { label: string; help: ReactNode; children: ReactNode }) {
  return <label className="pfield"><span><b>{label}</b> {children}</span><span className="small muted">{help}</span></label>
}

/** The open conversation's settings in three groups (instructions, length of the reply, randomness, with the finer
 *  sampling folded), then the preset's actions: saved under a name, renamed or deleted when it is one's own. */
export function PresetEditor({ preset, setPreset, mine, reload, toast }: {
  preset: Preset; setPreset: (p: Preset) => void; mine: Preset[]; reload: () => void; toast: Ctx['toast']
}) {
  const [ask, setAsk] = useState<null | 'name' | 'rename' | 'delete' | { replace: string }>(null)
  const own = mine.some((p) => p.name === preset.name)
  const finer = preset.top_p != null || preset.top_k != null || preset.min_p != null || preset.seed != null
  const save = async (name: string, replace: boolean) => {
    if (PRESETS.some((p) => p.name === name)) { toast('That name is one of the built-in presets', 'bad'); return }
    if (!replace && mine.some((p) => p.name === name)) { setAsk({ replace: name }); return }
    try {
      const p = await api.chatPresetSave({ ...preset, name, stop: preset.stop?.filter(Boolean) }, replace)
      setPreset(p); setAsk(null); reload(); toast(replace ? `Preset ${name} replaced` : `Preset ${name} saved`)
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) setAsk({ replace: name })
      else toast(String(e), 'bad')
    }
  }
  return (
    <div className="presetbox">
      <div className="phead"><b>This conversation's settings</b><span className="small muted">from the preset “{preset.name}”; they apply to the next messages and are kept with the conversation</span></div>
      <div className="pgroups">
        <section className="pgroup pwide" aria-label="Instructions">
          <h4>Instructions</h4>
          <label className="pfield"><span className="small muted">What the model is told before every message (the system prompt).</span>
            <textarea className="input" rows={3} value={preset.system} onChange={(e) => setPreset({ ...preset, system: e.target.value })} placeholder="None: the model's own" /></label>
        </section>
        <section className="pgroup" aria-label="Length of the reply">
          <h4>Length of the reply</h4>
          <Field label="Reply up to" help="The longest reply, in tokens (a token is about three quarters of a word).">
            <input className="input num" type="number" min={16} max={65536} step={16} value={preset.max_tokens ?? ''} placeholder="model" onChange={(e) => setPreset({ ...preset, max_tokens: num(e.target.value, true) })} /> <span className="small muted">tokens</span></Field>
          <Field label="Stop at" help="Texts that end the reply where they appear: one per line, up to four.">
            <textarea className="input" rows={2} value={(preset.stop ?? []).join('\n')} placeholder="none" onChange={(e) => setPreset({ ...preset, stop: e.target.value.split('\n').slice(0, 4).map((x) => x.slice(0, 64)).filter((x, i, all) => x || i === all.length - 1) })} /></Field>
        </section>
        <section className="pgroup" aria-label="Randomness">
          <h4>Randomness</h4>
          <Field label="Temperature" help="How free the wording is: 0 always takes the likeliest word, about 1 is the model's usual, higher is looser.">
            <input className="input num" type="number" min={0} max={2} step={0.1} value={preset.temperature ?? ''} placeholder="model" onChange={(e) => setPreset({ ...preset, temperature: num(e.target.value) })} /></Field>
          <details className="pmore" open={finer}>
            <summary className="small">Finer control: top_p, top_k, min_p, seed{finer ? ' (set)' : ''}</summary>
            <div className="pgrid">
              <Field label="top_p" help="Draw only among the likeliest words that together reach this share: 0.95 leaves out the unlikely 5%.">
                <input className="input num" type="number" min={0} max={1} step={0.05} value={preset.top_p ?? ''} placeholder="model" onChange={(e) => setPreset({ ...preset, top_p: num(e.target.value) })} /></Field>
              <Field label="top_k" help="Draw only among the K likeliest words; 40 is usual, 1 always takes the likeliest.">
                <input className="input num" type="number" min={1} max={1000} step={1} value={preset.top_k ?? ''} placeholder="model" onChange={(e) => setPreset({ ...preset, top_k: num(e.target.value, true) })} /></Field>
              <Field label="min_p" help="Leave out a word less likely than this share of the best one; 0.05 is usual.">
                <input className="input num" type="number" min={0} max={1} step={0.01} value={preset.min_p ?? ''} placeholder="model" onChange={(e) => setPreset({ ...preset, min_p: num(e.target.value) })} /></Field>
              <Field label="seed" help="Makes the draw repeatable: the same seed and question give the same reply. Left empty, TensorFold Web draws a new one for each reply (kept with it), so asking again gives another answer.">
                <input className="input num" type="number" min={0} step={1} value={preset.seed ?? ''} placeholder="none" onChange={(e) => setPreset({ ...preset, seed: num(e.target.value, true) })} /></Field>
            </div>
            <p className="small muted" style={{ margin: 0 }}>Repetition penalties are left out: TensorFold accepts them and does not apply them.</p>
          </details>
          <p className="small muted" style={{ margin: 0 }}>A replay and the speed-up check write without randomness: they keep the instructions and the length, not these.</p>
        </section>
      </div>
      <div className="pactions">
        <span className="small muted">Preset</span>
        <button type="button" className="btn sm" onClick={() => setAsk('name')}>Save as a preset…</button>
        {own && <button type="button" className="btn sm quiet" onClick={() => save(preset.name, true)}>Save changes to {preset.name}</button>}
        {own && <button type="button" className="btn sm quiet" onClick={() => setAsk('rename')}>Rename</button>}
        {own && <button type="button" className="btn sm quiet danger-text" onClick={() => setAsk('delete')}>Delete…</button>}
        <span className="small muted pnote">Your presets are kept on this TensorFold Web, for every browser that opens it.</span>
      </div>
      {ask === 'name' && <AskName text="A name for this preset" initial={own || preset.name.startsWith('My ') ? preset.name : `My ${preset.name.toLowerCase()}`} ok="Save it" onOk={(n) => save(n, false)} onCancel={() => setAsk(null)} />}
      {ask && typeof ask === 'object' && <Ask yes="Replace it" onYes={() => save(ask.replace, true)} onNo={() => setAsk('name')} text={<>A preset named “{ask.replace}” is kept already. Replace it with these settings?</>} />}
      {ask === 'rename' && <AskName text="New name of the preset" initial={preset.name} ok="Rename" onOk={async (n) => {
        if (PRESETS.some((p) => p.name === n)) { toast('That name is one of the built-in presets', 'bad'); return }
        try { const p = await api.chatPresetRename(preset.name, n); setPreset({ ...preset, name: p.name }); setAsk(null); reload(); toast('Preset renamed') }
        catch (e) { toast(e instanceof ApiError && e.status === 409 ? `A preset named ${n} is kept already` : String(e), 'bad') }
      }} onCancel={() => setAsk(null)} />}
      {ask === 'delete' && <Ask danger yes="Delete it" onYes={async () => {
        try { await api.chatPresetRemove(preset.name); setAsk(null); reload(); setPreset(PRESETS[0]); toast('Preset deleted') }
        catch (e) { toast(String(e), 'bad') }
      }} onNo={() => setAsk(null)} text={<>Delete the preset “{preset.name}”? The conversations that used it keep their settings; this one goes back to Model defaults.</>} />}
    </div>
  )
}
