import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import { builtIn, customBrand, type Brand, type BrandKind, type CustomBrand } from '../brand'
import { logoOf, Tile } from '../brandLogo'
import { useWorking } from '../working'

const LOGO_KB = 150
const KINDS: { kind: BrandKind; title: string; hint: string }[] = [
  { kind: 'model', title: 'Model makers', hint: 'Recognised in a model\'s name: "llama" for Meta, "devstral" for Mistral AI.' },
  { kind: 'hardware', title: 'Hardware makers', hint: 'Recognised in a machine\'s chip: "snapdragon" for Qualcomm.' },
]
type Draft = { id: string | null; label: string; kind: BrandKind; words: string; logo: string | null | undefined }
const EMPTY: Draft = { id: null, label: '', kind: 'model', words: '', logo: undefined }
const slug = (label: string) => label.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40)

/** A picked image as a data URL, refused past the size TensorFold Web keeps. */
function readLogo(file: File): Promise<string> {
  if (file.size > LOGO_KB * 1024) return Promise.reject(new Error(`The logo is over ${LOGO_KB} KB`))
  return new Promise((ok, no) => {
    const r = new FileReader()
    r.onload = () => ok(String(r.result))
    r.onerror = () => no(new Error('The file could not be read'))
    r.readAsDataURL(file)
  })
}

/** The makers TensorFold Web recognises, its own and those added here, with their logos; a new family or a missing
 *  logo is added here, kept on this TensorFold Web, without a new version. */
export default function SettingsBrands({ toast, refreshBrands }: { toast: (t: string, k?: 'ok' | 'bad') => void; refreshBrands: () => Promise<unknown> }) {
  const [draft, setDraft] = useState<Draft | null>(null)
  const [own, setOwn] = useState<CustomBrand[] | null>(null)
  const pick = useRef<HTMLInputElement>(null)
  const forLogo = useRef<{ b: Brand; kind: BrandKind } | null>(null)
  const { run, is } = useWorking()
  const reload = () => Promise.all([api.brands().then((r) => setOwn(r.brands)), refreshBrands()])
  useEffect(() => { api.brands().then((r) => setOwn(r.brands)).catch(() => setOwn([])) }, [])
  const fail = (e: unknown) => toast(String((e as Error).message ?? e), 'bad')

  const save = (b: Parameters<typeof api.brandSave>[0], done: string) => run(b.id, async () => {
    try { await api.brandSave(b); await reload(); toast(done); setDraft(null) } catch (e) { fail(e) }
  })
  const remove = (b: CustomBrand, done: string) => run(b.id, async () => {
    try { await api.brandRemove(b.id, b.kind); await reload(); toast(done) } catch (e) { fail(e) }
  })
  // a logo for a maker in the list: kept as a Settings entry of the same id, its words those of the entry if any
  const giveLogo = (b: Brand, kind: BrandKind) => { forLogo.current = { b, kind }; pick.current?.click() }
  const onPicked = async (file: File | undefined) => {
    const target = forLogo.current
    if (!file || !target) return
    try {
      const logo = await readLogo(file)
      const known = customBrand(target.b.id, target.kind)
      save({ id: target.b.id, label: known?.label ?? target.b.label, kind: target.kind, words: known?.words ?? [], logo }, `Logo of ${target.b.label} kept`)
    } catch (e) { fail(e) }
  }
  const submit = () => {
    if (!draft) return
    const id = draft.id ?? slug(draft.label)
    if (!id) return toast('Give it a name', 'bad')
    if (!draft.id && own?.some((b) => b.id === id && b.kind === draft.kind)) return toast(`${draft.label} is in the list already: change it there`, 'bad')
    const words = draft.words.split(',').map((w) => w.trim()).filter(Boolean)
    save({ id, label: draft.label, kind: draft.kind, words, ...(draft.logo !== undefined ? { logo: draft.logo } : {}) }, `${draft.label} kept`)
  }

  return (
    <div className="card brands">
      <b>Makers and their logos</b>
      <p className="small muted">TensorFold Web shows a model's maker and the machine's maker beside their names, recognised by the words below. A new family, or a logo TensorFold Web lacks, can be added here; it is kept on this TensorFold Web and needs no new version. A maker with no logo shows its initial.</p>
      <input ref={pick} type="file" accept="image/png,image/jpeg,image/webp,image/svg+xml" hidden onChange={(e) => { onPicked(e.target.files?.[0]); e.target.value = '' }} />
      {KINDS.map(({ kind, title, hint }) => {
        const mine = (own ?? []).filter((b) => b.kind === kind)
        const added = mine.filter((b) => !builtIn(kind).some((k) => k.id === b.id))
        return (
          <div key={kind} className="brands-group">
            <div className="membar-head"><b>{title}</b><span className="small muted">{hint}</span></div>
            <details className="brands-more"><summary className="small">The {builtIn(kind).length + added.length} it recognises</summary>
            <ul className="brands-list">
              {[...builtIn(kind), ...added].map((b) => {
                const entry = mine.find((m) => m.id === b.id)
                const isOwn = !builtIn(kind).some((k) => k.id === b.id)
                const label = entry?.label ?? b.label
                const words = entry?.words.length ? entry.words.join(', ') : 'words' in b ? b.words : ''
                return (
                  <li key={b.id}>
                    <Tile b={{ id: b.id, label }} url={logoOf(b.id, kind)} size="sm" />
                    <span className="brands-name">{label}<span className="small muted">{words ? ` · ${words}` : ''}{isOwn ? ' · added here' : entry?.logo ? ' · logo given here' : ''}</span></span>
                    <span className="brands-actions">
                      <button className="btn sm" aria-busy={is(b.id)} onClick={() => giveLogo({ id: b.id, label }, kind)}>{logoOf(b.id, kind) ? 'Change logo' : 'Add a logo'}</button>
                      {entry && <button className="btn sm" onClick={() => setDraft({ id: entry.id, label: entry.label, kind, words: entry.words.join(', '), logo: undefined })}>Edit</button>}
                      {entry && <button className="btn sm" onClick={() => remove(entry, isOwn ? `${label} removed` : `${b.label} is back to TensorFold Web's own`)}>{isOwn ? 'Remove' : 'Undo'}</button>}
                    </span>
                  </li>
                )
              })}
            </ul>
            </details>
          </div>
        )
      })}
      {draft ? (
        <div className="brands-form">
          <div className="field"><span>Name</span><input value={draft.label} maxLength={40} onChange={(e) => setDraft({ ...draft, label: e.target.value })} placeholder="Falcon" aria-label="Maker name" /></div>
          <div className="field"><span>For</span><select className="sel" aria-label="Kind" value={draft.kind} disabled={!!draft.id} onChange={(e) => setDraft({ ...draft, kind: e.target.value as BrandKind })}><option value="model">models</option><option value="hardware">machines</option></select></div>
          <div className="field"><span>Words</span><input value={draft.words} onChange={(e) => setDraft({ ...draft, words: e.target.value })} placeholder="falcon, tii (commas between them)" aria-label="Words that recognise it" /></div>
          <div className="field"><span>Logo</span><input type="file" accept="image/png,image/jpeg,image/webp,image/svg+xml" aria-label="Logo" onChange={(e) => { const f = e.target.files?.[0]; if (f) readLogo(f).then((logo) => setDraft((d) => d && { ...d, logo }), fail) }} />{draft.logo && <img className="brands-preview" src={draft.logo} alt="" />}</div>
          <div className="actions"><button className="btn primary" aria-busy={is(draft.id ?? slug(draft.label))} disabled={!draft.label.trim()} onClick={submit}>Keep</button><button className="btn" onClick={() => setDraft(null)}>Cancel</button></div>
        </div>
      ) : <div className="actions"><button className="btn" onClick={() => setDraft(EMPTY)}>Add a maker</button></div>}
    </div>
  )
}
