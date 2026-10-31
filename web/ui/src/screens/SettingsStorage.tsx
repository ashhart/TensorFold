import { useCallback, useEffect, useState } from 'react'
import { api, num, type StorageModel, type StorageStore } from '../api'
import { bytes } from '../units'
import { Ask } from '../ask'

// how far back each kind is cleaned; the Monitor's readings are only ever cleaned whole (they last eight hours)
const SCOPES: { days: number | null; label: string }[] = [{ days: 30, label: 'older than 30 days' }, { days: 90, label: 'older than 90 days' }, { days: null, label: 'everything' }]
const NOUN: Record<StorageStore['id'], [string, string]> = { bench: ['Bench run', 'Bench runs'], conversations: ['conversation', 'conversations'], usage: ['hour of usage', 'hours of usage'], history: ['reading', 'readings'] }
const day = (t: number | null) => (t == null ? '—' : new Date(t * 1000).toLocaleDateString())

/** Settings: what TensorFold Web keeps, and clean-ups by kind and age, each counted before it is done. */
export default function SettingsStorage({ toast }: { toast: (t: string, k?: 'ok' | 'bad') => void }) {
  const [stores, setStores] = useState<StorageStore[] | null>(null)
  const [models, setModels] = useState<StorageModel[]>([])
  const [cache, setCache] = useState('')
  const [scope, setScope] = useState<Record<string, number | null>>({ bench: 90, conversations: 90, usage: 90, history: null })
  const [busy, setBusy] = useState<string | null>(null)
  // the count is made first; the question then sits by the row, with what it would remove
  const [ask, setAsk] = useState<{ kind: 'clean'; st: StorageStore; days: number | null; text: string } | { kind: 'model'; m: StorageModel; text: string } | null>(null)
  const load = useCallback(() => api.storage().then((s) => { setStores(s.stores); setModels(s.models); setCache(s.cache) }).catch((e) => toast(`Storage not read: ${e.message ?? e}`, 'bad')), [toast])
  useEffect(() => { load() }, [load])

  const clean = async (st: StorageStore) => {
    const days = st.id === 'history' ? null : scope[st.id]
    const what = days == null ? 'all' : `those ${SCOPES.find((x) => x.days === days)?.label}`
    setBusy(st.id); setAsk(null)
    try {
      const preview = await api.storageClean(st.id, days, true)
      const [one, many] = NOUN[st.id]
      if (!preview.matched) { toast(`No ${many} ${days == null ? 'kept' : `older than ${days} days`}`); return }
      setAsk({ kind: 'clean', st, days, text: `Remove ${num(preview.matched)} ${preview.matched === 1 ? one : many} (${what})? This cannot be undone.${st.id === 'bench' ? ' A run still going is kept.' : st.id === 'conversations' ? ' Pinned conversations are kept.' : ''}` })
    } catch (e) { toast(String((e as Error).message ?? e), 'bad') } finally { setBusy(null) }
  }
  const cleanNow = async (st: StorageStore, days: number | null) => {
    setBusy(st.id); setAsk(null)
    try {
      const [one, many] = NOUN[st.id]
      const done = await api.storageClean(st.id, days, false)
      toast(`${num(done.removed)} ${done.removed === 1 ? one : many} removed`)
      load()
    } catch (e) { toast(String((e as Error).message ?? e), 'bad') } finally { setBusy(null) }
  }
  const removeModel = async (m: StorageModel) => {
    setBusy(m.repo); setAsk(null)
    try {
      const preview = await api.storageRemoveModel(m.repo, true)
      setAsk({ kind: 'model', m, text: `Remove ${m.repo} from this machine's disk? It frees ${bytes(preview.freed, 1)}; downloading it again takes a while.` })
    } catch (e) { toast(String((e as Error).message ?? e), 'bad') } finally { setBusy(null) }
  }
  const removeNow = async (m: StorageModel) => {
    setBusy(m.repo); setAsk(null)
    try {
      const done = await api.storageRemoveModel(m.repo, false)
      toast(`${m.repo} removed, ${bytes(done.freed, 1)} freed`)
      load()
    } catch (e) { toast(String((e as Error).message ?? e), 'bad') } finally { setBusy(null) }
  }
  if (!stores) return null
  return (
    <div className="card" aria-labelledby="storage-title">
      <div className="membar-head"><b id="storage-title">Storage</b><span className="small muted">kept in TensorFold Web's folder on this machine</span></div>
      {ask?.kind === 'clean' && <Ask danger yes="Remove them" busy={busy === ask.st.id} onYes={() => cleanNow(ask.st, ask.days)} onNo={() => setAsk(null)} text={ask.text} />}
      <div className="tablewrap"><table className="tbl storage">
        <thead><tr><th>What</th><th>Kept</th><th>Size</th><th>Since</th><th><span className="sr-only">Clean up</span></th></tr></thead>
        <tbody>{stores.map((st) => (
          <tr key={st.id}>
            <td>{st.label}</td><td>{num(st.count)}</td><td>{bytes(st.bytes, 1)}</td><td>{day(st.oldest)}</td>
            <td className="act">
              {st.id !== 'history' && <select className="sel" aria-label={`Which ${st.label.toLowerCase()} to remove`} value={String(scope[st.id])} onChange={(e) => setScope((s) => ({ ...s, [st.id]: e.target.value === 'null' ? null : Number(e.target.value) }))}>
                {SCOPES.map((x) => <option key={String(x.days)} value={String(x.days)}>{x.label}</option>)}</select>}
              <button className="btn sm" aria-busy={busy === st.id} disabled={!st.count || busy != null} onClick={() => clean(st)} aria-label={`Clean up ${st.label.toLowerCase()}`}>{busy === st.id ? 'Counting…' : 'Clean up…'}</button>
            </td>
          </tr>))}</tbody>
      </table></div>
      <div className="usub small muted">Models from the catalog on disk{cache ? <>, in <code>{cache}</code></> : null}</div>
      {ask?.kind === 'model' && <Ask danger yes="Remove it from disk" busy={busy === ask.m.repo} onYes={() => removeNow(ask.m)} onNo={() => setAsk(null)} text={ask.text} />}
      {models.length ? <div className="tablewrap"><table className="tbl storage">
        <thead><tr><th>Model</th><th>Size</th><th>Last used</th><th><span className="sr-only">Remove</span></th></tr></thead>
        <tbody>{models.map((m) => (
          <tr key={m.repo}><td>{m.repo}</td><td>{bytes(m.bytes, 1)}</td><td>{day(m.last_used)}</td>
            <td className="act">{m.busy ? <span className="small muted">{m.busy}</span> : <button className="btn sm" aria-busy={busy === m.repo} disabled={busy != null} onClick={() => removeModel(m)} aria-label={`Remove ${m.repo} from disk`}>{busy === m.repo ? 'Counting…' : 'Remove…'}</button>}</td></tr>))}</tbody>
      </table></div> : <p className="small muted">None on this machine.</p>}
      <p className="small muted" style={{ marginBottom: 0 }}>Every clean-up says how much it would remove and asks before it does. Only the catalog's models are listed: other models in the same cache are left alone.</p>
    </div>
  )
}
