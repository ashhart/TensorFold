import { useEffect, useState } from 'react'
import { api } from '../../lib/ipc'
import type { EngineRuntimeSnapshot } from '../../../electron/agent/types/engine-runtime'
import { Button } from '../ui'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'

export function EngineUpdates() {
  const tx = useAuthoredText()
  const [snapshot, setSnapshot] = useState<EngineRuntimeSnapshot | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [pending, setPending] = useState(false)
  useEffect(() => {
    let alive = true
    const read = () => { void api.engineRuntimeStatus().then(s => { if (alive) setSnapshot(s) }).catch(e => { if (alive) setError(String(e.message)) }) }
    read(); const timer = setInterval(read, 1000)
    return () => { alive = false; clearInterval(timer) }
  }, [])
  async function run(action: () => Promise<EngineRuntimeSnapshot>) {
    setError(null); setPending(true)
    try { setSnapshot(await action()) } catch (e) { setError((e as Error).message) } finally { setPending(false) }
  }
  const busy = pending || snapshot?.state !== 'idle'
  const installed = new Set(snapshot?.installed.map(p => p.id))
  const choices = [...snapshot?.installed ?? [], ...snapshot?.available.filter(p => !installed.has(p.id)) ?? []]
  return <section aria-label={tx('Engine updates')} className="rounded-xl border border-line bg-card p-4">
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div><h3 className="text-[14px] font-medium">{tx('Engine updates')}</h3><p className="mt-1 text-[12px] text-ink-2">{snapshot ? `TensorFold ${snapshot.active?.version ?? snapshot.bundledVersion}${snapshot.active ? ` · ${snapshot.active.backend}` : ` · ${tx('bundled')}`}` : tx('Checking installed runtime…')}</p></div>
      <Button size="sm" disabled={busy} onClick={() => void run(() => api.engineRuntimeCheck())}>{tx('Check for engine updates')}</Button>
    </div>
    <p className="mt-2 text-[12px] text-ink-2">{tx('Update the local engine without reinstalling the app. Finish local work before switching. Models and conversations stay in place.')}</p>
    {snapshot?.state !== 'idle' && snapshot && <p role="status" className="mt-2 text-[12px]">{tx(snapshot.state === 'checking' ? 'Checking signed releases…' : snapshot.state === 'switching' ? 'Switching runtime…' : 'Downloading and verifying…')}{snapshot.progress?.totalBytes ? ` ${Math.floor(100 * snapshot.progress.receivedBytes / snapshot.progress.totalBytes)}%` : ''}</p>}
    {snapshot && ['checking', 'downloading', 'installing'].includes(snapshot.state) && <Button size="sm" onClick={() => void api.engineRuntimeCancel().catch(e => setError(e.message))}>{tx('Cancel')}</Button>}
    {(error || snapshot?.error) && <p role="alert" className="mt-2 text-[12px] text-danger">{error || snapshot?.error}</p>}
    {snapshot?.checkedAt && choices.length === 0 && !error && !snapshot.error && <p role="status" className="mt-2 text-[12px] text-ink-2">{tx('No compatible engine updates are available for this device.')}</p>}
    {choices.map(pkg => <div key={pkg.id} className="mt-3 border-t border-line pt-3">
      <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-[13px]">TensorFold {pkg.version} · {pkg.backend}{snapshot?.active?.id === pkg.id ? ` · ${tx('active')}` : ''}</p>
        {snapshot?.active?.id !== pkg.id && <Button size="sm" disabled={busy} onClick={() => void run(() => installed.has(pkg.id) ? api.engineRuntimeUse(pkg.id) : api.engineRuntimeInstall(pkg.id))}>{installed.has(pkg.id) ? tx('Use this runtime') : tx('Download')}</Button>}</div>
      <p className="mt-1 whitespace-pre-wrap text-[12px] text-ink-2">{pkg.notes}</p>
    </div>)}
    {snapshot?.active && <Button className="mt-3" size="sm" disabled={busy} onClick={() => void run(() => api.engineRuntimeUse(null))}>{tx('Use bundled engine')}</Button>}
  </section>
}
