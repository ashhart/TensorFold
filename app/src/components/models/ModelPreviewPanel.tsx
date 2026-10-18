import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { loadEngineModel } from '../../lib/engine'
import type { EngineInstalledModel, EngineModelPreviewResult } from '../../lib/types'
import { Button } from '../ui'

export function ModelPreviewPanel({ models, activeModel, locked, fixedModel, lockReason, onBusyChange }: { models: EngineInstalledModel[]; activeModel: string | null; locked: boolean; fixedModel?: string; lockReason?: string; onBusyChange?: (busy: boolean) => void }) {
  const tx = useAuthoredText()
  const [selected, setSelected] = useState(fixedModel ?? activeModel ?? models[0]?.repo ?? '')
  const [busy, setBusy] = useState(false), [using, setUsing] = useState(false), [error, setError] = useState('')
  const [result, setResult] = useState<EngineModelPreviewResult | null>(null)
  const [notice, setNotice] = useState('')
  const ownsPreview = useRef(false), ownsUse = useRef(false), mounted = useRef(true)
  const generation = useRef(0)
  const target = fixedModel ?? selected
  const available = models.find((model) => model.repo === target)
  useEffect(() => { onBusyChange?.(busy || using); return () => onBusyChange?.(false) }, [busy, using, onBusyChange])
  useEffect(() => {
    mounted.current = true
    const unsubscribe = api.onEngineEvent((event) => {
      if (event.kind !== 'comparison-cleared' || !mounted.current) return
      generation.current++
      if (ownsPreview.current) void api.engineCancelTryModel().catch(() => {})
      ownsPreview.current = false; ownsUse.current = false
      setResult(null); setError(''); setBusy(false); setUsing(false)
      setNotice(tx("Preview data was cleared with local memory."))
    })
    return () => { mounted.current = false; generation.current++; unsubscribe(); if (ownsPreview.current) void api.engineCancelTryModel().catch(() => {}) }
  }, [])
  useEffect(() => { if (fixedModel) { if (selected !== fixedModel) setSelected(fixedModel) } else if (!models.some((model) => model.repo === selected)) setSelected(activeModel ?? models[0]?.repo ?? '') }, [models, activeModel, selected, fixedModel])
  const trial = async () => {
    if (ownsPreview.current || ownsUse.current || locked || !available || available.fitsMemory === false) return
    const startedGeneration = generation.current
    ownsPreview.current = true; setBusy(true); setError(''); setNotice(''); setResult(null)
    try { const measured = await api.engineTryModel(target); if (mounted.current && generation.current === startedGeneration) setResult(measured) }
    catch (failure) { if (mounted.current && generation.current === startedGeneration) setError((failure as Error).message.replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '')) }
    finally { if (generation.current === startedGeneration) { ownsPreview.current = false; if (mounted.current) setBusy(false) } }
  }
  const cancel = async () => { const startedGeneration = generation.current; try { await api.engineCancelTryModel() } catch (failure) { if (mounted.current && generation.current === startedGeneration) setError((failure as Error).message) } }
  const useModel = async () => {
    if (!result || ownsUse.current || ownsPreview.current || locked || result.model !== target || !available || available.fitsMemory === false) return
    const startedGeneration = generation.current
    ownsUse.current = true; setUsing(true); setError(''); setNotice('')
    try { await loadEngineModel(result.model) } catch (failure) { if (mounted.current && generation.current === startedGeneration) setError((failure as Error).message) }
    finally { if (generation.current === startedGeneration) { ownsUse.current = false; if (mounted.current) setUsing(false) } }
  }
  return <section aria-label={tx("Model preview")} className="rounded-xl border border-line bg-card p-4">
    <div className="flex flex-wrap items-start justify-between gap-3"><div className="min-w-0 flex-1 basis-56"><h2 className="text-[15px] font-medium text-ink">{tx("Try on my Mac")}</h2><p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Run a short local sample and see measured speed before choosing a model.")}</p></div>
      <div className="flex min-w-0 flex-wrap items-center gap-2">{fixedModel ? <span className="max-w-full break-words text-[13px] text-ink-2">{available?.title ?? target}</span> : <label className="min-w-0 flex-1 text-[13px] text-ink-2"><span className="sr-only">{tx("Preview model")}</span><select aria-label={tx("Preview model")} value={selected} onChange={(event) => { setSelected(event.target.value); setResult(null) }} disabled={busy || using || locked} className="h-8 max-w-full rounded-lg border border-line bg-card-2 px-2 text-[13px] text-ink">{models.map((model) => <option key={model.repo} value={model.repo} disabled={model.fitsMemory === false}>{model.title ?? model.repo}</option>)}</select></label>}
        {busy ? <Button size="sm" onClick={() => void cancel()}>{tx("Cancel preview")}</Button> : <Button size="sm" disabled={!available || locked || using || available.fitsMemory === false} onClick={() => void trial()}>{tx("Try model")}</Button>}</div></div>
    <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("This loads the selected model and uses a fixed sample with no files or tools; your chat draft and chosen model stay the same.")}</p>
    {available?.fitsMemory === false && <p role="status" className="mt-2 text-[12px] text-warn">{tx("This model exceeds the available memory budget on this Mac.")}</p>}
    {locked && <p className="mt-2 text-[12px] text-ink-2">{lockReason ?? tx("Finish active local work before measuring or changing models.")}</p>}
    {busy && <p role="status" className="mt-3 text-[13px] text-ink-2">{tx("Loading and measuring the model…")}</p>}
    {notice && <p role="status" className="mt-3 text-[13px] text-ink-2">{notice}</p>}
    {error && <p role="alert" className="mt-3 break-words text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    {result && <div role="status" className="mt-4 border-t border-line pt-4"><dl className="grid grid-cols-2 gap-4 sm:grid-cols-4">
      <Measure label={tx("Decode speed")} value={result.decodeTokensPerSecond == null ? 'Not reported' : `${result.decodeTokensPerSecond.toFixed(1)} tok/s`} />
      <Measure label={tx("First token")} value={result.firstTokenMs == null ? 'Not reported' : `${(result.firstTokenMs / 1000).toFixed(2)} s`} />
      <Measure label={tx("Load and preparation")} value={`${(result.loadMs / 1000).toFixed(2)} s`} />
      <Measure label={tx("Generated tokens")} value={result.completionTokens == null ? 'Not reported' : String(result.completionTokens)} /></dl>
      <p className="mt-3 text-[12px] text-ink-2">{tx("A short sample measured on this Mac")}{result.cachedTokens != null ? tx("; {value0} prompt tokens reused", { value0: result.cachedTokens }) : tx("; cache reuse was not reported")}{tx(". Performance varies with prompt length, context and other running apps.")}</p>
      <p className="mt-3 whitespace-pre-wrap break-words rounded-lg bg-card-2 p-3 text-[13px] leading-relaxed text-ink">{result.text}</p>
      <div className="mt-3 flex flex-wrap items-center gap-2"><Button size="sm" disabled={locked || using || !available || available.fitsMemory === false || result.model !== target} onClick={() => void useModel()}>{using ? tx("Selecting…") : tx("Use this model")}</Button><span className="text-[12px] text-ink-2">{tx("Sets the model for new chats.")}</span></div></div>}
  </section>
}
function Measure({ label, value }: { label: string; value: string }) { return <div className="min-w-0"><dt className="text-[12px] text-ink-2">{label}</dt><dd className="mt-1 break-words text-[16px] font-medium tabular-nums text-ink">{value}</dd></div> }
