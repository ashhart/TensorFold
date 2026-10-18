import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { ENGINE_PROVIDER_ID, formatBytes, loadEngineModel, refreshEngine, useEngine } from '../../lib/engine'
import { useLearning, learningBusy } from '../../lib/learning'
import { useStore } from '../../lib/store'
import type { EngineModelLibrarySnapshot } from '../../lib/types'
import { Button, Toggle } from '../ui'

export function ModelFoldersSettings({ compact = false }: { compact?: boolean }) {
  const tx = useAuthoredText()
  const [library, setLibrary] = useState<EngineModelLibrarySnapshot | null>(null)
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState('')
  const mounted = useRef(true)
  const { snapshot: engine } = useEngine()
  const { snapshot: learning } = useLearning()
  const threads = useStore((s) => s.threads)
  const statuses = useStore((s) => s.status)
  const activeRuns = threads.some((t) => t.providerId === ENGINE_PROVIDER_ID && !t.orchestratorHostId && t.workspace?.kind !== 'ssh' && ['thinking', 'streaming', 'tool', 'awaiting'].includes(statuses[t.id] ?? (t.status === 'running' ? 'thinking' : 'idle')))
  const modelLocked = activeRuns || learningBusy(learning) || engine?.status.state === 'starting'
  useEffect(() => {
    mounted.current = true
    api.engineLibrary().then((next) => { if (mounted.current) setLibrary(next) }).catch((e) => { if (mounted.current) setError(String(e?.message ?? e)) })
    return () => { mounted.current = false }
  }, [])
  const update = async (name: string, fn: () => Promise<EngineModelLibrarySnapshot | null>) => {
    setBusy(name); setError('')
    try {
      const next = await fn()
      if (mounted.current && next) setLibrary(next)
      await refreshEngine()
    } catch (e) { if (mounted.current) setError(String((e as Error)?.message ?? e)) }
    finally { if (mounted.current) setBusy(null) }
  }
  const scan = (includeHuggingFace?: boolean) => update('scan', () => api.engineScanLibrary(includeHuggingFace == null ? undefined : { includeHuggingFace }))
  const scanning = busy === 'scan' || !!library?.scanning
  const loading = busy?.startsWith('load:')
  return <section data-testid="model-folders" className={compact ? 'mt-6' : ''}>
    <div className="flex flex-wrap items-start justify-between gap-3">
      <div className="min-w-0 flex-1 basis-48"><h2 className="text-[16px] font-medium text-ink">{tx("Use models you already have")}</h2><p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Scan your Hugging Face cache or choose model folders, including external drives; compatible checkpoints load in place without another download.")}</p></div>
      <Button size="sm" icon="folder" disabled={!!busy || api.platform === 'web-remote'} onClick={() => void update('add', () => api.engineAddLibraryFolder())}>{tx("Add folder")}</Button>
    </div>
    <div className="mt-4 rounded-xl border border-line bg-card p-4">
      <div className="flex items-start justify-between gap-3"><div><p className="text-[14px] font-medium text-ink">{tx("Scan the Hugging Face cache")}</p><p className="mt-1 text-[13px] leading-relaxed text-ink-2">{tx("Uses your configured Hugging Face cache and the standard local cache; files stay where they are.")}</p></div><Toggle checked={library?.includeHuggingFace ?? false} onChange={(value) => void scan(value)} disabled={!!busy || !library || api.platform === 'web-remote'} label={tx("Scan the Hugging Face cache")} /></div>
      {(library?.folders ?? []).map((folder) => <div key={folder.id} className="mt-4 flex items-center gap-3 border-t border-line pt-3"><div className="min-w-0 flex-1"><p className="text-[13px] font-medium text-ink">{folder.label}</p><p className="mt-1 break-all font-mono text-[12px] text-ink-2">{folder.path}</p></div><Button size="xs" variant="ghost" disabled={!!busy} aria-label={tx("Stop scanning {value0}", { value0: folder.label })} onClick={() => void update(folder.id, () => api.engineRemoveLibraryFolder(folder.id))}>{tx("Remove folder")}</Button></div>)}
      <div className="mt-4 flex flex-wrap items-center gap-3"><Button size="sm" icon="refresh" disabled={!!busy || !library || api.platform === 'web-remote'} onClick={() => void scan()}>{scanning ? tx("Scanning…") : tx("Scan now")}</Button>{scanning && <Button size="sm" onClick={() => void api.engineCancelLibraryScan().catch((e) => setError(String(e?.message ?? e)))}>{tx("Cancel scan")}</Button>}<span role="status" className="text-[12px] text-ink-2">{scanning ? tx("Checking local files and model compatibility…") : library?.scannedAt ? tx("Last scan {value0}", { value0: new Date(library.scannedAt).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }) : tx("Scanning is optional and runs only when you ask.")}</span></div>
      <p className="mt-3 text-[12px] leading-relaxed text-ink-2">{tx("Removing a folder stops discovery there; it never deletes its models, and a running model stays loaded until you stop the server.")}</p>
    </div>
    {error && <p role="alert" className="mt-3 text-[13px] text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    {library?.warnings.map((warning, index) => <p key={index} role="status" className="mt-3 text-[13px] text-ink-2">{warning}</p>)}
    {library?.truncated && <p role="status" className="mt-3 text-[13px] text-ink-2">{tx("The scan reached its limit; choose a smaller model folder to see the remaining files.")}</p>}
    {library?.cancelled && <p role="status" className="mt-3 text-[13px] text-ink-2">{tx("Scan cancelled; scan again when you are ready.")}</p>}
    {library && !scanning && library.scannedAt && library.models.length === 0 && <p className="mt-4 rounded-xl border border-dashed border-line p-4 text-[13px] text-ink-2">{tx("No checkpoints found in the selected locations; add a folder containing model files or download one from Models.")}</p>}
    <div className="mt-4 space-y-3">{library?.models.map((model) => <article key={model.id} className="rounded-xl border border-line bg-card p-4">
      <div className="flex flex-wrap items-start justify-between gap-3"><div className="min-w-0 flex-1 basis-48"><h3 className="break-words text-[14px] font-medium text-ink">{model.title}</h3><p className="mt-1 text-[12px] text-ink-2">{model.source === 'huggingface' ? tx("Hugging Face cache") : tx("Local folder")} · {model.format ?? tx("Unknown format")}{model.weightsBytes != null ? ` · ${formatBytes(model.weightsBytes)}` : ''}</p><p className="mt-1 break-all font-mono text-[12px] text-ink-3">{model.path}</p>{model.reason && <p className="mt-2 text-[13px] leading-relaxed text-ink-2">{model.reason}</p>}</div><Button size="sm" icon="play" disabled={!model.loadable || !!busy || modelLocked || engine?.status.state === 'ready' && engine.status.model === model.id} onClick={() => void update(`load:${model.id}`, async () => { if (!model.loadable || modelLocked) throw new Error(tx("This model cannot load right now.")); await loadEngineModel(model.id); return api.engineLibrary() })}>{loading && busy === `load:${model.id}` ? tx("Loading…") : engine?.status.state === 'ready' && engine.status.model === model.id ? tx("Loaded") : tx("Load")}</Button></div>
    </article>)}</div>
    <p className="mt-4 text-[12px] leading-relaxed text-ink-2">{tx("Finding a model does not guarantee support; TensorFold checks architecture, quantization, required files and available memory before enabling Load.")}</p>
  </section>
}
