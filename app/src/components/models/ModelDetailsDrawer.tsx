import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useCallback, useEffect, useRef, useState, type KeyboardEvent } from 'react'
import { formatBytes } from '../../lib/engine'
import { repoDisplayName } from '../../lib/turn-stats-line'
import type { EngineInstalledModel, EngineStatus } from '../../lib/types'
import { Button } from '../ui'
import { ModelPreviewPanel } from './ModelPreviewPanel'
import { ModelProfileEditor, type ModelProfileEditorControls, type ModelProfileEditorState } from './ModelProfileEditor'
import { ModelsDrawer } from './ModelsDrawer'
import { ModelQualificationPanel } from './qualification/ModelQualificationPanel'
import { cachedBenchmarkModels } from './benchmark/benchmark-display'

type DetailsTab = 'overview' | 'settings' | 'performance'
const tabs: Array<{ id: DetailsTab; title: string }> = [{ id: 'overview', title: 'Overview' }, { id: 'settings', title: 'Settings' }, { id: 'performance', title: 'Performance' }]

export function ModelDetailsDrawer({ model, status, blocked, onClose, onBenchmark }: { model: EngineInstalledModel; status: EngineStatus | null; blocked: boolean; onClose: () => void; onBenchmark?: () => void }) {
  const tx = useAuthoredText()
  const [tab, setTab] = useState<DetailsTab>('overview')
  const [editing, setEditing] = useState<ModelProfileEditorState>({ dirty: false, busy: false })
  const [confirmClose, setConfirmClose] = useState(false)
  const [previewBusy, setPreviewBusy] = useState(false)
  const controller = useRef<ModelProfileEditorControls | null>(null)
  const tabList = useRef<HTMLDivElement>(null)
  const closePrompt = useRef<HTMLDivElement>(null)
  const settingsPanel = useRef<HTMLElement>(null)
  const returnToSettings = useRef(false)
  const keepEditing = () => { returnToSettings.current = true; setConfirmClose(false); setTab('settings') }
  useEffect(() => {
    if (confirmClose) {
      closePrompt.current?.scrollIntoView({ block: 'nearest' })
      closePrompt.current?.querySelector<HTMLElement>('button[data-keep-editing="true"]:not(:disabled)')?.focus({ preventScroll: true })
    } else if (returnToSettings.current && tab === 'settings') {
      returnToSettings.current = false
      const field = settingsPanel.current?.querySelector<HTMLElement>('#model-profile-contextTokens:not(:disabled)') ?? settingsPanel.current?.querySelector<HTMLElement>('input:not(:disabled),select:not(:disabled),button:not(:disabled)')
      field?.scrollIntoView({ block: 'nearest' })
      field?.focus({ preventScroll: true })
    }
  }, [confirmClose, tab])
  const stateChange = useCallback((state: ModelProfileEditorState) => setEditing(state), [])
  const requestClose = () => {
    if (controller.current?.isBusy()) return
    if (confirmClose) { keepEditing(); return }
    if (controller.current?.hasChanges()) setConfirmClose(true)
    else onClose()
  }
  const saveAndClose = async () => { setTab('settings'); if (controller.current && await controller.current.save()) onClose() }
  const changeTab = (next: DetailsTab) => { if (!controller.current?.isBusy()) { setConfirmClose(false); setTab(next) } }
  const tabKey = (event: KeyboardEvent<HTMLButtonElement>) => {
    const index = tabs.findIndex((entry) => entry.id === tab)
    const next = event.key === 'ArrowRight' ? (index + 1) % tabs.length : event.key === 'ArrowLeft' ? (index + tabs.length - 1) % tabs.length : event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : null
    if (next == null) return
    event.preventDefault()
    if (controller.current?.isBusy()) return
    changeTab(tabs[next].id)
    tabList.current?.querySelector<HTMLButtonElement>(`[data-tab="${tabs[next].id}"]`)?.focus()
  }
  const current = status?.model === model.repo
  const liveContext = current && status?.state === 'ready' ? status.contextTokens : null
  const title = model.title ?? repoDisplayName(model.repo)
  const source = model.source === 'library' ? 'Linked folder' : model.source === 'cache' ? 'Hugging Face cache' : model.source === 'download' ? 'Downloaded by TensorFold' : 'Installed locally'
  const benchmarkBlocked = blocked || editing.busy || editing.dirty || previewBusy || cachedBenchmarkModels([model]).length === 0
  const benchmark = () => { if (!benchmarkBlocked && !controller.current?.isBusy() && !controller.current?.hasChanges()) onBenchmark?.() }
  return <ModelsDrawer title={title} labelId="model-details-title" description={tx("Settings and measured performance for this model.")} onClose={requestClose} closeDisabled={editing.busy}>
    {confirmClose && <div ref={closePrompt} role="alert" className="mb-5 space-y-3 rounded-xl border border-warn/30 bg-warn/10 p-4"><p className="text-[14px] font-medium">{tx("Save your settings before closing?")}</p><p className="text-[13px] leading-relaxed text-ink-2">{tx("Your changes are still a draft; they have not been applied to this model.")}</p><div className="flex flex-wrap gap-2"><Button size="sm" disabled={blocked || previewBusy || editing.busy} onClick={() => void saveAndClose()}>{tx("Save & close")}</Button><Button size="sm" variant="danger" disabled={editing.busy} onClick={() => { if (controller.current?.isBusy()) return; controller.current?.discard(); onClose() }}>{tx("Discard changes")}</Button><Button size="sm" variant="ghost" disabled={editing.busy} data-keep-editing="true" onClick={keepEditing}>{tx("Keep editing")}</Button></div></div>}
    <div ref={tabList} role="tablist" aria-label={tx("Model details")} className="mb-5 grid grid-cols-3 gap-1 rounded-xl bg-card-2 p-1">{tabs.map((entry) => <button key={entry.id} type="button" role="tab" data-tab={entry.id} id={`model-tab-${entry.id}`} aria-selected={tab === entry.id} aria-controls={`model-panel-${entry.id}`} tabIndex={tab === entry.id ? 0 : -1} disabled={editing.busy} onKeyDown={tabKey} onClick={() => changeTab(entry.id)} className={`min-h-10 min-w-0 rounded-lg px-2 text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-accent ${tab === entry.id ? 'bg-card text-ink shadow-card' : 'text-ink-2 hover:text-ink'}`}>{entry.title}{entry.id === 'settings' && editing.dirty && <span aria-label={tx("Unsaved changes")} className="ml-1 text-accent">•</span>}</button>)}</div>
    <section role="tabpanel" id="model-panel-overview" aria-labelledby="model-tab-overview" hidden={tab !== 'overview'} className="space-y-5">
      <p role="status" className={`rounded-xl px-3 py-3 text-[13px] leading-relaxed ${model.fitsMemory === false ? 'bg-warn/10 text-warn' : 'bg-card-2 text-ink-2'}`}>{model.fitsMemory === false ? tx("This model exceeds the available memory budget on this Mac.") : current && status?.state === 'ready' ? tx("Ready to use on this Mac.") : model.fitsMemory === true ? tx("Fits the current model memory budget on this Mac.") : tx("Memory compatibility will be verified when loading.")}</p>
      <dl className="grid grid-cols-2 gap-x-5 gap-y-5"><Fact label={tx("Model size")} value={model.weightsBytes == null ? 'Not reported' : formatBytes(model.weightsBytes)} /><Fact label={tx("Estimated memory")} value={model.memoryFloorBytes == null ? 'Not reported' : formatBytes(model.memoryFloorBytes)} /><Fact label={tx("Model context capacity")} value={model.contextTokens ? `${model.contextTokens.toLocaleString(currentIntlLocale())} tokens` : 'Not reported'} /><Fact label={tx("Running context window")} value={liveContext ? `${liveContext.toLocaleString(currentIntlLocale())} tokens` : 'Reported when this model is ready'} /></dl>
      <p className="text-[13px] leading-relaxed text-ink-2">{tx("The running window can be smaller than the model capacity; choose a per-model profile in Settings to balance context and memory.")}</p>
      {current && status?.message && <p role={status.state === 'failed' ? 'alert' : 'status'} className="break-words text-[13px] text-ink-2">{status.message}</p>}
      <ModelQualificationPanel route={{ providerId: 'tensorfold', accountId: null, model: model.repo }} disabled={blocked || editing.busy || previewBusy || !current || status?.state !== 'ready'} unavailableReason={!current || status?.state !== 'ready' ? 'Load this model explicitly before checking its capabilities; a check never starts a model automatically.' : undefined} />
      <details className="rounded-xl border border-line p-4"><summary className="cursor-pointer text-[13px] font-medium focus-visible:outline-2 focus-visible:outline-accent">{tx("Source and technical details")}</summary><dl className="mt-4 space-y-4"><Fact label={tx("Source")} value={source} /><Fact label={tx("Model identifier")} value={model.repo} /><Fact label={tx("Family")} value={model.family || 'Not reported'} /><Fact label={tx("Model type")} value={model.modelType || 'Not reported'} /><Fact label={tx("Weight format")} value={model.format ?? 'Not reported'} />{model.localPath && <Fact label={tx("Local folder")} value={model.localPath} />}<Fact label={tx("Draft acceleration")} value={model.drafter ? `${model.drafter}${model.drafterInstalled ? ' · installed' : ' · not installed'}` : 'No companion drafter reported'} /></dl><p className="mt-4 text-[12px] leading-relaxed text-ink-2">{tx("Quantization is not reported separately by this engine; check the model identifier and its source details.")}</p></details>
    </section>
    <section ref={settingsPanel} role="tabpanel" id="model-panel-settings" aria-labelledby="model-tab-settings" hidden={tab !== 'settings'}><ModelProfileEditor model={model} blocked={blocked || previewBusy} embedded controllerRef={controller} onStateChange={stateChange} onClose={requestClose} /></section>
    <section role="tabpanel" id="model-panel-performance" aria-labelledby="model-tab-performance" hidden={tab !== 'performance'}><ModelPreviewPanel models={[model]} fixedModel={model.repo} activeModel={status?.model ?? null} locked={blocked || editing.busy || editing.dirty} lockReason={editing.dirty ? "Save or cancel your Settings edits before measuring this model." : undefined} onBusyChange={setPreviewBusy} />{onBenchmark && <div className="mt-5 space-y-2 border-t border-line pt-4"><Button size="sm" disabled={benchmarkBlocked} onClick={benchmark}>{tx("Open repeatable benchmarks")}</Button><p className="text-[12px] leading-relaxed text-ink-3">{tx("Use fixed community-v1 code/chat fixtures with five measured samples per cell; save settings edits and finish local work first, and the model must be cached and confirmed to fit this Mac.").replace('community-v1', 'community-v2')}</p></div>}</section>
  </ModelsDrawer>
}
function Fact({ label, value }: { label: string; value: string }) { return <div className="min-w-0"><dt className="text-[12px] text-ink-2">{label}</dt><dd className="mt-1 break-words text-[14px] leading-relaxed text-ink [overflow-wrap:anywhere]">{value}</dd></div> }
