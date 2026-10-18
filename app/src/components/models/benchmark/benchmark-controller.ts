import type { ModelBenchmarkInput, ModelBenchmarkSnapshot } from '../../../../shared/model-benchmark'
import { benchmarkActive, benchmarkBasicSummary } from './benchmark-display'

export interface BenchmarkPort {
  status(id?: string): Promise<ModelBenchmarkSnapshot | null>
  history(): Promise<ModelBenchmarkSnapshot[]>
  start(input: ModelBenchmarkInput): Promise<ModelBenchmarkSnapshot>
  cancel(id: string): Promise<boolean>
  export(id: string): Promise<{ saved: boolean }>
  copy(text: string): Promise<{ ok: boolean }>
}
export interface BenchmarkViewState {
  selected: ModelBenchmarkSnapshot | null
  active: ModelBenchmarkSnapshot | null
  history: ModelBenchmarkSnapshot[]
  loading: boolean
  action: 'start' | 'cancel' | 'export' | 'copy' | null
  error: string
  notice: string
  cancellationRequested: boolean
}
const initial = (): BenchmarkViewState => ({ selected: null, active: null, history: [], loading: true, action: null, error: '', notice: '', cancellationRequested: false })

/** One mounted drawer owns reads/actions, while the host owns the durable running job. */
export function createBenchmarkController(port: BenchmarkPort, changed: (state: BenchmarkViewState) => void) {
  let state = initial(), alive = true, epoch = 0, reading: number | null = null, acting = false
  const publish = (patch: Partial<BenchmarkViewState>) => { state = { ...state, ...patch }; if (alive) changed(state) }
  const newest = (run: ModelBenchmarkSnapshot) => {
    return [state.active, state.selected, ...state.history].reduce<ModelBenchmarkSnapshot>((latest, item) => item?.id === run.id && item.revision > latest.revision ? item : latest, run)
  }
  const read = async (historyRequested: boolean) => {
    if (!alive || reading === epoch || acting) return
    const ownEpoch = epoch
    reading = ownEpoch
    if (historyRequested) publish({ loading: true, error: '' })
    try {
      const current = await port.status()
      if (!alive || ownEpoch !== epoch) return
      const latest = current ? newest(current) : null
      let history = state.history
      let historyFailed = false
      if (historyRequested || state.active && !benchmarkActive(latest)) {
        try { history = (await port.history()).map((run) => latest?.id === run.id && latest.revision > run.revision ? latest : newest(run)) }
        catch { historyFailed = true }
      }
      if (!alive || ownEpoch !== epoch) return
      const active = benchmarkActive(latest) ? latest : null
      const selected = !state.selected || state.selected.id === state.active?.id || state.selected.id === latest?.id
        ? latest ?? history[0] ?? null : history.find((run) => run.id === state.selected?.id) ?? state.selected
      const notice = state.active && !active ? latest ? `The active benchmark ${latest.state === 'complete' ? 'finished' : latest.state === 'cancelled' ? 'was cancelled' : 'stopped with an issue'}; open its result to check restoration and measurements.` : 'The host no longer reports the active job; check saved history before starting again.' : state.notice
      publish({ active, selected, history, notice, loading: false, cancellationRequested: active?.id === state.active?.id && state.cancellationRequested, error: historyFailed ? 'Saved benchmark history could not be read; active run status is still shown, and you can refresh to retry history.' : '' })
    } catch { if (alive && ownEpoch === epoch) publish({ loading: false, error: 'Benchmark status could not be read; a running job may still be active, so refresh before starting another run.' }) }
    finally { if (reading === ownEpoch) reading = null }
  }
  const action = async (kind: BenchmarkViewState['action'], work: () => Promise<void>) => {
    if (!alive || acting) return
    acting = true; epoch++; publish({ action: kind, error: '', notice: '' })
    try { await work() }
    catch { if (alive) publish({ error: kind === 'start' ? 'The benchmark could not start; finish local chat/API work, check the cached model and engine, then refresh and retry.' : 'This benchmark action failed; refresh its saved status before trying again.' }) }
    finally { acting = false; if (alive) publish({ action: null }) }
  }
  return {
    snapshot: () => state,
    activate: () => { alive = true },
    dispose: () => { alive = false; epoch++ },
    refresh: () => read(true),
    poll: () => read(false),
    select(id: string) { const selected = [state.active, ...state.history].find((run) => run?.id === id); if (selected) publish({ selected, error: '', notice: '' }) },
    start(input: ModelBenchmarkInput) {
      if (benchmarkActive(state.active) || state.loading || state.error) return Promise.resolve()
      return action('start', async () => { const run = await port.start(input); if (alive) publish({ selected: run, active: benchmarkActive(run) ? run : null, loading: false, cancellationRequested: false }) })
    },
    cancel() {
      const id = state.active?.id
      if (!id || state.active?.state === 'restoring' || state.cancellationRequested) return Promise.resolve()
      return action('cancel', async () => { const accepted = await port.cancel(id); if (alive) publish({ cancellationRequested: accepted, notice: accepted ? 'Cancellation requested; the engine may need time to restore its previous model and profile.' : 'This run is no longer active; refresh to see its final state.' }) })
    },
    export() {
      const run = state.selected
      if (!run?.receiptAvailable || benchmarkActive(run)) return Promise.resolve()
      return action('export', async () => { const result = await port.export(run.id); if (alive) publish({ notice: result.saved ? 'Reviewed benchmark JSON exported locally.' : 'Export cancelled; no benchmark file was saved.' }) })
    },
    copy() {
      const run = state.selected
      if (!run || benchmarkActive(run)) return Promise.resolve()
      return action('copy', async () => { if (!(await port.copy(benchmarkBasicSummary(run))).ok) throw new Error('Clipboard unavailable'); if (alive) publish({ notice: 'Basic benchmark results copied.' }) })
    },
  }
}
