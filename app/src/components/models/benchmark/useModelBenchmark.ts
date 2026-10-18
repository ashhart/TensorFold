import { useEffect, useRef, useState } from 'react'
import { api } from '../../../lib/ipc'
import { createBenchmarkController, type BenchmarkViewState } from './benchmark-controller'

export function useModelBenchmark() {
  const [state, setState] = useState<BenchmarkViewState>({ selected: null, active: null, history: [], loading: true, action: null, error: '', notice: '', cancellationRequested: false })
  const owned = useRef<ReturnType<typeof createBenchmarkController> | null>(null)
  if (!owned.current) owned.current = createBenchmarkController({ status: (id) => api.engineBenchmarkStatus(id), history: () => api.engineBenchmarkHistory(), start: (input) => api.engineStartBenchmark(input), cancel: (id) => api.engineCancelBenchmark(id), export: (id) => api.engineExportBenchmark(id), copy: (text) => api.writeClipboard(text) }, setState)
  const controller = owned.current
  useEffect(() => { controller.activate(); void controller.refresh(); return () => controller.dispose() }, [controller])
  useEffect(() => {
    if (!state.active) return
    let current = true
    let timer: ReturnType<typeof setTimeout>
    const poll = async () => { await controller.poll(); if (current && controller.snapshot().active) timer = setTimeout(poll, 1200) }
    timer = setTimeout(poll, 1200)
    return () => { current = false; clearTimeout(timer) }
  }, [controller, state.active?.id])
  return { state, controller }
}
