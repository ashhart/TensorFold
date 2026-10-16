import { useEffect, useSyncExternalStore } from 'react'
import { api } from './ipc'
import type { EngineLearningSnapshot } from './types'

export const LEARNING_MODEL = 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit'
export const learningBusy = (snapshot: EngineLearningSnapshot | null) => !!snapshot && ['preparing', 'teaching', 'cancelling'].includes(snapshot.state)
export const learningError = (error: unknown) => String((error as Error)?.message ?? error).replace(/^Error invoking remote method '[^']*':\s*(Error:\s*)?/, '')

type View = { snapshot: EngineLearningSnapshot | null; error: string | null }
let view: View = { snapshot: null, error: null }
let started = false
let revision = 0
let request: Promise<void> | null = null
const listeners = new Set<() => void>()
function publish(next: View) { view = next; for (const notify of listeners) notify() }
function start() {
  if (started) return
  started = true
  api.onEngineEvent((event) => {
    if (event.kind !== 'learning') return
    revision++
    publish({ snapshot: event.snapshot, error: null })
  })
}
export function refreshLearning(): Promise<void> {
  start()
  if (request) return request
  const before = revision
  const pending = api.engineLearning().then((snapshot) => {
    // A response started before a pushed update must not restore an old model/job.
    if (before === revision) publish({ snapshot, error: null })
  }).catch((error) => {
    if (before === revision) publish({ ...view, error: learningError(error) })
  })
  request = pending
  return pending.finally(() => { if (request === pending) request = null })
}
export function useLearning(poll = false): View {
  start()
  useEffect(() => {
    void refreshLearning()
    if (!poll) return
    const timer = setInterval(() => { if (document.visibilityState === 'visible') void refreshLearning() }, 2_000)
    return () => clearInterval(timer)
  }, [poll])
  return useSyncExternalStore((notify) => { listeners.add(notify); return () => { listeners.delete(notify) } }, () => view)
}
