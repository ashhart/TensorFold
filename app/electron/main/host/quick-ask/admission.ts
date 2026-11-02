let reserve: (() => { release(): void }) | null = null
let accepting = true
const active = new Map<AbortController, Promise<void>>()
export function configureAskAdmission(value: () => { release(): void }): void { reserve = value }
export function reserveAskOperation(): { release(): void } {
  if (!accepting || !reserve) throw new Error('Ask TensorFold is stopping or the workspace is unavailable.')
  return reserve()
}
export function trackAskOperation(signal: AbortSignal) {
  if (active.size >= 4) throw new Error('The Ask TensorFold host is busy. Try again when a current answer finishes.')
  const lease = reserveAskOperation(), controller = new AbortController()
  let resolve!: () => void
  active.set(controller, new Promise<void>(done => { resolve = done }))
  return { signal: AbortSignal.any([signal, controller.signal]), release: () => { active.delete(controller); resolve(); lease.release() } }
}
export async function quiesceAskBackend(): Promise<void> {
  accepting = false
  const pending = [...active.values()]
  for (const controller of active.keys()) controller.abort()
  await Promise.allSettled(pending)
}
export function resumeAskBackend(): void { accepting = true }
