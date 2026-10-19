// The boot splash in index.html is static: if the renderer throws while starting, or the main
// process never answers, the window stays on that splash with no explanation. Surface the first
// failure in the static panel and also report a boot that never completes.

export interface BootGuardWindow {
  addEventListener(type: 'error' | 'unhandledrejection', handler: (event: any) => void): void
  setTimeout(handler: () => void, ms: number): number
}

export const BOOT_GUARD_TIMEOUT_MS = 10_000

export function bootFailureText(value: unknown): string {
  if (value && typeof value === 'object' && 'stack' in value && typeof (value as { stack?: unknown }).stack === 'string') {
    return String((value as { stack: string }).stack)
  }
  if (typeof value === 'string' && value.trim()) return value
  if (value instanceof Error) return value.message
  return 'No error detail was provided.'
}

export function installBootGuard(
  win: BootGuardWindow,
  doc: { body: { classList: { contains(name: string): boolean } }; getElementById(id: string): { textContent: string | null; setAttribute(name: string, value: string): void } | null },
): void {
  let shown = false
  const show = (detail: unknown): void => {
    if (shown || doc.body.classList.contains('ready')) return
    shown = true
    const panel = doc.getElementById('boot-failure')
    const text = doc.getElementById('boot-failure-detail')
    if (!panel || !text) return
    text.textContent = bootFailureText(detail)
    panel.setAttribute('data-visible', 'true')
  }
  win.addEventListener('error', (event: any) => show(event?.error ?? event?.message ?? 'Script error'))
  win.addEventListener('unhandledrejection', (event: any) => show(event?.reason ?? 'Unhandled rejection'))
  win.setTimeout(() => {
    if (!doc.body.classList.contains('ready')) {
      show('Still waiting for the workspace after 10 seconds. The renderer may not have loaded its bundle, or the main process did not answer.')
    }
  }, BOOT_GUARD_TIMEOUT_MS)
}
