import { describeComputerAction, formatObservation, matchesApp, type ComputerApp, type ComputerObservation, type ComputerPoint, type ComputerUseInput } from './actions'

// The native helper's commands the runner needs (electron/agent/computer-use/helper.ts implements them).
export interface ComputerControl {
  permissions(): Promise<{ accessibility: boolean; screenRecording: boolean }>
  requestPermissions(input: { accessibility?: boolean; screenRecording?: boolean }): Promise<{ accessibility: boolean; screenRecording: boolean }>
  apps(): Promise<{ apps: Array<ComputerApp & { active: boolean }> }>
  frontmost(): Promise<{ app: ComputerApp; window: ComputerObservation['window'] }>
  observe(input: { maxElements?: number; pid?: number }): Promise<ComputerObservation>
  click(input: { x?: number; y?: number; element?: number; button?: 'left' | 'right'; count?: 1 | 2 }): Promise<unknown>
  drag(input: { from: ComputerPoint; to: ComputerPoint }): Promise<unknown>
  scroll(input: { x?: number; y?: number; element?: number; dx?: number; dy?: number }): Promise<unknown>
  type(input: { text: string }): Promise<unknown>
  key(input: { keys: string }): Promise<unknown>
  openApp(input: { name?: string; bundleId?: string }): Promise<unknown>
}

// Shown while TensorFold is operating another app; its Stop ends the thread's run.
export interface ComputerIndicator {
  show(label: string, onStop: () => void): void
  hide(): void
}

export interface ComputerUseRunContext { threadId: string | null; signal?: AbortSignal; cancel?: () => void }

export interface ComputerUseResult { ok: boolean; output: string; title: string; meta?: string }

// Helper refusal codes, rewritten as instructions the model can act on.
const CODE_ADVICE: Record<string, string> = {
  own_app: 'TensorFold cannot operate its own window. Use open_app to bring the target app to the front.',
  no_accessibility: 'TensorFold needs Accessibility permission. Ask the user to turn TensorFold on in System Settings > Privacy & Security > Accessibility, then try again.',
  secure_field: 'The focused field is a password field. TensorFold never types into password fields; ask the user to type it themselves.',
  stale_element: 'That element id is from an older observation. Call observe again and use the new ids.',
  bad_key: 'Unknown key name. Use names like "return", "tab", "escape", "cmd+l", "cmd+shift+t", "down".',
  bad_point: 'That point is outside every display. Use coordinates from the latest observe.',
}

const sleep = (ms: number, signal?: AbortSignal) => new Promise<void>((resolve, reject) => {
  if (signal?.aborted) return reject(signal.reason ?? new Error('Stopped.'))
  const timer = setTimeout(resolve, ms)
  signal?.addEventListener('abort', () => { clearTimeout(timer); reject(signal.reason ?? new Error('Stopped.')) }, { once: true })
})

export function createComputerUseRunner(deps: { control: () => ComputerControl; indicator?: ComputerIndicator; settleMs?: number; idleHideMs?: number }) {
  const settleMs = deps.settleMs ?? 350
  let active = 0
  let hideTimer: ReturnType<typeof setTimeout> | null = null

  async function summary(control: ComputerControl): Promise<string> {
    const front = await control.frontmost()
    return `Now in front: ${front.app.name}${front.window?.title ? `, window ${JSON.stringify(front.window.title)}` : ''}. Call observe to read the controls.`
  }

  async function requireFront(control: ComputerControl, app: string): Promise<string | null> {
    const front = await control.frontmost()
    if (matchesApp(app, front.app)) return null
    return `${app} is not the app in front (${front.app.name} is). Call open_app for ${app} first, then observe.`
  }

  async function act(control: ComputerControl, input: ComputerUseInput, ctx: ComputerUseRunContext): Promise<{ ok: boolean; text: string }> {
    const target = input.element != null ? { element: input.element } : input.x != null && input.y != null ? { x: input.x, y: input.y } : {}
    switch (input.action) {
      case 'observe': {
        const running = (await control.apps()).apps.find((candidate) => matchesApp(input.app, candidate))
        if (!running) return { ok: false, text: `${input.app} is not running. Call open_app first.` }
        return { ok: true, text: formatObservation(await control.observe({ pid: running.pid, maxElements: 150 })) }
      }
      case 'open_app':
        await control.openApp(input.app.includes('.') && !input.app.includes(' ') ? { bundleId: input.app } : { name: input.app })
        await sleep(settleMs * 2, ctx.signal)
        return { ok: true, text: await summary(control) }
      case 'wait':
        await sleep(input.seconds! * 1000, ctx.signal)
        return { ok: true, text: await summary(control) }
    }
    const notFront = await requireFront(control, input.app)
    if (notFront) return { ok: false, text: notFront }
    if (input.action === 'click') await control.click({ ...target, button: 'left', count: 1 })
    else if (input.action === 'double_click') await control.click({ ...target, button: 'left', count: 2 })
    else if (input.action === 'right_click') await control.click({ ...target, button: 'right', count: 1 })
    else if (input.action === 'scroll') await control.scroll({ ...target, dx: input.dx, dy: input.dy })
    else if (input.action === 'drag') await control.drag({ from: { x: input.x!, y: input.y! }, to: input.to! })
    else if (input.action === 'type') await control.type({ text: input.text! })
    else if (input.action === 'key') await control.key({ keys: input.keys! })
    await sleep(settleMs, ctx.signal)
    return { ok: true, text: `Done: ${describeComputerAction(input)}. ${await summary(control)}` }
  }

  async function run(input: ComputerUseInput, ctx: ComputerUseRunContext): Promise<ComputerUseResult> {
    const title = describeComputerAction(input)
    const control = deps.control()
    let counted = false
    try {
      const granted = await control.permissions()
      if (!granted.accessibility) {
        await control.requestPermissions({ accessibility: true }).catch(() => null)
        return { ok: false, title, output: CODE_ADVICE.no_accessibility }
      }
      active++
      counted = true
      if (hideTimer) { clearTimeout(hideTimer); hideTimer = null }
      deps.indicator?.show(input.app, () => { deps.indicator?.hide(); ctx.cancel?.() })
      const result = await act(control, input, ctx)
      return { ok: result.ok, title, output: result.text, meta: input.app }
    } catch (error) {
      const code = (error as { code?: string }).code
      const advice = code ? CODE_ADVICE[code] : undefined
      return { ok: false, title, output: advice ?? `Computer use failed: ${(error as Error).message}` }
    } finally {
      if (counted && --active === 0) hideTimer = setTimeout(() => { hideTimer = null; if (!active) deps.indicator?.hide() }, deps.idleHideMs ?? 6000)
    }
  }

  return { run, isActive: () => active > 0 }
}
