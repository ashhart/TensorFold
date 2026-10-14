import type { BrowserWindow, Menu, MenuItemConstructorOptions, Tray } from 'electron'
import type { EngineStatus, Settings } from '../../agent/types'

export type DesktopMode = 'minimal' | 'full'
export type DesktopTarget = 'overview' | 'models' | 'api'
type DesktopWindow = Pick<BrowserWindow, 'isDestroyed' | 'isMinimized' | 'restore' | 'show' | 'hide' | 'focus' | 'on' | 'webContents'>
type DesktopTray = Pick<Tray, 'isDestroyed' | 'destroy' | 'setToolTip' | 'setContextMenu' | 'on'>

export interface DesktopModeDeps {
  headless: boolean
  getSettings(): Settings | undefined
  getWindow(): DesktopWindow | null
  createWindow(): void
  createTray(): DesktopTray
  buildMenu(items: MenuItemConstructorOptions[]): Menu
  showDock(): void
  hideDock(): void
  changeMode(mode: DesktopMode): Promise<unknown>
  engineStatus(): Promise<EngineStatus | null>
  startModel(repo: string): Promise<unknown>
  stopModel(): Promise<unknown>
  activeLocalRuns(): number
  learningBusy(): boolean
  navigate(target: DesktopTarget): void
  quit(): void
  reportError(message: string): void
}

export function normalizeDesktopMode(value: unknown): DesktopMode {
  return value === 'minimal' ? 'minimal' : 'full'
}

// All visibility decisions require a live tray. An unavailable system tray must leave a
// normal, reachable window and dock icon, including on a completed Minimal-mode restart.
export function createDesktopModeRuntime(deps: DesktopModeDeps) {
  let tray: DesktopTray | null = null
  let backgroundOnLaunch = false
  let quitting = false
  let status: EngineStatus | null = null
  let polling: ReturnType<typeof setInterval> | null = null
  let refreshing = false
  let modelOperation = false
  let lastMenuSignature = ''
  let pendingTarget: DesktopTarget | null = null
  let trayErrorReported = false

  const trayAvailable = () => {
    try { return !!tray && !tray.isDestroyed() } catch { return false }
  }
  const completedMinimal = () => {
    const settings = deps.getSettings()
    return normalizeDesktopMode(settings?.desktopMode) === 'minimal' && settings?.desktopOnboardingComplete === true
  }
  const canBackground = () => !deps.headless && completedMinimal() && trayAvailable() && !quitting

  function report(error: unknown) {
    deps.reportError(error instanceof Error ? error.message : 'TensorFold could not complete this action.')
  }

  function show(target?: DesktopTarget) {
    if (deps.headless || quitting) return
    backgroundOnLaunch = false
    deps.showDock()
    let win = deps.getWindow()
    let created = false
    if (!win || win.isDestroyed()) {
      if (target) pendingTarget = target
      deps.createWindow()
      created = true
      win = deps.getWindow()
    }
    if (!win || win.isDestroyed()) return
    if (win.isMinimized()) win.restore()
    win.show()
    win.focus()
    if (target && !created) {
      if (win.webContents.isLoadingMainFrame()) pendingTarget = target
      else deps.navigate(target)
    }
  }

  function hide(): { hidden: boolean } {
    const win = deps.getWindow()
    if (!canBackground() || !win || win.isDestroyed()) return { hidden: false }
    win.hide()
    deps.hideDock()
    return { hidden: true }
  }

  function dropTray() {
    if (polling) clearInterval(polling)
    polling = null
    try { if (trayAvailable()) tray!.destroy() } catch { /* already destroyed */ }
    tray = null
    lastMenuSignature = ''
  }

  async function runModelAction() {
    if (modelOperation) return
    modelOperation = true
    updateMenu()
    try {
      // Re-read immediately before mutating the supervised engine; menu state may be stale.
      const current = await deps.engineStatus()
      if (deps.activeLocalRuns() > 0) throw new Error('Finish or stop local chats and workers before changing the running model.')
      if (deps.learningBusy()) throw new Error('Wait for Living Weights to finish saving before changing the running model.')
      if (current?.state === 'ready' || current?.state === 'starting') await deps.stopModel()
      else {
        const selected = deps.getSettings()?.activeProviderId === 'tensorfold' ? deps.getSettings()?.activeModel : null
        if (!selected) { show('models'); return }
        await deps.startModel(selected)
      }
    } catch (error) { report(error) }
    finally { modelOperation = false; await refreshStatus(); updateMenu() }
  }

  function updateMenu() {
    if (!trayAvailable() || quitting) return
    const settings = deps.getSettings()
    const selected = settings?.activeProviderId === 'tensorfold' ? settings.activeModel : null
    const active = status?.state === 'ready' || status?.state === 'starting'
    const runs = deps.activeLocalRuns()
    const learningBusy = deps.learningBusy()
    const stateLabel = status?.state === 'ready' ? 'Server running on this Mac'
      : status?.state === 'starting' ? 'Starting local model…'
      : status?.state === 'failed' ? 'Local server needs attention'
      : 'Local server stopped'
    const signature = JSON.stringify([stateLabel, status?.model, selected, modelOperation, runs > 0, learningBusy])
    if (signature === lastMenuSignature) return
    const items: MenuItemConstructorOptions[] = [
      { label: stateLabel, enabled: false },
      ...(status?.model ? [{ label: status.model, enabled: false }] : []),
      { type: 'separator' },
      { label: 'Open Minimal', click: () => show('overview') },
      { label: 'Models and downloads…', click: () => show('models') },
      { label: 'API setup…', click: () => show('api') },
      { label: modelOperation ? 'Updating local server…' : active ? 'Stop local server' : 'Start selected model', enabled: !modelOperation && !learningBusy && runs === 0 && (active || !!selected) && status?.available !== false, click: () => { void runModelAction() } },
      { type: 'separator' },
      { label: 'Open Full workspace', click: () => { void deps.changeMode('full').then(() => show()).catch(report) } },
      { type: 'separator' },
      { label: 'Quit TensorFold', click: () => deps.quit() },
    ]
    try {
      tray!.setToolTip(`TensorFold · ${stateLabel}`)
      tray!.setContextMenu(deps.buildMenu(items))
      lastMenuSignature = signature
    } catch {
      // A tray can disappear after creation. Recover a visible app rather than strand it.
      dropTray()
      backgroundOnLaunch = false
      show()
    }
  }

  async function refreshStatus() {
    if (refreshing || quitting || !trayAvailable()) return
    refreshing = true
    try { status = await deps.engineStatus() } catch { status = null }
    finally { refreshing = false; updateMenu() }
  }

  function applySettings(options: { startup?: boolean } = {}) {
    if (deps.headless || quitting) return
    if (normalizeDesktopMode(deps.getSettings()?.desktopMode) === 'full') {
      backgroundOnLaunch = false
      dropTray()
      deps.showDock()
      if (!options.startup) show()
      return
    }
    if (!trayAvailable()) {
      try {
        tray = deps.createTray()
        if (!trayAvailable()) throw new Error('The system menu bar is unavailable.')
        tray!.on('click', () => show('overview'))
        lastMenuSignature = ''
      } catch {
        dropTray()
        backgroundOnLaunch = false
        deps.showDock()
        if (!trayErrorReported) {
          trayErrorReported = true
          deps.reportError('The system menu bar is unavailable, so TensorFold will keep its window open.')
        }
        return
      }
    }
    backgroundOnLaunch = options.startup === true && canBackground()
    if (backgroundOnLaunch) deps.hideDock()
    else deps.showDock()
    updateMenu()
    if (!polling) {
      polling = setInterval(() => { void refreshStatus() }, 5000)
      polling.unref?.()
    }
    void refreshStatus()
  }

  function attachWindow(win: DesktopWindow) {
    win.on('close', (event) => {
      if (deps.getWindow() !== win || !canBackground()) return
      event.preventDefault()
      hide()
    })
    win.webContents.on('did-finish-load', () => {
      if (deps.getWindow() !== win || !pendingTarget) return
      const target = pendingTarget
      pendingTarget = null
      deps.navigate(target)
    })
  }

  function dispose() {
    quitting = true
    dropTray()
  }

  return { applySettings, attachWindow, hide, show, dispose, refreshStatus, canBackground, shouldShowWindow: () => !backgroundOnLaunch }
}
