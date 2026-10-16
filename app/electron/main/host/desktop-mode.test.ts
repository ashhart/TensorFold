import assert from 'node:assert/strict'
import { createDesktopModeRuntime, normalizeDesktopMode, type DesktopModeDeps } from './desktop-mode'
import { createLearningQuitGuard } from './learning-quit'
import type { EngineStatus, Settings } from '../../agent/types'

type Handler = (...args: any[]) => void
class FakeWindow {
  handlers = new Map<string, Handler>()
  webHandlers = new Map<string, Handler>()
  shown = 0
  hidden = 0
  focused = 0
  destroyed = false
  minimized = false
  loading = false
  webContents = {
    isLoadingMainFrame: () => this.loading,
    on: (name: string, handler: Handler) => this.webHandlers.set(name, handler),
  }
  isDestroyed() { return this.destroyed }
  isMinimized() { return this.minimized }
  restore() { this.minimized = false }
  show() { this.shown++ }
  hide() { this.hidden++ }
  focus() { this.focused++ }
  on(name: string, handler: Handler) { this.handlers.set(name, handler) }
  close() {
    let prevented = false
    this.handlers.get('close')?.({ preventDefault: () => { prevented = true } })
    return prevented
  }
}
class FakeTray {
  destroyed = false
  failMenu = false
  handlers = new Map<string, Handler>()
  menu: any[] = []
  tooltip = ''
  isDestroyed() { return this.destroyed }
  destroy() { this.destroyed = true }
  setToolTip(text: string) { this.tooltip = text }
  setContextMenu(menu: any) { if (this.failMenu) throw new Error('tray removed'); this.menu = menu }
  on(name: string, handler: Handler) { this.handlers.set(name, handler) }
}
const flush = () => new Promise<void>((resolve) => setImmediate(resolve))

function harness(options: { headless?: boolean; complete?: boolean; mode?: 'minimal' | 'full'; failTray?: boolean } = {}) {
  let settings = { desktopMode: options.mode ?? 'minimal', desktopOnboardingComplete: options.complete ?? true, activeProviderId: 'tensorfold', activeModel: 'example/small-model' } as Settings
  let window: FakeWindow | null = new FakeWindow()
  const tray = new FakeTray()
  const errors: string[] = []
  const navigations: string[] = []
  let dockShown = 0
  let dockHidden = 0
  let engineStarts = 0
  let engineStops = 0
  let runs = 0
  let learningBusy = false
  let quitCalls = 0
  let engine = { available: true, state: 'stopped', model: null } as EngineStatus
  const deps: DesktopModeDeps = {
    headless: options.headless ?? false,
    getSettings: () => settings,
    getWindow: () => window as any,
    createWindow: () => { window = new FakeWindow(); runtime.attachWindow(window as any) },
    createTray: () => { if (options.failTray) throw new Error('no tray'); return tray as any },
    buildMenu: (items) => items as any,
    showDock: () => { dockShown++ },
    hideDock: () => { dockHidden++ },
    changeMode: async (mode) => { settings = { ...settings, desktopMode: mode }; runtime.applySettings() },
    engineStatus: async () => engine,
    startModel: async (repo) => { engineStarts++; engine = { ...engine, state: 'ready', model: repo } },
    stopModel: async () => { engineStops++; engine = { ...engine, state: 'stopped', model: null } },
    activeLocalRuns: () => runs,
    learningBusy: () => learningBusy,
    navigate: (target) => { navigations.push(target) },
    quit: () => { quitCalls++ },
    reportError: (error) => { errors.push(error) },
  }
  const runtime = createDesktopModeRuntime(deps)
  runtime.attachWindow(window as any)
  const menu = (label: string) => {
    const item = tray.menu.find((item) => item.label === label)
    assert.ok(item, `menu action exists: ${label}`)
    return item
  }
  return {
    runtime, tray, deps, errors, navigations, menu,
    get window() { return window! },
    get settings() { return settings },
    get counts() { return { dockShown, dockHidden, engineStarts, engineStops, quitCalls } },
    setRuns(count: number) { runs = count },
    setLearning(busy: boolean) { learningBusy = busy },
    setEngine(next: Partial<EngineStatus>) { engine = { ...engine, ...next } },
    setSettings(next: Partial<Settings>) { settings = { ...settings, ...next } },
    loseWindow() { window = null },
  }
}

assert.equal(normalizeDesktopMode('minimal'), 'minimal')
for (const input of [null, undefined, 'MINIMAL', 'background', {}, 1]) assert.equal(normalizeDesktopMode(input), 'full')

// Selecting Minimal during first-run setup cannot strand the unfinished wizard.
{
  const h = harness({ complete: false })
  h.runtime.applySettings({ startup: true })
  assert.equal(h.runtime.shouldShowWindow(), true)
  assert.equal(h.runtime.hide().hidden, false)
  assert.equal(h.window.close(), false)
  assert.equal(h.counts.dockHidden, 0)
  assert.equal(h.counts.engineStarts, 0, 'mode selection never starts or downloads a model')
  h.runtime.dispose()
}

// A completed restart can run quietly, and red close explicitly preserves the supervised server.
{
  const h = harness()
  h.runtime.applySettings({ startup: true })
  await flush()
  assert.equal(h.runtime.shouldShowWindow(), false)
  assert.equal(h.runtime.canBackground(), true)
  assert.equal(h.counts.dockHidden, 1)
  assert.equal(h.window.close(), true)
  assert.equal(h.window.hidden, 1)
  assert.equal(h.counts.engineStops, 0)
  h.menu('API setup…').click()
  assert.equal(h.window.shown, 1)
  assert.equal(h.runtime.shouldShowWindow(), true)
  assert.deepEqual(h.navigations, ['api'])
  h.menu('Models and downloads…').click()
  assert.deepEqual(h.navigations, ['api', 'models'])
  h.menu('Open Full workspace').click()
  await flush()
  assert.equal(h.settings.desktopMode, 'full')
  assert.equal(h.tray.destroyed, true)
  assert.equal(h.runtime.canBackground(), false)
  assert.equal(h.window.close(), false, 'Full returns the normal window lifecycle')
  h.runtime.dispose()
}

// Missing or removed tray support always leaves a reachable window.
{
  const h = harness({ failTray: true })
  h.runtime.applySettings({ startup: true })
  assert.equal(h.runtime.shouldShowWindow(), true)
  assert.equal(h.runtime.canBackground(), false)
  assert.equal(h.runtime.hide().hidden, false)
  assert.equal(h.window.close(), false)
  assert.equal(h.counts.dockHidden, 0)
  assert.equal(h.errors.length, 1)
  h.runtime.dispose()
}
{
  const h = harness()
  h.runtime.applySettings({ startup: true })
  await flush()
  h.tray.failMenu = true
  h.setEngine({ state: 'failed' })
  await h.runtime.refreshStatus()
  assert.equal(h.runtime.canBackground(), false)
  assert.equal(h.runtime.shouldShowWindow(), true)
  assert.equal(h.window.shown, 1)
  h.runtime.dispose()
}

// Menus re-check active work after click, rather than trusting their earlier enabled state.
{
  const h = harness()
  h.runtime.applySettings()
  await flush()
  const start = h.menu('Start selected model')
  assert.equal(start.enabled, true)
  h.setRuns(1)
  start.click()
  await flush()
  assert.equal(h.counts.engineStarts, 0)
  assert.match(h.errors.at(-1)!, /local chats and workers/)
  h.setRuns(0)
  h.setLearning(true)
  start.click()
  await flush()
  assert.equal(h.counts.engineStarts, 0)
  assert.match(h.errors.at(-1)!, /Living Weights/)
  h.setLearning(false)
  start.click()
  await flush()
  assert.equal(h.counts.engineStarts, 1)
  h.menu('Stop local server').click()
  await flush()
  assert.equal(h.counts.engineStops, 1)
  h.runtime.dispose()
}

// Recovery of a destroyed window queues navigation until its trusted renderer has loaded.
{
  const h = harness()
  h.runtime.applySettings()
  await flush()
  h.loseWindow()
  h.runtime.show('models')
  h.window.webHandlers.get('did-finish-load')?.()
  assert.equal(h.window.shown, 1)
  assert.deepEqual(h.navigations, ['models'], 'a replacement renderer receives navigation once, after load')
  h.runtime.dispose()
}

// Explicit tray Quit uses the existing asynchronous checkpoint fence before destroying the tray.
{
  const h = harness()
  h.runtime.applySettings()
  let busy = true
  let finish!: () => void
  let shutdowns = 0
  const saved = new Promise<void>((resolve) => { finish = resolve })
  const guard = createLearningQuitGuard({
    service: () => ({ isLearningBusy: () => busy, prepareToQuit: () => saved }),
    quit: () => h.deps.quit(),
    failed: (error) => { throw error },
  })
  h.deps.quit = () => {
    const deferred = guard({ preventDefault() {} })
    if (!deferred) { h.runtime.dispose(); shutdowns++ }
  }
  h.menu('Quit TensorFold').click()
  assert.equal(h.tray.destroyed, false)
  assert.equal(shutdowns, 0)
  assert.equal(h.window.close(), true)
  busy = false
  finish()
  await flush()
  assert.equal(h.tray.destroyed, true)
  assert.equal(shutdowns, 1)
  assert.equal(h.window.close(), false, 'close interception must not cancel an approved quit')
}
{
  const h = harness({ headless: true })
  h.runtime.applySettings({ startup: true })
  assert.equal(h.runtime.canBackground(), false)
  assert.equal(h.tray.menu.length, 0)
  assert.equal(h.counts.dockShown, 0)
  assert.equal(h.counts.engineStarts, 0)
  h.runtime.dispose()
}
console.log('Desktop Minimal/Full tray, restart, model-action and graceful-quit tests passed')
