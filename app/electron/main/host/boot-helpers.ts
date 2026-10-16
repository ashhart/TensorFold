import { app as electronApp, BrowserWindow, dialog, nativeTheme } from 'electron'
import { promises as fs } from 'node:fs'
import type { Settings } from '../../agent/types'
import { assertRemoteOrchestratorUrl } from '../../agent/security'
import { normalizeChangedRemoteDestinations } from '../../agent/settings-save'
import { migrateLegacyElectronUserData } from './electron-paths'
import { terminateFailedHeadlessBoot } from './headless-bootstrap'
import { APP_NAME, PROFILE_APP_NAME, HEADLESS_MODE, LEGACY_APP_NAMES, isolatedElectronPathPlan } from '../early-electron-paths'

export type ApplyThemeFn = () => void

/**
 * Returns a no-arg `applyTheme()` that mirrors the current settings theme onto Electron's
 * nativeTheme. `getSettings` is an accessor because settings can be reassigned (store.settings
 * is replaced wholesale on save), so reading it at call time keeps the live value.
 *
 * Extracted verbatim from main-app.ts.
 */
export function createApplyTheme(getSettings: () => Settings | undefined): ApplyThemeFn {
  return function applyTheme() {
    const settings = getSettings()
    if (!settings) return
    nativeTheme.themeSource = settings.theme
  }
}

export interface EnsureWindowVisibleDeps {
  installSessionHardening(): unknown
  createWindow(): unknown
}

/**
 * Guarantees a window exists even when the boot sequence fails partway. createWindow()
 * does not depend on store.settings (it reads only its own deps), so it is always safe
 * to call here once Electron is ready.
 *
 * Extracted verbatim from main-app.ts; HEADLESS_MODE is read directly (singleton).
 */
export function createEnsureWindowVisible(deps: EnsureWindowVisibleDeps): () => void {
  const { installSessionHardening, createWindow } = deps
  return function ensureWindowVisible() {
    if (HEADLESS_MODE) return
    if (BrowserWindow.getAllWindows().length > 0) return
    try {
      installSessionHardening()
      createWindow()
    } catch (err) {
      console.error('[boot] failed to create window during degraded startup', err)
    }
  }
}

export interface HandleBootFailureDeps {
  ensureWindowVisible(): void
  getSettingsRef(): { settings: Settings | undefined }
  setSettings(settings: Settings): void
  defaultSettings(): Settings
}

/**
 * A crypto/persistence failure during boot (e.g. loadOrCreateKeySync throwing on a
 * read-only home or full disk) used to reject the whenReady promise, so createWindow()
 * never ran and the app launched windowless. Degrade gracefully instead: log, fall back
 * to default settings so later handlers don't crash, surface a fatal-error dialog, and
 * STILL show a window.
 *
 * Extracted verbatim from main-app.ts; the settings slot is reached via the `deps`
 * accessors so the live `store` identity is mutated, not a copy.
 */
export function createHandleBootFailure(deps: HandleBootFailureDeps): (err: unknown) => void {
  const { ensureWindowVisible, getSettingsRef, setSettings, defaultSettings } = deps
  return function handleBootFailure(err: unknown) {
    console.error('[boot] startup failed before completion', err)
    if (!getSettingsRef().settings) {
      try { setSettings(defaultSettings()) } catch { /* leave undefined; consumers guard */ }
    }
    ensureWindowVisible()
    if (HEADLESS_MODE) {
      // A headless process with no listening control plane is unusable and can mislead a
      // supervisor into treating a dead Orchestrator as healthy. Fail fast and non-zero.
      terminateFailedHeadlessBoot((code) => electronApp.exit(code))
      return
    }
    if (!HEADLESS_MODE) {
      try {
        const detail = err instanceof Error ? (err.stack || err.message) : String(err)
        void dialog.showMessageBox({
          type: 'error',
          title: `${APP_NAME} — startup problem`,
          message: 'TensorFold started but hit an error while loading. Some data may not have loaded; settings changes might not persist.',
          detail,
          buttons: ['Continue'],
        }).catch(() => {})
      } catch { /* dialog best-effort */ }
    }
  }
}

/**
 * One-shot migration of legacy Electron user-data into the isolated app-data path.
 * `app` defaults to Electron's singleton; injected so tests can stub the app paths.
 *
 * Extracted verbatim from main-app.ts.
 */
export async function migrateLegacyUserData(app: typeof electronApp = electronApp): Promise<void> {
  await migrateLegacyElectronUserData({
    isolationPlan: isolatedElectronPathPlan,
    appName: PROFILE_APP_NAME,
    legacyAppNames: LEGACY_APP_NAMES,
    appPaths: {
      getPath: () => app.getPath('appData'),
      setPath: (_name, value) => app.setPath('userData', value),
    },
    files: fs,
  })
}

/**
 * Normalizes the remote-orchestrator destination diff between two settings snapshots,
 * validating each changed URL against the orchestrator URL guard. Extracted verbatim.
 */
export async function normalizeRemoteOrchestratorDestinations(next: Settings, current: Settings): Promise<Settings> {
  return normalizeChangedRemoteDestinations(next, current, assertRemoteOrchestratorUrl)
}
