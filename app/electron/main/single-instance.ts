import { app, BrowserWindow } from 'electron'
import { HEADLESS_MODE } from './early-electron-paths'

// One process per profile. Each process keeps its own store and rewrites threads.json in
// full on every save, so a second copy (the headless orchestrator beside the GUI, or a dev
// build beside the packaged app) would silently overwrite the other's messages. Electron
// keys the lock on userData, which VONTRA_HOME already relocated, so isolated profiles can
// still run side by side.
export function secondInstanceMessage(headless: boolean): string {
  return headless
    ? 'TensorFold is already running on this profile. Quit it first, or set VONTRA_HOME to another folder.'
    : 'TensorFold is already running on this profile; focusing the open window.'
}

if (!app.requestSingleInstanceLock()) {
  console.error(secondInstanceMessage(HEADLESS_MODE))
  process.exit(HEADLESS_MODE ? 1 : 0)
}

app.on('second-instance', () => {
  const win = BrowserWindow.getAllWindows()[0]
  if (!win) return
  if (win.isMinimized()) win.restore()
  win.show()
  win.focus()
})
