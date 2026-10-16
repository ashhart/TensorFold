import { app } from 'electron'
import { applyIsolatedElectronPathPlan, planIsolatedElectronPaths } from './host/electron-paths'
import { privateUpdateRelaunchProfile } from './host/update-profile'
import { configureAppIdentity } from './host/app-identity'

export { APP_NAME, PROFILE_APP_NAME } from './host/app-identity'
export const LEGACY_APP_NAMES = ['Vontra', 'vontra', 'Vontra Studio'] as const
export const HEADLESS_MODE = process.argv.includes('--headless-server') || process.env.VONTRA_HEADLESS === '1'

const updateProfile = privateUpdateRelaunchProfile(process.execPath, !process.env.VONTRA_HOME)
export const isolatedElectronPathPlan = planIsolatedElectronPaths(process.env.VONTRA_HOME || updateProfile?.root)
configureAppIdentity(app, Boolean(isolatedElectronPathPlan))
if (isolatedElectronPathPlan) {
  // Keep store.ts and every Electron-owned path on the same canonical root.
  process.env.VONTRA_HOME = isolatedElectronPathPlan.root
  applyIsolatedElectronPathPlan(isolatedElectronPathPlan, {
    setPath: (name, value) => app.setPath(name, value),
  })
  if (updateProfile?.root === isolatedElectronPathPlan.root && updateProfile.debugPort != null) {
    app.commandLine.appendSwitch('remote-debugging-address', '127.0.0.1')
    app.commandLine.appendSwitch('remote-debugging-port', String(updateProfile.debugPort))
  }
}
