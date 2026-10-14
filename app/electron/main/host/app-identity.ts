import path from 'node:path'

export const APP_NAME = 'TensorFold'
// Stable across the public rename: profile locks, Chromium state and cookie encryption.
export const PROFILE_APP_NAME = 'TensorFold Studio'
// package.json build.appId; kept from before the rename so installs update in place.
export const APP_BUNDLE_ID = 'dev.tensorfold.studio'

interface IdentityApp {
  setName(name: string): void
  getPath(name: 'appData'): string
  setPath(name: 'userData', value: string): void
  once(event: 'ready', listener: () => void): unknown
}

export function configureAppIdentity(app: IdentityApp, isolated: boolean): void {
  // Electron 42 captures its macOS Keychain service/account name before ready.
  // Keep that existing namespace; the ready callback sets the public runtime name.
  app.setName(PROFILE_APP_NAME)
  if (!isolated) app.setPath('userData', path.join(app.getPath('appData'), PROFILE_APP_NAME))
  app.once('ready', () => app.setName(APP_NAME))
}
