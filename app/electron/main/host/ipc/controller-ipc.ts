import { buildControllerStatus, captureControllerSnapshot, requestControllerAccess } from '../../../agent/controller'
import { recordConnectorInteraction } from '../../../agent/connector-spine'
import type { ControllerSnapshot, ControllerStatus } from '../../../agent/types'
import { auditPreview } from '../../../agent/security'
import { saveSettings } from '../../../store'
import { store } from '../../state'
import { auditEvent, requireSecurityRole } from '../audit'
import { handle } from '../ipc'
import { persistSettings } from '../persistence'

export interface ControllerIpcDeps {
  appName: string
  rememberConnectorEvents(events: any[]): void
}

export function registerControllerIpc(deps: ControllerIpcDeps) {
  const { appName, rememberConnectorEvents } = deps

handle('controller:status', async (): Promise<ControllerStatus> => {
  return buildControllerStatus(store.settings.controller)
})

handle('controller:enable', async (): Promise<ControllerStatus> => {
  requireSecurityRole('operator', 'enable Controller capture')
  store.settings = { ...store.settings, controller: { ...store.settings.controller, enabled: true, lastError: null } }
  const status = requestControllerAccess(store.settings.controller)
  if (!status.canCapture && status.supported) store.settings.controller = { ...store.settings.controller, lastError: status.message }
  await auditEvent('controller.enable', { supported: status.supported, accessibilityTrusted: status.accessibilityTrusted })
  await saveSettings(store.settings)
  return status
})

handle('controller:capture', async (): Promise<ControllerSnapshot> => {
  requireSecurityRole('operator', 'capture Controller context')
  try {
    const snapshot = await captureControllerSnapshot(store.settings.controller, appName)
    store.settings = { ...store.settings, controller: { ...store.settings.controller, lastCapturedAt: snapshot.capturedAt, lastError: snapshot.warning } }
    await saveSettings(store.settings)
    await auditEvent('controller.capture', { appName: auditPreview(snapshot.appName), windowTitle: auditPreview(snapshot.windowTitle), textChars: snapshot.textCharCount, truncated: snapshot.truncated })
    rememberConnectorEvents([recordConnectorInteraction({
      connectorId: 'controller',
      pluginName: 'Controller',
      action: 'window_captured',
      direction: 'internal',
      target: snapshot.appName,
      output: { title: `Controller captured ${snapshot.appName}`, summary: snapshot.windowTitle, textChars: snapshot.textCharCount, threadKey: `controller:${snapshot.bundleId || snapshot.appName}` },
    })])
    return snapshot
  } catch (e: any) {
    const message = String(e?.message || e || 'Controller capture failed')
    store.settings = { ...store.settings, controller: { ...store.settings.controller, lastError: message } }
    persistSettings('controller.captureError')
    await auditEvent('controller.capture_failed', { error: auditPreview(message) })
    throw e
  }
})

}
