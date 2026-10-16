import { dialog } from 'electron'
import { HEADLESS_MODE } from '../early-electron-paths'
import { store } from '../state'

export interface ConfirmSensitiveActionInput {
  title: string
  message: string
  detail: string
  confirmLabel: string
}

/**
 * Confirm-before-destructive-action dialog. Fails closed (returns false) in headless mode
 * or when no usable window exists, so remote/automated callers can never bypass the prompt.
 *
 * Extracted verbatim from main-app.ts; takes `store` and `HEADLESS_MODE` as direct deps
 * (both are singletons read at call time, so no factory is needed).
 */
export async function confirmSensitiveAction({ title, message, detail, confirmLabel }: ConfirmSensitiveActionInput): Promise<boolean> {
  if (HEADLESS_MODE || !store.win || store.win.isDestroyed()) return false
  const result = await dialog.showMessageBox(store.win, {
    type: 'warning',
    title,
    message,
    detail,
    buttons: [confirmLabel, 'Cancel'],
    defaultId: 1,
    cancelId: 1,
    noLink: true,
  })
  return result.response === 0
}
