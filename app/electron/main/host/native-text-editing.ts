import type { BrowserWindow, Menu } from 'electron'
import { validateSender } from './ipc'
import { installTextEditingMenu } from './text-edit-menu'
export { applicationEditMenu, textEditContextMenu, ensureApplicationEditMenu } from './text-edit-menu'
export function installNativeTextEditing(win: BrowserWindow, menu: typeof Menu, locale: () => unknown = () => undefined): void {
  installTextEditingMenu(win, menu, validateSender, locale)
}
