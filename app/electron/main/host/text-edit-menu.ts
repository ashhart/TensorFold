import { nativeUiText, NATIVE_EDIT_LABELS, knownNativeLabel } from './native-ui-text'
import type { BrowserWindow, ContextMenuParams, Menu, MenuItem, MenuItemConstructorOptions, WebContents, WebFrameMain } from 'electron'

type TextContext = Pick<ContextMenuParams, 'isEditable' | 'selectionText'> & {
  editFlags: Pick<ContextMenuParams['editFlags'], 'canUndo' | 'canRedo' | 'canCut' | 'canCopy' | 'canPaste' | 'canSelectAll'>
}
type TrustedTextSender = (frame: WebFrameMain | null, sender: WebContents) => boolean
// Constructor roles use camelCase, while Electron's native MenuItem getter can return
// lowercase names. Compare the role identity independently of that representation.
const normalizedRole = (role: string | undefined): string => role?.toLowerCase() ?? ''
function copyMenuItemOptions(item: MenuItem): MenuItemConstructorOptions {
  // Keep custom presentation and handlers on the one container/role being replaced.
  // Other MenuItem instances are reused verbatim by buildFromTemplate.
  const options: Record<string, unknown> = {}
  for (const key of ['id', 'role', 'type', 'label', 'accessibilityLabel', 'sublabel', 'toolTip', 'icon',
    'enabled', 'visible', 'checked', 'accelerator', 'acceleratorWorksWhenHidden', 'registerAccelerator',
    'sharingItem', 'badge', 'click', 'submenu'] as const) {
    const value = (item as unknown as Record<string, unknown>)[key]
    if (value !== undefined) options[key] = value
  }
  return options as MenuItemConstructorOptions
}

// Electron roles execute inside Chromium's focused editor instead of replacing its value.
// This preserves the user's selection, native clipboard semantics, and undo transaction.
export function applicationEditMenu(locale?: unknown): MenuItemConstructorOptions[] {
  return [
    { role: 'undo' }, { role: 'redo' }, { type: 'separator' },
    { role: 'cut' }, { role: 'copy' }, { role: 'paste' },
    { role: 'pasteAndMatchStyle', label: nativeUiText(locale, 'Paste as Plain Text'), accelerator: 'CmdOrCtrl+Shift+V' },
    { type: 'separator' }, { role: 'selectAll' },
  ].map((item) => {
    const source = NATIVE_EDIT_LABELS[normalizedRole(item.role)]
    return source ? { ...item, label: nativeUiText(locale, source) } : item
  }) as MenuItemConstructorOptions[]
}

export function textEditContextMenu(params: TextContext, locale?: unknown): MenuItemConstructorOptions[] {
  if (!params.isEditable) return params.selectionText ? [{ role: 'copy', label: nativeUiText(locale, 'Copy'), enabled: params.editFlags.canCopy }] : []
  const allowed = {
    undo: params.editFlags.canUndo, redo: params.editFlags.canRedo,
    cut: params.editFlags.canCut, copy: params.editFlags.canCopy,
    paste: params.editFlags.canPaste, pasteAndMatchStyle: params.editFlags.canPaste,
    selectAll: params.editFlags.canSelectAll,
  }
  return applicationEditMenu(locale).map((item) => item.role && item.role in allowed
    ? { ...item, enabled: allowed[item.role as keyof typeof allowed] } : item)
}

export function installTextEditingMenu(win: BrowserWindow, menu: typeof Menu, trustedSender: TrustedTextSender, locale: () => unknown = () => undefined): void {
  const contents = win.webContents
  contents.on('context-menu', (_event, params) => {
    // The original frame can be null after navigation/destruction; the trust callback also
    // requires identity with the registered window's main frame and exact entry document.
    if (!trustedSender(params.frame, contents)) return
    const items = textEditContextMenu(params, locale())
    if (items.length) menu.buildFromTemplate(items).popup({ window: win })
  })
}

/** Preserve custom item objects while rebuilding immutable native menu structure. */
export function ensureApplicationEditMenu(menu: typeof Menu, createItem: (options: MenuItemConstructorOptions) => MenuItem, locale?: unknown): void {
  const current = menu.getApplicationMenu()
  if (!current) {
    menu.setApplicationMenu(menu.buildFromTemplate([
      ...(process.platform === 'darwin' ? [{ role: 'appMenu' as const }] : []),
      { id: 'tensorfold-edit-menu', label: nativeUiText(locale, 'Edit'), submenu: applicationEditMenu(locale) }, { role: 'windowMenu' },
    ]))
    return
  }
  const edit = current.items.find((item) => item.id === 'tensorfold-edit-menu' || normalizedRole(item.role) === 'editmenu' || item.label.replace(/&/g, '') === 'Edit')
  // getApplicationMenu() is installed native structure: never append/insert into it,
  // or assign MenuItem accelerator/submenu/role (read-only in Electron's runtime).
  const topLevel = [...current.items]
  if (!edit?.submenu) {
    const before = topLevel.findIndex((item) => normalizedRole(item.role) === 'windowmenu' || normalizedRole(item.role) === 'help')
    const added = createItem({ id: 'tensorfold-edit-menu', label: nativeUiText(locale, 'Edit'), submenu: applicationEditMenu(locale) })
    topLevel.splice(before < 0 ? topLevel.length : before, 0, added)
    menu.setApplicationMenu(menu.buildFromTemplate(topLevel))
    return
  }
  let changed = false
  const items = edit.submenu.items.map((item) => {
    const role = normalizedRole(item.role), source = NATIVE_EDIT_LABELS[role]
    if (source && role !== 'pasteandmatchstyle' && knownNativeLabel(source, item.label) && item.label !== nativeUiText(locale, source)) {
      changed = true
      return createItem({ ...copyMenuItemOptions(item), label: nativeUiText(locale, source) })
    }
    if (normalizedRole(item.role) !== 'pasteandmatchstyle' || (item.label === nativeUiText(locale, 'Paste as Plain Text') && item.accelerator === 'CmdOrCtrl+Shift+V')) return item
    changed = true
    return createItem({
      ...menuItemPresentation(item), role: 'pasteAndMatchStyle', label: nativeUiText(locale, 'Paste as Plain Text'), accelerator: 'CmdOrCtrl+Shift+V',
    })
  })
  const present = new Set(items.map((item) => normalizedRole(item.role)))
  const missing = applicationEditMenu(locale).filter((item) => item.role && !present.has(normalizedRole(item.role)))
  if (missing.length) {
    changed = true
    if (items.length) items.push(createItem({ type: 'separator' }))
    items.push(...missing.map(createItem))
  }
  const editLabel = edit.id === 'tensorfold-edit-menu' || knownNativeLabel('Edit', edit.label) ? nativeUiText(locale, 'Edit') : edit.label
  if (edit.label !== editLabel) changed = true
  if (!changed) return
  // buildFromTemplate explicitly accepts existing MenuItem instances. Reusing those
  // keeps every unrelated custom action/callback and nested submenu intact.
  const replacement = createItem({ ...menuItemPresentation(edit), role: edit.role, label: editLabel, submenu: menu.buildFromTemplate(items) })
  menu.setApplicationMenu(menu.buildFromTemplate(topLevel.map((item) => item === edit ? replacement : item)))
}

function menuItemPresentation(item: MenuItem): MenuItemConstructorOptions {
  return {
    id: item.id, enabled: item.enabled, visible: item.visible,
    sublabel: item.sublabel, toolTip: item.toolTip, icon: item.icon,
    registerAccelerator: item.registerAccelerator,
  }
}
