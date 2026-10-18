import { nativeUiText } from './native-ui-text'
import assert from 'node:assert/strict'
import type { MenuItemConstructorOptions } from 'electron'
import { ensureApplicationEditMenu, installNativeTextEditing } from './native-text-editing'
import { applicationEditMenu, textEditContextMenu } from './text-edit-menu'
import { configureTrustedRenderer, clearTrustedRenderer } from './ipc'

class FakeMenu {
  items: FakeItem[] = []
  append() { throw new Error('Installed menu structure must not be appended to') }
  insert() { throw new Error('Installed menu structure must not be inserted into') }
}
class FakeItem {
  role?: string
  id?: string
  label: string
  submenu?: FakeMenu
  options: MenuItemConstructorOptions
  accelerator?: string
  constructor(options: MenuItemConstructorOptions) {
    this.options = options
    this.id = options.id
    this.role = options.role
    this.label = options.label ?? ''
    this.accelerator = options.accelerator
    if (Array.isArray(options.submenu)) {
      this.submenu = new FakeMenu()
      this.submenu.items = options.submenu.map((item) => new FakeItem(item))
    } else if (options.submenu) this.submenu = options.submenu as unknown as FakeMenu
    // Match Electron42's read-only descriptors, not the looser public typings.
    for (const key of ['role', 'accelerator', 'submenu']) Object.defineProperty(this, key, { value: this[key as keyof this], writable: false, configurable: true, enumerable: true })
  }
}
function menuPort(initial: FakeMenu | null) {
  let current = initial
  return {
    getApplicationMenu: () => current,
    setApplicationMenu: (next: FakeMenu) => { current = next },
    buildFromTemplate: (items: Array<MenuItemConstructorOptions | FakeItem>) => {
      const result = new FakeMenu()
      result.items = items.map((item) => item instanceof FakeItem ? item : new FakeItem(item))
      return result
    },
  }
}
const expectedRoles = ['undo', 'redo', 'cut', 'copy', 'paste', 'pasteAndMatchStyle', 'selectAll']
const role = (value?: string) => value?.toLowerCase()
const editIn = (menu: FakeMenu) => menu.items.find((item) => item.id === 'tensorfold-edit-menu' || role(item.role) === 'editmenu' || item.label === 'Edit')!
function assertRoles(edit: FakeItem) { for (const expected of expectedRoles) assert.equal(edit.submenu!.items.filter((item) => role(item.role) === role(expected)).length, 1, `native role ${expected} occurs exactly once`) }
const app = new FakeMenu()
const copy = new FakeItem({ role: 'copy' })
const custom = new FakeItem({ label: 'Custom action', click: () => {} })
const edit = new FakeItem({ role: 'editMenu', label: 'Edit', submenu: [] })
edit.submenu!.items = [copy, custom]
const help = new FakeItem({ role: 'help', label: 'Help' })
app.items = [edit, help]
const port = menuPort(app)
const createItem = (options: MenuItemConstructorOptions) => new FakeItem(options) as any
ensureApplicationEditMenu(port as any, createItem)
let rebuilt = port.getApplicationMenu()!
let repairedEdit = editIn(rebuilt)
assert.notEqual(rebuilt, app, 'installed native structure is rebuilt through the supported template API')
assert.equal(app.items[0], edit); assert.equal(edit.submenu!.items.length, 2, 'old native menu remains untouched')
assert.equal(rebuilt.items[1], help)
assert.equal(repairedEdit.submenu!.items[0], copy)
assert.equal(repairedEdit.submenu!.items[1], custom)
assert.equal(repairedEdit.submenu!.items[1].options.click, custom.options.click)
assertRoles(repairedEdit)
ensureApplicationEditMenu(port as any, createItem)
assert.equal(port.getApplicationMenu(), rebuilt, 'complete native menu is idempotent without rebuilding')
const oldPlain = repairedEdit.submenu!.items.find((item) => role(item.role) === 'pasteandmatchstyle')!
oldPlain.label = 'Paste and Match Style'
Object.defineProperty(oldPlain, 'accelerator', { value: 'CmdOrCtrl+Alt+Shift+V', writable: false, configurable: true })
ensureApplicationEditMenu(port as any, createItem)
rebuilt = port.getApplicationMenu()!; repairedEdit = editIn(rebuilt)
const newPlain = repairedEdit.submenu!.items.find((item) => role(item.role) === 'pasteandmatchstyle')!
assert.notEqual(newPlain, oldPlain, 'immutable accelerator is replaced by a new MenuItem')
assert.equal(newPlain.label, 'Paste as Plain Text'); assert.equal(newPlain.accelerator, 'CmdOrCtrl+Shift+V')
assert.equal(oldPlain.accelerator, 'CmdOrCtrl+Alt+Shift+V', 'old native descriptor is never assigned')
assert.equal(repairedEdit.submenu!.items[0], copy); assert.equal(repairedEdit.submenu!.items[1], custom)
assertRoles(repairedEdit)
// Runtime role getters are lowercase and localized labels need not be “Edit”.
const lowerApp = new FakeMenu()
const lowerEdit = new FakeItem({ role: 'editMenu', label: 'Localized menu', submenu: applicationEditMenu() })
Object.defineProperty(lowerEdit, 'role', { value: 'editmenu', writable: false, configurable: true })
for (const item of lowerEdit.submenu!.items) Object.defineProperty(item, 'role', { value: item.role?.toLowerCase(), writable: false, configurable: true })
const lowerPlain = lowerEdit.submenu!.items.find((item) => item.role === 'pasteandmatchstyle')!
Object.defineProperty(lowerPlain, 'accelerator', { value: 'CmdOrCtrl+Alt+Shift+V', writable: false, configurable: true })
const lowerWindow = new FakeItem({ role: 'windowMenu' })
Object.defineProperty(lowerWindow, 'role', { value: 'windowmenu', writable: false, configurable: true })
lowerApp.items = [lowerEdit, lowerWindow, help]
const lowerPort = menuPort(lowerApp)
ensureApplicationEditMenu(lowerPort as any, createItem)
assert.equal(lowerPort.getApplicationMenu()!.items.length, 3)
const lowerRebuilt = editIn(lowerPort.getApplicationMenu()!)
assert.equal(lowerRebuilt.label, 'Localized menu'); assertRoles(lowerRebuilt)
assert.equal(lowerRebuilt.submenu!.items.find((item) => role(item.role) === 'pasteandmatchstyle')!.accelerator, 'CmdOrCtrl+Shift+V')
const noEdit = new FakeMenu()
const file = new FakeItem({ label: 'File', submenu: [{ label: 'Export' }] })
noEdit.items = [file, lowerWindow, help]
const noEditPort = menuPort(noEdit)
ensureApplicationEditMenu(noEditPort as any, createItem)
assert.deepEqual(noEditPort.getApplicationMenu()!.items.filter((item) => item.label !== 'Edit'), [file, lowerWindow, help])
assert.equal(noEditPort.getApplicationMenu()!.items[1].label, 'Edit'); assertRoles(editIn(noEditPort.getApplicationMenu()!))
assert.equal(noEdit.items.length, 3, 'missing Edit insertion never mutates the installed menu')
const fresh = menuPort(null)
ensureApplicationEditMenu(fresh as any, createItem)
const freshEdit = editIn(fresh.getApplicationMenu()!)
assertRoles(freshEdit)
const plain = freshEdit.submenu!.items.find((item) => item.role === 'pasteAndMatchStyle')!
assert.equal(plain.label, 'Paste as Plain Text'); assert.equal(plain.accelerator, 'CmdOrCtrl+Shift+V')
assert.ok(applicationEditMenu().every((item) => !item.click), 'native editor roles never replace text from custom menu callbacks')

// Changing the selected language rebuilds immutable items and keeps one owned Edit container.
const languagePort = menuPort(null)
for (const locale of ['en', 'zh-Hans', 'ja', 'ru', 'zh-Hant', 'en']) {
  ensureApplicationEditMenu(languagePort as any, createItem, locale)
  const current = languagePort.getApplicationMenu()!, localized = editIn(current)
  assert.equal(current.items.filter(item => item.id === 'tensorfold-edit-menu').length, 1)
  assert.equal(localized.label, nativeUiText(locale, 'Edit'))
  assertRoles(localized)
  assert.equal(localized.submenu!.items.find(item => item.role === 'copy')?.label, nativeUiText(locale, 'Copy'))
  const paste = localized.submenu!.items.find(item => item.role === 'pasteAndMatchStyle')!
  assert.equal(paste.label, nativeUiText(locale, 'Paste as Plain Text')); assert.equal(paste.accelerator, 'CmdOrCtrl+Shift+V')
  const same = languagePort.getApplicationMenu(); ensureApplicationEditMenu(languagePort as any, createItem, locale)
  assert.equal(languagePort.getApplicationMenu(), same, 'Language-stable settings saves do not duplicate or rebuild installed menus')
}

// Every renderer edit flag is applied separately, including plain-text paste.
const flags = ['canUndo', 'canRedo', 'canCut', 'canCopy', 'canPaste', 'canSelectAll'] as const
for (let mask = 0; mask < 1 << flags.length; mask++) {
  const editFlags = Object.fromEntries(flags.map((flag, index) => [flag, !!(mask & 1 << index)])) as any
  const context = textEditContextMenu({ isEditable: true, selectionText: 'selected text', editFlags })
  assert.deepEqual(context.filter((item) => item.role).map((item) => item.role), expectedRoles)
  for (const item of context) {
    const flag = item.role === 'pasteAndMatchStyle' ? 'canPaste'
      : item.role ? `can${item.role[0].toUpperCase()}${item.role.slice(1)}` : undefined
    if (flag) assert.equal(item.enabled, editFlags[flag], `native ${item.role} must respect ${flag}`)
  }
  const readOnly = textEditContextMenu({ isEditable: false, selectionText: 'selected text', editFlags })
  assert.deepEqual(readOnly, [{ role: 'copy', label: 'Copy', enabled: editFlags.canCopy }])
  assert.deepEqual(textEditContextMenu({ isEditable: false, selectionText: '', editFlags }), [])
}

// Exercise the real installed wrapper and global trust boundary, not a matching-URL stub.
const entry = 'file:///private/qa/TensorFold.app/Contents/Resources/app.asar/dist/index.html'
const mainFrame = { url: entry }
let handler: ((event: unknown, params: any) => void) | undefined
const contents = { mainFrame, on: (_name: string, next: typeof handler) => { handler = next } }
const win = { webContents: contents }
const popups: MenuItemConstructorOptions[][] = []
const popupPort = { buildFromTemplate: (items: MenuItemConstructorOptions[]) => ({ popup: () => { popups.push(items) } }) }
const context = {
  frame: mainFrame,
  isEditable: true,
  selectionText: 'selected text',
  editFlags: { canUndo: true, canRedo: false, canCut: true, canCopy: true, canPaste: true, canSelectAll: true },
}
configureTrustedRenderer(contents as any, entry)
try {
  installNativeTextEditing(win as any, popupPort as any)
  handler!({}, context)
  assert.equal(popups.length, 1)
  const destroyedFrame = { get url(): string { throw new Error('Object has been destroyed') } }
  for (const frame of [null, { url: entry }, { url: 'file:///private/other.html' }, destroyedFrame]) handler!({}, { ...context, frame })
  assert.equal(popups.length, 1, 'null, same-URL child frames, and foreign frames cannot open native editing menus')
  mainFrame.url = `${entry}?unexpected=1`
  handler!({}, context)
  assert.equal(popups.length, 1, 'a navigated main frame is no longer trusted')
  mainFrame.url = `${entry}#chat`
  handler!({}, context)
  assert.equal(popups.length, 2, 'same-document navigation remains valid')
  const replacement = { mainFrame: { url: entry } }
  configureTrustedRenderer(replacement as any, entry)
  handler!({}, context)
  assert.equal(popups.length, 2, 'a replaced window cannot open editing menus')
  clearTrustedRenderer(replacement as any)
} finally { clearTrustedRenderer(contents as any) }

console.log('native text editing menu flags, partial-menu repair, and exact main-frame trust passed')
