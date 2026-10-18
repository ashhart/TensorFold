import assert from 'node:assert/strict'
import { textEditContextMenu, applicationEditMenu } from './native-text-editing'

const editable = { isEditable: true, selectionText: 'selected', editFlags: { canDelete: true, canEditRichly: false, canUndo: true, canRedo: false, canCut: true, canCopy: true, canPaste: true, canSelectAll: true } }
const menu = textEditContextMenu(editable)
assert.deepEqual(menu.filter((item) => item.role).map((item) => item.role), ['undo', 'redo', 'cut', 'copy', 'paste', 'pasteAndMatchStyle', 'selectAll'])
assert.equal(menu.find((item) => item.role === 'redo')?.enabled, false)
assert.equal(menu.find((item) => item.role === 'pasteAndMatchStyle')?.label, 'Paste as Plain Text')
assert.deepEqual(textEditContextMenu({ ...editable, isEditable: false }).map((item) => item.role), ['copy'])
assert.deepEqual(textEditContextMenu({ ...editable, isEditable: false, selectionText: '' }), [])
assert.equal(applicationEditMenu().find((item) => item.role === 'pasteAndMatchStyle')?.accelerator, 'CmdOrCtrl+Shift+V')
