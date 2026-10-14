// Run: npx tsx electron/agent/controller-actions.test.ts
import assert from 'node:assert/strict'
import {
  CONTROLLER_ACTIONS,
  controllerActionScript,
  formatControllerActionResult,
  normalizeControllerActionInput,
} from './controller-actions'

assert.deepEqual(CONTROLLER_ACTIONS, ['activate_app', 'keystroke', 'menu_item'])

const activate = normalizeControllerActionInput({ action: 'activate_app', appName: 'Terminal', note: 'bring logs forward' }, { ownAppName: 'TensorFold Studio' })
assert.equal(activate.action, 'activate_app')
assert.equal(activate.appName, 'Terminal')
assert.equal(activate.note, 'bring logs forward')

const key = normalizeControllerActionInput({ action: 'keystroke', appName: 'Terminal', keystroke: 'l', modifiers: ['command', 'shift'] }, { ownAppName: 'TensorFold Studio' })
assert.deepEqual(key.modifiers, ['command', 'shift'])
assert.match(controllerActionScript(key), /keystroke "l" using \{command down, shift down\}/)

const menu = normalizeControllerActionInput({ action: 'menu_item', appName: 'Safari', menuPath: ['File', 'New Tab'] }, { ownAppName: 'TensorFold Studio' })
assert.deepEqual(menu.menuPath, ['File', 'New Tab'])
assert.match(controllerActionScript(menu), /click menu item "New Tab" of menu "File" of menu bar 1/)
assert.match(controllerActionScript(menu), /tell application "Safari" to activate/)

assert.throws(() => normalizeControllerActionInput({ action: 'keystroke', appName: 'Terminal' }, { ownAppName: 'TensorFold Studio' }), /keystroke requires/)
assert.throws(() => normalizeControllerActionInput({ action: 'menu_item', appName: 'Safari', menuPath: ['File'] }, { ownAppName: 'TensorFold Studio' }), /menuPath/)
assert.throws(() => normalizeControllerActionInput({ action: 'activate_app', appName: 'TensorFold Studio' }, { ownAppName: 'TensorFold Studio' }), /cannot target TensorFold Studio/)
for (const currentName of ['TensorFold', 'TensorFold Studio']) {
  for (const targetName of ['TensorFold', 'TensorFold Studio']) {
    assert.throws(() => normalizeControllerActionInput({ action: 'activate_app', appName: targetName }, { ownAppName: currentName }), /cannot target/)
  }
}

const output = formatControllerActionResult({
  ok: true,
  action: 'menu_item',
  appName: 'Safari',
  target: 'File > New Tab',
  executedAt: Date.UTC(2026, 5, 20, 12, 0, 0),
  note: 'Opened a tab.',
})
assert.match(output, /Action: menu_item/)
assert.match(output, /App: Safari/)
assert.match(output, /Target: File > New Tab/)
assert.match(output, /Approval: controller_action tool approval/)

console.log('controller action tests passed')
