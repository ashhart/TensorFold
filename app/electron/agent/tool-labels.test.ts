// Run: npx jiti electron/agent/tool-labels.test.ts
import assert from 'node:assert/strict'
import { toolLabel } from './tool-labels'

// A well-formed controller_action still produces a useful label.
const good = toolLabel('controller_action', { action: 'keystroke', appName: 'Terminal', keystroke: 'l', modifiers: ['command'] })
assert.equal(good.verb, 'Controller keystroke')
assert.equal(good.target, 'Terminal Command+l')

// Regression: malformed/partial controller_action args must NOT throw.
// normalizeControllerActionInput throws on these; toolLabel must stay total so
// the throw never escapes the per-call loop and aborts the whole agent turn.
assert.doesNotThrow(() => toolLabel('controller_action', { action: 'click' }), 'invalid action must not throw')
assert.doesNotThrow(() => toolLabel('controller_action', { appName: '' }), 'empty appName must not throw')
assert.doesNotThrow(() => toolLabel('controller_action', {}), 'empty args must not throw')
assert.doesNotThrow(() => toolLabel('controller_action', { action: 'activate_app', appName: 'TensorFold Studio' }), 'own-app target must not throw')

// And the fallback still carries whatever raw context is available.
const invalidAction = toolLabel('controller_action', { action: 'click', appName: 'Notes' })
assert.equal(invalidAction.verb, 'Controller click')
assert.equal(invalidAction.target, 'Notes')

const emptyApp = toolLabel('controller_action', { action: '', appName: '' })
assert.equal(emptyApp.verb, 'Controller action')
assert.equal(emptyApp.target, '')

const ownApp = toolLabel('controller_action', { action: 'activate_app', appName: 'TensorFold Studio' })
assert.equal(ownApp.verb, 'Controller activate_app')
assert.equal(ownApp.target, 'TensorFold Studio')

console.log('tool-labels.test.ts: all assertions passed')
