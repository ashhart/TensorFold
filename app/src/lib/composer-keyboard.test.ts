import assert from 'node:assert/strict'
import { isComposerSendKey } from './composer-keyboard'
const key = { key: 'Enter', nativeEvent: { isComposing: false } }
assert.equal(isComposerSendKey(key), true)
assert.equal(isComposerSendKey(key, 'modified-enter'), false)
assert.equal(isComposerSendKey({ ...key, metaKey: true }, 'modified-enter'), true)
assert.equal(isComposerSendKey({ ...key, ctrlKey: true }, 'modified-enter'), true)
assert.equal(isComposerSendKey({ ...key, metaKey: true }), false)
for (const preference of ['enter', 'modified-enter'] as const) {
  for (const patch of [{ shiftKey: true, metaKey: true }, { altKey: true, ctrlKey: true }, { repeat: true, metaKey: true }, { nativeEvent: { isComposing: true }, metaKey: true }, { nativeEvent: { isComposing: false, keyCode: 229 }, ctrlKey: true }, { keyCode: 229, metaKey: true }, { key: 'v', metaKey: true }, { key: 'z', ctrlKey: true }, { key: 'a', metaKey: true }]) assert.equal(isComposerSendKey({ ...key, ...patch }, preference), false)
}
console.log('Send-key preference preserves newline, IME confirmation, repeated Enter and native paste/undo/select-all shortcuts')
