import assert from 'node:assert/strict'
import React from 'react'
import { createJiti } from 'jiti'
const { Composer } = await createJiti(import.meta.url, { jsx: { runtime: 'automatic' } }).import('./Composer.tsx')

// Exercise the actual textarea handler, rather than a copied keyboard predicate.
function input(overrides = {}) {
  let sent = 0, started = 0, prevented = 0
  const tree = Composer({ taskText: 'Draft a reply', setTaskText: () => {}, canDirectReply: true,
    selectedItem: { from: 'Example contact', title: 'Example message' }, selectedSource: 'all',
    quickActions: [], taskBusy: false, sendingReply: false,
    onSendReply: () => { sent++ }, onStartTask: () => { started++ }, onDraftTask: () => {}, ...overrides })
  function find(node) {
    for (const child of React.Children.toArray(node)) {
      if (!React.isValidElement(child)) continue
      if (child.type === 'textarea') return child
      const nested = find(child.props.children)
      if (nested) return nested
    }
  }
  const area = find(tree)
  return { key: (extra = {}) => area.props.onKeyDown({ key: 'Enter', shiftKey: false, nativeEvent: { isComposing: false }, preventDefault: () => { prevented++ }, ...extra }), result: () => ({ sent, started, prevented }) }
}

const composing = input()
composing.key({ nativeEvent: { isComposing: true } })
assert.deepEqual(composing.result(), { sent: 0, started: 0, prevented: 0 }, 'IME confirmation must not send a message')
const safariComposition = input()
safariComposition.key({ keyCode: 229 })
assert.equal(safariComposition.result().sent, 0)
const sending = input({ sendingReply: true })
sending.key()
assert.equal(sending.result().sent, 0, 'keyboard follows the disabled send control')
const working = input({ taskBusy: true, canDirectReply: false })
working.key()
assert.equal(working.result().started, 0, 'keyboard cannot launch a duplicate assistant task')
const blank = input({ taskText: '   ' })
blank.key()
assert.equal(blank.result().sent, 0)
const newline = input()
newline.key({ shiftKey: true })
assert.equal(newline.result().prevented, 0)
const reply = input()
reply.key()
assert.deepEqual(reply.result(), { sent: 1, started: 0, prevented: 1 })
const task = input({ canDirectReply: false })
task.key()
assert.equal(task.result().started, 1)
console.log('assistant composer keyboard safety: ok')
