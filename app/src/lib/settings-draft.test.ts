import assert from 'node:assert/strict'
import { createSettingsDraftPersistence } from './settings-draft'

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: Error) => void
  const promise = new Promise<T>((ok, fail) => { resolve = ok; reject = fail })
  return { promise, resolve, reject }
}

const dirty = { current: false }
const first = deferred<{ value: number }>()
let draft = { value: 0 }
const persistence = createSettingsDraftPersistence({ dirty, setDraft: (value) => { draft = value }, save: () => first.promise })
const saving = persistence.persist({ value: 1 })
assert.equal(dirty.current, true, 'a dispatched but unconfirmed Settings save must block restart')
first.reject(new Error('Synthetic persistence failure'))
await assert.rejects(saving, /Synthetic persistence failure/)
assert.equal(dirty.current, true, 'a rejected Settings save must retain the unsaved draft guard')
assert.deepEqual(draft, { value: 1 }, 'a rejected Settings save must retain the user edit')
persistence.discard({ value: 0 })
assert.equal(dirty.current, false, 'an explicit discard may release the draft guard')
assert.deepEqual(draft, { value: 0 })

const old = deferred<{ value: number }>()
const latest = deferred<{ value: number }>()
const calls: number[] = []
const serial = createSettingsDraftPersistence<{ value: number }>({ dirty, setDraft: (value) => { draft = value }, save: (value) => {
  calls.push(value.value)
  return value.value === 1 ? old.promise : latest.promise
} })
const a = serial.persist({ value: 1 })
const b = serial.persist({ value: 2 })
await Promise.resolve()
assert.deepEqual(calls, [1], 'overlapping Settings saves must reach persistence in user order')
old.resolve({ value: 1 })
await a
assert.equal(dirty.current, true, 'an old response cannot release a newer edit or save')
assert.deepEqual(draft, { value: 2 }, 'an old response cannot replace the newer draft')
latest.resolve({ value: 2 })
await b
assert.deepEqual(calls, [1, 2])
assert.equal(dirty.current, false)

const held = deferred<{ value: number }>()
const editing = createSettingsDraftPersistence({ dirty, setDraft: (value) => { draft = value }, save: () => held.promise })
const prior = editing.persist({ value: 3 })
editing.edited()
draft = { value: 4 }
held.resolve({ value: 3 })
await prior
assert.equal(dirty.current, true, 'editing while a save is pending must keep the draft protected')
assert.deepEqual(draft, { value: 4 }, 'save confirmation must not erase subsequent typing')
console.log('Settings draft failure, ordering, discard and typing regressions pass')
