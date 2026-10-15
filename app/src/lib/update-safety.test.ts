import assert from 'node:assert/strict'
import { registerUpdatePreparationBlocker, rendererUpdateBlockReason } from './update-safety'

let dirty = true
const remove = registerUpdatePreparationBlocker(() => dirty ? 'Save or discard the provider draft before restarting.' : null)
assert.match(rendererUpdateBlockReason() ?? '', /provider draft/)
dirty = false
assert.equal(rendererUpdateBlockReason(), null)
dirty = true
remove()
assert.equal(rendererUpdateBlockReason(), null, 'unmounted draft owners remove their restart blocker')
console.log('Unsaved renderer drafts block installation until saved or discarded')
