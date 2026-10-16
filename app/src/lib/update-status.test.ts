import assert from 'node:assert/strict'
import { updateEmptyMessage, updateStatusLabel } from './update-status'
import type { AppUpdateStatus } from './types'

const status = { state: 'not_checked', currentVersion: '1.0.0', checkedAt: null, signatureVerified: false, error: null } as AppUpdateStatus
assert.match(updateStatusLabel(status), /not been checked/)
assert.match(updateEmptyMessage(status), /No update check/)
assert.match(updateStatusLabel(null), /not been checked/)
assert.match(updateStatusLabel({ ...status, state: 'current' }), /not been checked/, 'Legacy current state without proof is still unchecked')
assert.match(updateEmptyMessage({ ...status, state: 'current' }), /No update check/)
const checked = { ...status, state: 'current' as const, checkedAt: 1000, signatureVerified: true }
assert.equal(updateStatusLabel(checked), 'TensorFold is current')
assert.match(updateEmptyMessage(checked), /latest version/)
assert.equal(updateEmptyMessage({ ...checked, state: 'error', error: 'Update feed returned HTTP 404' }), 'Update feed returned HTTP 404')
assert.match(updateStatusLabel({ ...checked, state: 'downloading' }), /Downloading and verifying/)
assert.match(updateStatusLabel({ ...checked, state: 'ready', latestVersion: '1.0.1' }), /ready to install/)
assert.match(updateStatusLabel({ ...checked, state: 'installing' }), /Preparing update and restart/)
assert.match(updateEmptyMessage({ ...checked, state: 'ready' }), /finish active work/)
console.log('Update status requires a completed verified check before claiming current')
