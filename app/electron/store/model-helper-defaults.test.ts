import assert from 'node:assert/strict'
import { normalizeStoredHelperDefaults } from './model-helper-defaults'
assert.deepEqual(normalizeStoredHelperDefaults({ naming: { providerId: 'local', accountId: null, model: 'fixture', baseUrl: 'https://invalid.example.test' } }), {})
assert.deepEqual(normalizeStoredHelperDefaults({ search: { providerId: 'local', accountId: null, model: 'fixture' }, unknown: 'ignored' }), { search: { providerId: 'local', accountId: null, model: 'fixture' } })
assert.deepEqual(normalizeStoredHelperDefaults({ document_extraction: { providerId: 'local', accountId: 'bad\nline', model: 'fixture' } }), {})
console.log('Helper defaults accept configured route identities only, excluding raw endpoints, credentials and control characters')
