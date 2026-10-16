import assert from 'node:assert/strict'
import { clearPersistentBrowserData } from './browser-data-purge'

const calls: string[] = []
await clearPersistentBrowserData({
  flushStorageData() { calls.push('flush') },
  async clearStorageData() { calls.push('storage') },
  async clearCache() { calls.push('cache') },
  async clearAuthCache() { calls.push('auth') },
  async clearCodeCaches() { calls.push('code') },
  async clearHostResolverCache() { calls.push('dns') },
})

assert.equal(calls[0], 'flush', 'pending browser storage is flushed before destructive clearing')
assert.equal(calls.at(-1), 'flush', 'partition deletion is flushed after clearing')
assert.deepEqual(new Set(calls.slice(1, -1)), new Set(['storage', 'cache', 'auth', 'code', 'dns']), 'storage, cache, auth, code, and DNS state are all cleared')

console.log('browser automation persistent-data purge tests passed')
