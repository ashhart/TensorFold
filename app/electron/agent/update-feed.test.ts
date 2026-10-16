// Run: jiti electron/agent/update-feed.test.ts
import assert from 'node:assert/strict'
import { VONTRA_UPDATE_FEED_URL, resolveUpdateFeedUrl } from './update-feed'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function main() {
  // Task #50: the feed is hardcoded to the canonical TensorFold Studio manifest.
  assert.equal(VONTRA_UPDATE_FEED_URL, 'https://tensorfold.dev/studio/update.json')
  assert.equal(resolveUpdateFeedUrl(), 'https://tensorfold.dev/studio/update.json')
  ok('resolveUpdateFeedUrl returns the canonical feed')

  // It is the only source of truth: no caller override is consulted. Even when
  // the historical override env vars are set, the hardcoded URL wins.
  const prevTensorFoldStudio = process.env.VONTRA_UPDATE_FEED_URL
  try {
    process.env.VONTRA_UPDATE_FEED_URL = 'https://evil.example.com/update.json'
    assert.equal(resolveUpdateFeedUrl(), 'https://tensorfold.dev/studio/update.json')
    ok('env var overrides are ignored')
  } finally {
    if (prevTensorFoldStudio === undefined) delete process.env.VONTRA_UPDATE_FEED_URL
    else process.env.VONTRA_UPDATE_FEED_URL = prevTensorFoldStudio
  }

  // The function takes no arguments and is pure: repeated calls are stable.
  assert.equal(resolveUpdateFeedUrl(), resolveUpdateFeedUrl())
  assert.equal((resolveUpdateFeedUrl as () => string).length, 0)
  ok('resolveUpdateFeedUrl is argument-free and stable')

  // Must be an https URL on the canonical host (defence-in-depth for #50).
  const url = new URL(resolveUpdateFeedUrl())
  assert.equal(url.protocol, 'https:')
  assert.equal(url.hostname, 'tensorfold.dev')
  ok('feed is https on tensorfold.dev')

  console.log(`update-feed.test.ts — ${passed} checks passed`)
}

main()
