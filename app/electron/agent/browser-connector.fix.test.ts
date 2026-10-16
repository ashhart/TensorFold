// Run: jiti electron/agent/browser-connector.fix.test.ts
// The bug: normalizeTime only special-cased ms-vs-s by a 1e13 threshold. Chrome/WebKit history
// stores last_visit_time as MICROSECONDS since 1601-01-01 (~1.3e16); that is >1e13 so it was
// divided by 1000 once and floored, yielding a far-future timestamp. It must be converted via
// (value/1000 - 11644473600000).
import assert from 'node:assert/strict'
import { normalizeTime } from './browser-connector.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('browser-connector fix tests:')

const WEBKIT_OFFSET_MS = 11_644_473_600_000

{
  // Build a WebKit microsecond value for a known instant (2024-01-01T00:00:00Z) and round-trip it.
  const unixMs = Date.UTC(2024, 0, 1)
  const webkitMicros = (unixMs + WEBKIT_OFFSET_MS) * 1000
  const got = normalizeTime(webkitMicros, null)
  assert.equal(got, unixMs)
  ok('WebKit/Chrome microseconds-since-1601 convert to the correct Unix ms')
}

{
  // The result must be a sane present-day timestamp, NOT a far-future one.
  const webkitNow = (Date.now() + WEBKIT_OFFSET_MS) * 1000
  const got = normalizeTime(webkitNow, null)!
  const year = new Date(got).getUTCFullYear()
  assert.ok(year >= 2020 && year <= 2100, `expected a present-era year, got ${year}`)
  ok('a current WebKit timestamp normalizes to a present-era year, not far future')
}

{
  // Unix milliseconds (now ~1.7e12) pass through unchanged.
  const ms = 1_700_000_000_000
  assert.equal(normalizeTime(ms, null), ms)
  ok('Unix milliseconds pass through unchanged')
}

{
  // Unix microseconds (~1.7e15, below the WebKit threshold) are divided by 1000 to ms.
  const micros = 1_700_000_000_000_000
  assert.equal(normalizeTime(micros, null), 1_700_000_000_000)
  ok('Unix microseconds are divided to milliseconds')
}

{
  // Unix seconds (~1.7e9) pass through as-is (existing behavior).
  assert.equal(normalizeTime(1_700_000_000, null), 1_700_000_000)
  ok('Unix seconds pass through as-is')
}

{
  // Non-numeric / empty falls back.
  assert.equal(normalizeTime('', 42), 42)
  assert.equal(normalizeTime(null, 7), 7)
  ok('empty/null fall back to the provided fallback')
}

console.log(`\n${passed} checks passed.`)
