import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  browserSourceId,
  browserSourceRef,
  makeBrowserCurrentTabSource,
  makeBrowserHistorySource,
  readBrowserItem,
  scanBrowserItems,
  summarizeBrowser,
  syncBrowserConnector,
} from './browser-connector.ts'
import type { BrowserConnectorSettings } from './types'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

async function main() {
console.log('browser connector tests:')

{
  const source = makeBrowserCurrentTabSource({
    url: 'https://user:pass@example.com/launch?view=plan#section',
    title: 'Launch Plan',
    excerpt: 'Current tab has auth edge cases and release notes.',
    capturedAt: 1_781_978_400_000,
  }, 1_781_978_400_000)
  assert.equal(source.kind, 'current_tab')
  assert.equal(source.url, 'https://example.com/launch?view=plan#section')
  assert.equal(source.title, 'Launch Plan')
  assert.equal(source.excerpt, 'Current tab has auth edge cases and release notes.')
  assert.throws(() => makeBrowserCurrentTabSource({ url: 'javascript:alert(1)', title: 'Bad' }), /http or https/)
  ok('current-tab sources are explicit and sanitize credentials')
}

{
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-browser-'))
  try {
    const historyPath = path.join(root, 'history.json')
    await fs.writeFile(historyPath, JSON.stringify([
      {
        url: 'https://example.com/launch',
        title: 'Launch notes',
        visitedAt: '2026-06-20T10:00:00Z',
        text: 'Release notes mention auth edge cases and QA evidence.',
      },
      {
        url: 'file:///Users/example/secret',
        title: 'Skipped local file',
        visitedAt: '2026-06-20T11:00:00Z',
        text: 'This must not be indexed.',
      },
      {
        url: 'https://user:pass@example.com/sanitized',
        title: 'Sanitized history',
        time: 1_781_982_000_000,
        excerpt: 'Credentials should be stripped from this explicit export.',
      },
    ]), 'utf8')
    const current = makeBrowserCurrentTabSource({
      url: 'https://docs.example.com/current',
      title: 'Current Tab',
      excerpt: 'Auth launch checklist in the foreground tab.',
      capturedAt: 1_781_979_000_000,
    }, 1_781_979_000_000)
    const history = makeBrowserHistorySource(historyPath, 1_781_979_500_000)
    const settings: BrowserConnectorSettings = {
      sources: [current, history],
      maxItems: 20,
      lastSyncAt: null,
    }

    const { items } = await scanBrowserItems(settings, { query: 'auth', limit: 10 })
    assert.equal(items.length, 2)
    assert.ok(items.every((item) => browserSourceId(item).startsWith('browser:')))
    assert.ok(items.every((item) => !item.url.includes('user:pass')))
    assert.ok(items.some((item) => item.kind === 'current_tab' && item.title === 'Current Tab'))
    assert.ok(items.some((item) => item.kind === 'history_export' && item.title === 'Launch notes'))

    const read = await readBrowserItem(settings, items[0].id)
    assert.match(read.content, /URL: https:\/\//)
    assert.match(read.content, /Sources:/)
    assert.equal(browserSourceRef(read).connectorId, 'browser')

    const summary = await summarizeBrowser(settings, { query: 'auth', limit: 10 })
    assert.match(summary.summary, /Sources:/)
    assert.match(summary.summary, /browser:/)
    assert.equal(summary.sources.length, 2)

    const synced = await syncBrowserConnector(settings, [])
    assert.equal(synced.events.length, 3)
    assert.ok(synced.events.every((event) => event.connectorId === 'browser'))
    assert.ok(synced.events.every((event) => event.action === 'browser_source_seen'))
    ok('browser history exports list, read, summarize, and sync with source ids')

    // Regression: every built event must carry a stable payload.messageId so
    // rememberConnectorEvents' insert-time dedup (which only checks payload.messageId) can
    // catch duplicates from two concurrent syncs that both snapshot existingEvents before a
    // slow scan completes. Simulate that race: two syncs of the same items, neither aware of
    // the other's in-flight events, then dedup by messageId the way the host does.
    for (const event of synced.events) {
      assert.equal(typeof event.payload?.messageId, 'string')
      assert.ok((event.payload!.messageId as string).length > 0)
    }
    const concurrentSyncA = await syncBrowserConnector(settings, [])
    const concurrentSyncB = await syncBrowserConnector(settings, [])
    const messageIdsA = concurrentSyncA.events.map((e) => e.payload?.messageId)
    const messageIdsB = concurrentSyncB.events.map((e) => e.payload?.messageId)
    assert.deepEqual(messageIdsA.sort(), messageIdsB.sort())
    const seenIds = new Set<string>()
    const afterInsertDedup = [...concurrentSyncA.events, ...concurrentSyncB.events].filter((event) => {
      const mid = typeof event.payload?.messageId === 'string' ? event.payload.messageId : null
      if (!mid) return true
      if (seenIds.has(mid)) return false
      seenIds.add(mid)
      return true
    })
    assert.equal(afterInsertDedup.length, concurrentSyncA.events.length)
    ok('concurrent browser syncs stamp matching messageId and insert-time dedup catches duplicates')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

{
  // Regression (#26): itemsFromHistoryExport pre-sliced to maxItems*3 BEFORE filtering out
  // invalid (no url/href) entries. With maxItems=10 and the first 50 of 100 records lacking a
  // url/href, only the first 30 raw records (all invalid) were ever examined, returning 0
  // items despite 50 valid records existing later in the export.
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-browser-slice-'))
  try {
    const historyPath = path.join(root, 'history.json')
    const invalid = Array.from({ length: 50 }, (_v, i) => ({ title: `Invalid ${i}`, text: 'no url or href here' }))
    const valid = Array.from({ length: 50 }, (_v, i) => ({
      url: `https://example.com/page-${i}`,
      title: `Valid ${i}`,
      visitedAt: `2026-06-2${i % 9}T10:00:00Z`,
    }))
    await fs.writeFile(historyPath, JSON.stringify([...invalid, ...valid]), 'utf8')
    const history = makeBrowserHistorySource(historyPath, 1_781_979_500_000)
    const settings: BrowserConnectorSettings = {
      sources: [history],
      maxItems: 10,
      lastSyncAt: null,
    }
    const { items } = await scanBrowserItems(settings, { limit: 10 })
    assert.equal(items.length, 10)
    assert.ok(items.every((item) => item.url.startsWith('https://example.com/page-')))
    ok('history export filters invalid entries before slicing to maxItems, not after')
  } finally {
    await fs.rm(root, { recursive: true, force: true })
  }
}

console.log(`\n${passed} checks passed.`)
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
