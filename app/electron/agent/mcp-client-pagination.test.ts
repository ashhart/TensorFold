// Run: jiti electron/agent/mcp-client-pagination.test.ts
//
// Covers the *_list pagination guard added to mcp-client.ts: a buggy/malicious MCP
// server that returns a non-advancing or never-terminating nextCursor must not wedge
// connect() in an unbounded request loop. assertListPageBudget is the pure guard called
// before each page request.
import assert from 'node:assert/strict'
import { assertListPageBudget } from './mcp-client.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

function main() {
  console.log('mcp-client pagination guard tests:')

  // 1) A normal, advancing sequence of cursors is allowed.
  {
    const seen = new Set<string>()
    assert.doesNotThrow(() => assertListPageBudget('S', 'tools/list', 1, undefined, seen))
    assert.doesNotThrow(() => assertListPageBudget('S', 'tools/list', 2, 'a', seen))
    assert.doesNotThrow(() => assertListPageBudget('S', 'tools/list', 3, 'b', seen))
    assert.doesNotThrow(() => assertListPageBudget('S', 'tools/list', 4, 'c', seen))
    ok('allows a normal advancing cursor sequence')
  }

  // 2) A repeated (non-advancing) cursor is rejected — this is the infinite-loop case.
  {
    const seen = new Set<string>()
    assertListPageBudget('S', 'tools/list', 1, undefined, seen)
    assertListPageBudget('S', 'tools/list', 2, 'same', seen)
    assert.throws(
      () => assertListPageBudget('S', 'tools/list', 3, 'same', seen),
      /repeated pagination cursor/,
      'a server returning the same cursor twice must abort listing',
    )
    ok('rejects a repeated (non-advancing) cursor')
  }

  // 3) Exceeding the page cap is rejected even if every cursor is unique (never-terminating).
  {
    const seen = new Set<string>()
    // maxPages=3: pages 1..3 ok, page 4 over budget.
    assert.doesNotThrow(() => assertListPageBudget('S', 'tools/list', 1, undefined, seen, 3))
    assert.doesNotThrow(() => assertListPageBudget('S', 'tools/list', 2, 'p1', seen, 3))
    assert.doesNotThrow(() => assertListPageBudget('S', 'tools/list', 3, 'p2', seen, 3))
    assert.throws(
      () => assertListPageBudget('S', 'tools/list', 4, 'p3', seen, 3),
      /exceeded 3 pages/,
      'a server emitting endless unique cursors must abort at the page cap',
    )
    ok('rejects pagination past the page cap')
  }

  // 4) The first page (cursor undefined) is never treated as a repeat, even called twice.
  {
    const seen = new Set<string>()
    assert.doesNotThrow(() => assertListPageBudget('S', 'r', 1, undefined, seen))
    // undefined is not added to seen, so a (hypothetical) second undefined wouldn't false-trip.
    assert.doesNotThrow(() => assertListPageBudget('S', 'r', 2, undefined, seen))
    ok('never flags the undefined first-page cursor as repeated')
  }

  console.log(`\n${passed} checks passed.`)
}

main()
