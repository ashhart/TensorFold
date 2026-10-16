// Run: jiti electron/agent/shared-room-memory.test.ts
import assert from 'node:assert/strict'
import {
  CANONICAL_BLOCK_NAMES,
  clearBlock,
  cloneSharedMemory,
  createSharedMemory,
  diffSharedMemory,
  listBlocks,
  MEMORY_MAX_BLOCK_CHARS,
  MEMORY_MAX_BLOCKS,
  readBlock,
  renderSharedMemory,
  writeBlock,
} from './shared-room-memory.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('shared-room-memory tests:')

// --- construction + basic write/read ---------------------------------------
{
  const m = createSharedMemory()
  assert.equal(listBlocks(m).length, 0)
  const r = writeBlock(m, { name: 'plan', content: 'ship it', author: 'alice', now: 100 })
  assert.ok(r.ok && r.changed)
  const b = readBlock(m, 'plan')
  assert.ok(b)
  assert.equal(b!.content, 'ship it')
  assert.equal(b!.updatedBy, 'alice')
  assert.equal(b!.revision, 1)
  assert.equal(b!.updatedAt, 100)
  ok('write then read a block with attribution + revision')
}

// --- last-writer-wins (replace default) ------------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'decisions', content: 'use postgres', author: 'a', now: 1 })
  const r = writeBlock(m, { name: 'decisions', content: 'use sqlite', author: 'b', now: 2 })
  assert.ok(r.ok)
  assert.equal(readBlock(m, 'decisions')!.content, 'use sqlite')
  assert.equal(readBlock(m, 'decisions')!.updatedBy, 'b')
  assert.equal(readBlock(m, 'decisions')!.revision, 2)
  ok('replace mode is last-writer-wins')
}

// --- name normalization (case + whitespace) --------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: '  Blockers ', content: 'x', author: 'a' })
  writeBlock(m, { name: 'blockers', content: 'y', author: 'b' })
  assert.equal(Object.keys(m.blocks).length, 1)
  assert.equal(readBlock(m, 'BLOCKERS')!.content, 'y')
  ok('block names normalize case + whitespace to one slot')
}

// --- empty name rejected ----------------------------------------------------
{
  const m = createSharedMemory()
  const r = writeBlock(m, { name: '   ', content: 'x', author: 'a' })
  assert.ok(!r.ok && r.reason === 'empty-name')
  ok('empty block name rejected')
}

// --- no-op write does not bump revision ------------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'plan', content: 'same', author: 'a', now: 1 })
  const r = writeBlock(m, { name: 'plan', content: 'same', author: 'b', now: 2 })
  assert.ok(r.ok && !r.changed)
  assert.equal(readBlock(m, 'plan')!.revision, 1)
  assert.equal(readBlock(m, 'plan')!.updatedBy, 'a') // unchanged: original author kept
  assert.equal(readBlock(m, 'plan')!.updatedAt, 1)
  ok('identical write is a no-op: revision + attribution preserved')
}

// --- append mode with de-dup ------------------------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'log', content: 'first', author: 'a', mode: 'append' })
  writeBlock(m, { name: 'log', content: 'second', author: 'b', mode: 'append' })
  const dup = writeBlock(m, { name: 'log', content: 'first', author: 'c', mode: 'append' })
  assert.equal(readBlock(m, 'log')!.content, 'first\nsecond')
  assert.ok(dup.ok && !dup.changed)
  ok('append mode concatenates and de-dups identical lines')
}

// --- append-bullet merge, case-insensitive de-dup, order stable ------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'blockers', content: 'db is down', author: 'a', mode: 'append-bullet' })
  writeBlock(m, { name: 'blockers', content: '- auth flaky\n- DB is down', author: 'b', mode: 'append-bullet' })
  assert.equal(readBlock(m, 'blockers')!.content, '- db is down\n- auth flaky')
  ok('append-bullet merges lists, case-insensitive de-dup, order preserved')
}

// --- optimistic concurrency: expectedRevision guard ------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'plan', content: 'v1', author: 'a' }) // rev 1
  const stale = writeBlock(m, { name: 'plan', content: 'v2-from-stale', author: 'b', expectedRevision: 0 })
  assert.ok(!stale.ok && stale.reason === 'conflict')
  assert.equal((stale as any).currentRevision, 1)
  assert.equal(readBlock(m, 'plan')!.content, 'v1') // not clobbered
  const fresh = writeBlock(m, { name: 'plan', content: 'v2', author: 'b', expectedRevision: 1 })
  assert.ok(fresh.ok)
  assert.equal(readBlock(m, 'plan')!.content, 'v2')
  ok('expectedRevision blocks stale writes, allows fresh ones')
}

// --- clearBlock -------------------------------------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'blockers', content: 'x', author: 'a' }) // rev 1
  assert.equal(clearBlock(m, 'blockers', 'b', 9), true)
  assert.equal(readBlock(m, 'blockers')!.content, '')
  assert.equal(readBlock(m, 'blockers')!.revision, 2)
  assert.equal(clearBlock(m, 'blockers', 'b'), false) // already empty
  assert.equal(clearBlock(m, 'nope', 'b'), false) // missing
  ok('clearBlock empties content + bumps revision, no-op when empty/missing')
}

// --- listBlocks ordering: canonical first, then alpha, empties omitted -----
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'zebra', content: 'z', author: 'a' })
  writeBlock(m, { name: 'plan', content: 'p', author: 'a' })
  writeBlock(m, { name: 'blockers', content: 'b', author: 'a' })
  writeBlock(m, { name: 'decisions', content: 'd', author: 'a' })
  writeBlock(m, { name: 'apple', content: 'ap', author: 'a' })
  writeBlock(m, { name: 'empty', content: '', author: 'a' })
  const names = listBlocks(m).map((b) => b.name)
  assert.deepEqual(names, ['decisions', 'plan', 'blockers', 'apple', 'zebra'])
  ok('listBlocks: canonical order first, free-form alpha, empties omitted')
}

// --- render for the seat prompt --------------------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'decisions', content: 'go with sqlite', author: 'alice', now: 1 })
  writeBlock(m, { name: 'blockers', content: '- none', author: 'bob', now: 2 })
  const out = renderSharedMemory(m)
  assert.ok(out.includes('SHARED MEMORY'))
  assert.ok(out.includes('## DECISIONS (last: alice, rev 1)'))
  assert.ok(out.includes('go with sqlite'))
  assert.ok(out.includes('## BLOCKERS (last: bob, rev 1)'))
  // decisions renders before blockers (canonical order)
  assert.ok(out.indexOf('DECISIONS') < out.indexOf('BLOCKERS'))
  ok('renderSharedMemory produces ordered, attributed prompt text')
}

// --- render empty -> '' -----------------------------------------------------
{
  assert.equal(renderSharedMemory(createSharedMemory()), '')
  const m = createSharedMemory()
  writeBlock(m, { name: 'plan', content: '', author: 'a' })
  assert.equal(renderSharedMemory(m), '')
  ok('renderSharedMemory returns empty string when nothing to show')
}

// --- render truncation ------------------------------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'plan', content: 'word '.repeat(500), author: 'a' })
  const out = renderSharedMemory(m, { maxBlockChars: 50 })
  assert.ok(out.includes('…(truncated)'))
  ok('renderSharedMemory truncates oversized blocks')
}

// --- render options: no attribution + custom heading -----------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'plan', content: 'x', author: 'a' })
  const out = renderSharedMemory(m, { attribution: false, heading: 'BOARD' })
  assert.ok(out.includes('BOARD'))
  assert.ok(!out.includes('(last:'))
  ok('render honors attribution:false + custom heading')
}

// --- seed + clone independence ---------------------------------------------
{
  const m = createSharedMemory({ plan: 'seeded', blockers: null as any })
  assert.equal(readBlock(m, 'plan')!.content, 'seeded')
  assert.equal(readBlock(m, 'plan')!.updatedBy, 'seed')
  assert.equal(readBlock(m, 'blockers'), null) // null seed skipped
  const snap = cloneSharedMemory(m)
  writeBlock(m, { name: 'plan', content: 'mutated', author: 'z' })
  assert.equal(snap.blocks['plan'].content, 'seeded') // clone not aliased
  ok('createSharedMemory seed + cloneSharedMemory is a deep snapshot')
}

// --- diffSharedMemory -------------------------------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'plan', content: 'v1', author: 'a' })
  const before = cloneSharedMemory(m)
  writeBlock(m, { name: 'plan', content: 'v2', author: 'b' }) // rev 1 -> 2
  writeBlock(m, { name: 'decisions', content: 'new', author: 'c' }) // added
  const d = diffSharedMemory(before, m)
  assert.deepEqual(d.changed, [{ name: 'plan', from: 1, to: 2 }])
  assert.deepEqual(d.added, ['decisions'])
  ok('diffSharedMemory reports changed revisions + added blocks')
}

// --- canonical constant sanity ---------------------------------------------
{
  assert.deepEqual([...CANONICAL_BLOCK_NAMES], ['decisions', 'plan', 'blockers'])
  ok('CANONICAL_BLOCK_NAMES exported as expected')
}

// --- size caps: oversized content and block-count limits are rejected -------
{
  const m = createSharedMemory()
  const big = writeBlock(m, { name: 'plan', content: 'x'.repeat(MEMORY_MAX_BLOCK_CHARS + 1), author: 'a' })
  assert.ok(!big.ok && big.reason === 'content-too-large', 'oversized content rejected')
  assert.equal(readBlock(m, 'plan'), null, 'rejected write leaves no block behind')
  const atCap = writeBlock(m, { name: 'plan', content: 'x'.repeat(MEMORY_MAX_BLOCK_CHARS), author: 'a' })
  assert.ok(atCap.ok, 'content exactly at the cap is accepted')

  const appendOverflow = writeBlock(m, { name: 'plan', content: 'y'.repeat(MEMORY_MAX_BLOCK_CHARS), author: 'b', mode: 'append' })
  assert.ok(!appendOverflow.ok && appendOverflow.reason === 'content-too-large', 'append that would exceed the cap is rejected')

  const full = createSharedMemory()
  for (let i = 0; i < MEMORY_MAX_BLOCKS; i++) writeBlock(full, { name: `block-${i}`, content: 'v', author: 'a' })
  const overflow = writeBlock(full, { name: 'one-more', content: 'v', author: 'a' })
  assert.ok(!overflow.ok && overflow.reason === 'block-limit', 'new block past the block limit rejected')
  const existing = writeBlock(full, { name: 'block-0', content: 'v2', author: 'a' })
  assert.ok(existing.ok, 'updating an existing block still works at the limit')
  ok('writeBlock enforces content size and block count caps')
}

// --- untrusted framing in the prompt render ---------------------------------
{
  const m = createSharedMemory()
  writeBlock(m, { name: 'decisions', content: 'use sqlite', author: 'alice' })
  const out = renderSharedMemory(m)
  assert.ok(out.includes('seat-authored'), 'default render frames shared memory as seat-authored')
  assert.ok(out.includes('not authoritative'), 'default render disclaims authoritative room state')
  ok('renderSharedMemory frames blocks as seat-authored, untrusted content')
}

console.log(`\n${passed} shared-room-memory tests passed`)
