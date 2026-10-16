// Run: jiti electron/agent/terminal-multiplex.test.ts
//
// Terminal multiplex + flow control (VST-039): ack buffer (reconnect replay bounds + ack
// backpressure), ANSI-safe scrollback capping, layout presets + persistence. Pure.
import assert from 'node:assert/strict'
import {
  TerminalOutputBuffer,
  capScrollbackAnsiSafe,
  initLayout,
  applyPreset,
  splitPane,
  assignTerminal,
  setActivePane,
  suspendPane,
  presetSlots,
  serializeLayout,
  restoreLayout,
} from './terminal-multiplex'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('terminal-multiplex tests:')

// --- ack buffer: reconnect replay bounds ---
{
  const buf = new TerminalOutputBuffer({ maxBytes: 1024 })
  buf.append('one')
  buf.append('two')
  buf.append('three')
  assert.equal(buf.head, 3)

  // A client that has seen seq 1 reconnects → replays 2,3, no gap.
  const replay = buf.since(1)
  assert.deepEqual(replay.chunks.map((c) => c.data), ['two', 'three'])
  assert.equal(replay.gap, false)
  assert.equal(replay.head, 3)

  // A fresh client (cursor 0) gets everything.
  assert.deepEqual(buf.since(0).chunks.map((c) => c.data), ['one', 'two', 'three'])
  // Caught-up client gets nothing.
  assert.deepEqual(buf.since(3).chunks, [])
  ok('buffer.since: replays after the cursor, reports head')
}

// --- ack frees the buffer; replay before the retained window = gap ---
{
  const buf = new TerminalOutputBuffer({ maxBytes: 1024 })
  for (const d of ['a', 'b', 'c', 'd']) buf.append(d)
  buf.ack(2) // consumer confirmed through seq 2 → 1,2 dropped
  assert.deepEqual(buf.since(2).chunks.map((c) => c.data), ['c', 'd'], 'still replays un-acked tail')
  // A client that reconnects claiming an OLD cursor (0) now hits a gap — 1,2 are gone.
  const stale = buf.since(0)
  assert.equal(stale.gap, true, 'a cursor below the retained window signals a gap (re-baseline)')
  assert.deepEqual(stale.chunks.map((c) => c.data), ['c', 'd'], 'returns only what survives')
  buf.ack(1) // stale/lower ack is a no-op
  assert.deepEqual(buf.since(2).chunks.map((c) => c.data), ['c', 'd'])
  ok('buffer.ack: frees acked chunks; stale replay → gap; lower ack is a no-op')
}

// --- ack backpressure: byte cap evicts an unacking client's backlog ---
{
  const buf = new TerminalOutputBuffer({ maxBytes: 10 })
  buf.append('12345')       // 5 bytes
  buf.append('67890')       // 10 total
  assert.equal(buf.backpressured(8), true, 'unacked backlog past the threshold trips backpressure')
  buf.append('ABCDE')       // 15 > cap → oldest evicted
  assert.ok(buf.retainedBytes <= 10, 'byte cap holds even when the consumer never acks')
  const replay = buf.since(0)
  assert.equal(replay.gap, true, 'the evicted prefix is reported as a gap, never silently dropped')
  assert.equal(buf.backpressured(100), false, 'a high threshold is not tripped')
  ok('buffer: byte cap bounds an unacking client; eviction surfaces as a gap')
}

// --- ANSI-safe scrollback capping ---
{
  const plain = 'abcdefghij' // 10 chars
  assert.equal(capScrollbackAnsiSafe(plain, 100), plain, 'under cap → unchanged')
  assert.equal(capScrollbackAnsiSafe(plain, 4), 'ghij', 'trims from the front to the byte cap')

  // A color escape straddling the cut point must not be split. "\x1b[31mRED\x1b[0m" —
  // cutting mid-escape would corrupt rendering; the cut advances past the escape.
  const colored = 'xxxxx\x1b[31mRED\x1b[0m'
  const capped = capScrollbackAnsiSafe(colored, 10)
  assert.ok(!capped.includes('\x1b[31m') || capped.startsWith('\x1b[31m'), 'never retains a half of the opening escape')
  // The cut lands inside "\x1b[31m" (which spans indices 5-9 for a 5-char escape); result
  // must start at a clean boundary, not mid-escape.
  assert.ok(!/^\d?m/.test(capped), 'result does not start with escape-sequence tail chars')
  assert.ok(capped.endsWith('\x1b[0m'), 'the tail content is preserved intact')
  ok('capScrollbackAnsiSafe: front-trims to cap without splitting an ANSI escape')
}

// --- layout presets ---
{
  let n = 0
  const id = () => `p${++n}`
  let layout = initLayout(id)
  assert.equal(layout.preset, 'single')
  assert.equal(layout.panes.length, 1)

  layout = applyPreset(layout, 'four', id)
  assert.equal(layout.panes.length, 4, 'four preset → 4 panes')
  assert.equal(presetSlots('four'), 4)
  const firstId = layout.panes[0].id

  // Bind a terminal, shrink to two-columns → the first pane's binding survives, extras drop.
  layout = assignTerminal(layout, firstId, 'term-1', 'build')
  layout = applyPreset(layout, 'two-columns', id)
  assert.equal(layout.panes.length, 2)
  assert.equal(layout.panes[0].terminalId, 'term-1', 'existing binding preserved across preset change')
  assert.equal(layout.panes[0].label, 'build')

  // Active pane reassigns if the active one was dropped.
  layout = setActivePane(layout, layout.panes[1].id)
  layout = applyPreset(layout, 'single', id)
  assert.equal(layout.panes.length, 1)
  assert.equal(layout.activePaneId, layout.panes[0].id, 'active reassigns when its pane is dropped')
  ok('layout presets: grow/shrink preserves bindings, reassigns active')
}

// --- split + suspend ---
{
  let n = 0
  const id = () => `p${++n}`
  let layout = initLayout(id)
  layout = splitPane(layout, id)
  assert.ok(layout.panes.length >= 2, 'split grows the pane set')
  assert.equal(layout.activePaneId, layout.panes[layout.panes.length - 1].id, 'the new pane becomes active')
  layout = suspendPane(layout, layout.activePaneId, true)
  assert.equal(layout.panes.find((p) => p.id === layout.activePaneId)?.suspended, true)
  ok('split: grows + focuses new pane; suspend flags it')
}

// --- layout persistence ---
{
  let n = 0
  const id = () => `p${++n}`
  let layout = applyPreset(initLayout(id), 'two-columns', id)
  layout = assignTerminal(layout, layout.panes[0].id, 'term-live', 'left')
  layout = assignTerminal(layout, layout.panes[1].id, 'term-dead', 'right')
  const raw = serializeLayout(layout)

  // Restore with only term-live still running → the dead binding is cleared, slot kept.
  const restored = restoreLayout(raw, ['term-live'], id)
  assert.equal(restored.preset, 'two-columns')
  assert.equal(restored.panes.length, 2, 'preset slot count preserved')
  assert.equal(restored.panes[0].terminalId, 'term-live', 'a live terminal binding survives a reload')
  assert.equal(restored.panes[1].terminalId, null, 'a dead terminal binding is cleared, not shown as bound')
  assert.equal(restored.panes[0].label, 'left', 'labels persist')

  // Corrupt JSON → a safe default layout, never a throw.
  const fallback = restoreLayout('not json', [], id)
  assert.equal(fallback.preset, 'single')
  ok('layout persistence: serialize/restore, dead bindings cleared, corrupt → safe default')
}

console.log(`\n${passed} passed`)
