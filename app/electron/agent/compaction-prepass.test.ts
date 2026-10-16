// Run: npx jiti electron/agent/compaction-prepass.test.ts
// Proves the cheap pre-pass (dedup + old-large-tool-result summarization + stale
// image stripping) shrinks the to-be-folded slice BEFORE the expensive LLM
// summarization call, without corrupting message shape or order.
import assert from 'node:assert/strict'
import type { ApiMessage, ContentBlock } from './types'
import { prePassCompact } from './compaction-prepass.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

const tool = (content: string, name = 'read_file', id = 'c1'): ApiMessage => ({ role: 'tool', tool_call_id: id, name, content })
const user = (t: string): ApiMessage => ({ role: 'user', content: t })
const asstCall = (id: string, name = 'read_file'): ApiMessage => ({ role: 'assistant', content: null, tool_calls: [{ id, type: 'function', function: { name, arguments: '{}' } }] })

console.log('compaction-prepass tests:')

// 1) Dedup identical tool-result messages: later byte-identical duplicates collapse
// to a short placeholder; the first occurrence is kept verbatim.
{
  const dup = 'same file contents over and over '.repeat(5)
  const msgs: ApiMessage[] = [user('go'), asstCall('c1'), tool(dup, 'read_file', 'c1'), asstCall('c2'), tool(dup, 'read_file', 'c2'), asstCall('c3'), tool(dup, 'read_file', 'c3')]
  const { messages: out, savedChars } = prePassCompact(msgs)
  assert.equal(out.length, msgs.length, 'message count unchanged (placeholders, not removals)')
  assert.equal(out[2].content, dup, 'first occurrence kept verbatim')
  assert.notEqual(out[4].content, dup, 'second duplicate replaced')
  assert.notEqual(out[6].content, dup, 'third duplicate replaced')
  assert.match(String(out[4].content), /duplicate tool result omitted/i, 'placeholder names itself')
  assert.match(String(out[6].content), /duplicate tool result omitted/i, 'placeholder names itself')
  assert.ok(savedChars > 0, 'reports saved chars from dedup')
  ok('dedupes byte-identical tool results, keeping the first occurrence')
}

// 2) Distinct tool-result content is never treated as a duplicate.
{
  const msgs: ApiMessage[] = [asstCall('c1'), tool('alpha contents', 'read_file', 'c1'), asstCall('c2'), tool('beta contents', 'read_file', 'c2')]
  const { messages: out } = prePassCompact(msgs)
  assert.equal(out[1].content, 'alpha contents')
  assert.equal(out[3].content, 'beta contents')
  ok('leaves distinct tool results untouched')
}

// 3) Old large tool-result messages get summarized into a one-liner; small ones don't.
{
  const big = 'x'.repeat(3000)
  const small = 'small output'
  const msgs: ApiMessage[] = [
    asstCall('c1'), tool(big, 'search', 'c1'),
    asstCall('c2'), tool(small, 'search', 'c2'),
    // pad so the big/small tool results above are NOT within the most-recent window
    user('a'), user('b'), user('c'), user('d'), user('e'), user('f'), user('g'), user('h'),
  ]
  const { messages: out, savedChars } = prePassCompact(msgs)
  const bigOut = String(out[1].content)
  assert.ok(bigOut.length < big.length, 'large old tool result shrunk')
  assert.match(bigOut, /truncated.*originally 3000 chars/i, 'one-liner says how big it originally was')
  assert.equal(out[3].content, small, 'small tool result left alone (under threshold)')
  assert.ok(savedChars >= big.length - bigOut.length, 'savedChars accounts for the shrink')
  ok('summarizes old large tool results into a short one-liner')
}

// 4) run_command-shaped output: the one-liner tries to surface "exit N" when present.
{
  const out1 = 'a '.repeat(1200) + '\nExited with code 1'
  const msgs: ApiMessage[] = [
    asstCall('c1', 'run_command'), tool(out1, 'run_command', 'c1'),
    user('a'), user('b'), user('c'), user('d'), user('e'), user('f'), user('g'), user('h'),
  ]
  const { messages: out } = prePassCompact(msgs)
  assert.match(String(out[1].content), /exit(ed)? (with )?code 1|exit 1/i, 'surfaces the exit code')
  ok('extracts an exit-code signal for run_command-shaped tool output')
}

// 5) Recent messages within the slice are left completely untouched by SUMMARIZATION,
// even if large — only OLD ones (outside the most-recent window) get summarized.
{
  const big = 'y'.repeat(5000)
  const msgs: ApiMessage[] = [
    user('pad1'), user('pad2'), user('pad3'), user('pad4'), user('pad5'), user('pad6'),
    asstCall('cRecent'), tool(big, 'search', 'cRecent'), // near the tail -> recent
  ]
  const { messages: out } = prePassCompact(msgs)
  assert.equal(out[7].content, big, 'large tool result inside the recent window is untouched')
  ok('does not summarize large content inside the recent window')
}

// 5b) Dedup is recency-agnostic by design: an exact duplicate of an already-seen tool
// result carries zero extra information no matter where it sits, including inside the
// recent window — so it still collapses to the placeholder there.
{
  const dup = 'shared content '.repeat(3)
  const msgs: ApiMessage[] = [
    asstCall('c1'), tool(dup, 'x', 'c1'),
    user('pad'),
    asstCall('c2'), tool(dup, 'x', 'c2'), // duplicate, but inside the last-3 recent window
  ]
  const { messages: out } = prePassCompact(msgs)
  assert.equal(out[1].content, dup, 'first occurrence kept verbatim')
  assert.match(String(out[4].content), /duplicate tool result omitted/i, 'duplicate inside the recent window is still deduped')
  ok('dedup applies even to duplicates that fall inside the recent window')
}

// 6) savedChars accurately sums (original length - new length) across ALL changed messages.
{
  const dup = 'z'.repeat(50)
  const big = 'w'.repeat(2500)
  const msgs: ApiMessage[] = [
    asstCall('c1'), tool(dup, 'read_file', 'c1'),
    asstCall('c2'), tool(dup, 'read_file', 'c2'), // dup of c1
    asstCall('c3'), tool(big, 'search', 'c3'),
    user('a'), user('b'), user('c'), user('d'), user('e'), user('f'),
  ]
  const before = msgs.reduce((n, m) => n + (typeof m.content === 'string' ? m.content.length : 0), 0)
  const { messages: out, savedChars } = prePassCompact(msgs)
  const after = out.reduce((n, m) => n + (typeof m.content === 'string' ? m.content.length : 0), 0)
  assert.equal(savedChars, before - after, 'savedChars matches the literal char delta')
  assert.ok(savedChars > 0, 'nonzero given a dup + an old large result')
  ok('savedChars is an accurate sum of (original - new) length across changed messages')
}

// 7) Never produces an empty or malformed (length-mismatched) message array, including
// on edge-case inputs: empty list, single message, all-recent, all-small.
{
  {
    const { messages: out, savedChars } = prePassCompact([])
    assert.deepEqual(out, [])
    assert.equal(savedChars, 0)
  }
  {
    const msgs = [user('hello')]
    const { messages: out } = prePassCompact(msgs)
    assert.equal(out.length, 1)
    assert.equal(out[0].content, 'hello')
  }
  {
    const msgs: ApiMessage[] = [asstCall('c1'), tool('tiny', 'read_file', 'c1')]
    const { messages: out } = prePassCompact(msgs)
    assert.equal(out.length, msgs.length)
  }
  ok('never produces an empty/malformed array on empty, single, or all-small/recent input')
}

// 8) Output is a new array — input is never mutated.
{
  const dup = 'q'.repeat(40)
  const msgs: ApiMessage[] = [asstCall('c1'), tool(dup, 'read_file', 'c1'), asstCall('c2'), tool(dup, 'read_file', 'c2')]
  const snapshot = msgs.map((m) => ({ ...m }))
  prePassCompact(msgs)
  assert.deepEqual(msgs, snapshot, 'input array/messages left untouched')
  ok('does not mutate the input array')
}

// 9) Non-tool messages (user/assistant/system) are never altered by dedup or summarization,
// even if two user turns happen to share identical text.
{
  const msgs: ApiMessage[] = [user('same text'), asstCall('c1'), tool('r1', 'read_file', 'c1'), user('same text'), user('pad'), user('pad2'), user('pad3'), user('pad4')]
  const { messages: out } = prePassCompact(msgs)
  assert.equal(out[0].content, 'same text')
  assert.equal(out[3].content, 'same text')
  ok('only role:tool messages participate in dedup/summarization')
}

// 10) Stale image content-parts (outside the recent window) on a tool message are
// stripped to a short text placeholder, leaving sibling text blocks intact; images
// inside the recent window are left alone.
{
  // Realistic-sized fake payload (real image data: URLs run tens of KB) so the
  // "stripping an image saves chars" assertion below is actually meaningful.
  const fakeImageData = 'data:image/png;base64,' + 'A'.repeat(4000)
  const blocks: ContentBlock[] = [{ type: 'text', text: 'screenshot attached' }, { type: 'image_url', image_url: { url: fakeImageData } }]
  const oldImageMsg: ApiMessage = { role: 'tool', tool_call_id: 'cOld', name: 'screenshot', content: blocks }
  const recentImageMsg: ApiMessage = { role: 'tool', tool_call_id: 'cRecent', name: 'screenshot', content: blocks.map((b) => ({ ...b })) as ContentBlock[] }
  const msgs: ApiMessage[] = [
    asstCall('cOld', 'screenshot'), oldImageMsg,
    user('pad1'), user('pad2'), user('pad3'), user('pad4'),
    asstCall('cRecent', 'screenshot'), recentImageMsg, // near the tail -> recent
  ]
  const { messages: out, savedChars } = prePassCompact(msgs)
  const oldOut = out[1].content as ContentBlock[]
  assert.ok(Array.isArray(oldOut), 'old image message stays multimodal-shaped (array content)')
  assert.ok(!oldOut.some((b) => b.type === 'image_url'), 'stale image part stripped from the old message')
  assert.ok(oldOut.some((b) => b.type === 'text' && /screenshot attached/.test(b.text)), 'sibling text block kept intact')
  assert.ok(oldOut.some((b) => b.type === 'text' && /image omitted/i.test(b.text)), 'image replaced with a named placeholder')
  const recentOut = out[7].content as ContentBlock[]
  assert.ok(recentOut.some((b) => b.type === 'image_url'), 'image inside the recent window is left alone')
  assert.ok(savedChars > 0, 'reports saved chars from stripping the stale image')
  ok('strips stale image content-parts outside the recent window, leaves recent ones alone')
}

console.log(`\n${passed} checks passed.`)
