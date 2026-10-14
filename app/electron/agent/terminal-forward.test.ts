// Run: jiti electron/agent/terminal-forward.test.ts
import assert from 'node:assert/strict'
import {
  parseTerminalStart, parseTerminalWrite, parseTerminalResize, parseTerminalRename,
  terminalDataEvent, decodeTerminalChunk, makeOutputCoalescer,
  remoteTerminalEventToRenderer, stampRemoteTerminalSession,
} from './terminal-forward.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('terminal-forward tests:')

{
  assert.deepEqual(
    stampRemoteTerminalSession({ id: 'remote', hostLocation: 'local', orchestratorHostId: 'spoofed' }, 'mini-a'),
    { id: 'remote', hostLocation: 'orchestrator', managedBy: 'local', orchestratorHostId: 'mini-a' },
    'the authenticated IPC connection stamps and overrides remote origin metadata',
  )
  ok('remote terminal responses are stamped with their authenticated Orchestrator host')
}

// ---- parseTerminalStart ----
{
  assert.deepEqual(parseTerminalStart({ projectId: 'p1', cliId: 'shell' }), { projectId: 'p1', cliId: 'shell' })
  assert.deepEqual(parseTerminalStart({ projectId: 'p1', cliId: 'custom:ls', command: 'ls -la', label: 'ls' }), { projectId: 'p1', cliId: 'custom:ls', command: 'ls -la', label: 'ls' })
  assert.ok('error' in (parseTerminalStart({ cliId: 'shell' }) as any), 'missing projectId → error')
  assert.ok('error' in (parseTerminalStart({ projectId: 'p1' }) as any), 'missing cliId → error')
  assert.ok('error' in (parseTerminalStart(null) as any), 'non-object → error')
  assert.ok('error' in (parseTerminalStart('nope') as any))
  ok('parseTerminalStart: validates projectId+cliId, carries optional command/label')
}

// ---- parseTerminalWrite ----
{
  assert.deepEqual(parseTerminalWrite({ data: 'ls\n' }), { data: 'ls\n' })
  assert.deepEqual(parseTerminalWrite({ data: '' }), { data: '' }) // empty keystroke payload is valid
  assert.ok('error' in (parseTerminalWrite({}) as any), 'missing data → error')
  assert.ok('error' in (parseTerminalWrite({ data: 5 }) as any), 'non-string data → error')
  ok('parseTerminalWrite: requires a string data field (empty allowed)')
}

// ---- parseTerminalResize ----
{
  assert.deepEqual(parseTerminalResize({ cols: 120, rows: 40 }), { cols: 120, rows: 40 })
  assert.deepEqual(parseTerminalResize({ cols: 80.9, rows: 24.9 }), { cols: 80, rows: 24 }) // floored
  assert.ok('error' in (parseTerminalResize({ cols: 0, rows: 24 }) as any), 'non-positive cols → error')
  assert.ok('error' in (parseTerminalResize({ cols: 'x', rows: 24 }) as any), 'non-numeric → error')
  assert.ok('error' in (parseTerminalResize({ cols: 80 }) as any), 'missing rows → error')
  ok('parseTerminalResize: positive integer cols+rows, floors, rejects junk')
}

// ---- parseTerminalRename ----
{
  assert.deepEqual(parseTerminalRename({ title: '  Remote shell  ' }), { title: 'Remote shell' })
  assert.ok('error' in (parseTerminalRename({ title: '   ' }) as any), 'blank title → error')
  assert.ok('error' in (parseTerminalRename({}) as any), 'missing title → error')
  assert.ok('error' in (parseTerminalRename({ title: 5 }) as any), 'non-string title → error')
  ok('parseTerminalRename: requires a non-empty string title and trims it')
}

// ---- data event: base64 round-trip survives multibyte / ANSI ----
{
  const raw = 'héllo [31mred[0m 🚀\n'
  const ev = terminalDataEvent('sess-1', raw)
  assert.equal(ev.type, 'data')
  assert.equal(ev.sessionId, 'sess-1')
  assert.notEqual(ev.chunk, raw, 'chunk is encoded, not raw')
  assert.equal(decodeTerminalChunk(ev.chunk), raw, 'base64 round-trip reconstructs exactly (emoji/ANSI safe)')
  ok('terminalDataEvent base64-encodes; decodeTerminalChunk reverses it (multibyte-safe)')
}

// ---- coalescer: batches per session, drain combines + clears ----
{
  const c = makeOutputCoalescer()
  c.push('a', 'one ')
  c.push('a', 'two ')
  c.push('b', 'B1')
  assert.ok(c.pending() > 0)
  const drained = c.drain()
  // two sessions, session 'a' combined in order
  const byId = Object.fromEntries(drained.map((d) => [d.sessionId, decodeTerminalChunk(d.chunk)]))
  assert.equal(byId['a'], 'one two ', 'session a chunks concatenated in push order')
  assert.equal(byId['b'], 'B1')
  assert.equal(c.pending(), 0, 'drain clears the buffer')
  assert.deepEqual(c.drain(), [], 'a second drain with nothing buffered is empty')
  ok('coalescer: per-session ordered batching, drain combines + clears')
}

// ---- coalescer: maxBytes signals a flush is due ----
{
  const c = makeOutputCoalescer({ maxBytes: 8 })
  assert.equal(c.push('a', 'abc'), false, 'under maxBytes → no flush signal')
  assert.equal(c.push('a', 'defghij'), true, 'crossing maxBytes → flush signal')
  ok('coalescer: push returns true when buffered bytes cross maxBytes')
}

// ---- client mapping: forwarded event → local renderer channel (base64 decoded) ----
{
  const raw = 'output 🚀\n'
  const d = remoteTerminalEventToRenderer(terminalDataEvent('s1', raw), 'mini-a')!
  assert.equal(d.channel, 'terminal:data')
  assert.deepEqual(d.payload, { sessionId: 's1', chunk: raw, orchestratorHostId: 'mini-a' }, 'data chunk is decoded and origin-stamped for the renderer')

  const st = remoteTerminalEventToRenderer({ type: 'status', sessionId: 's1', status: 'busy' }, 'mini-a')!
  assert.equal(st.channel, 'terminal:status')
  assert.deepEqual(st.payload, { sessionId: 's1', status: 'busy', orchestratorHostId: 'mini-a' })

  const ss = remoteTerminalEventToRenderer({ type: 'sessions', projectId: 'p1', sessions: [{ id: 'x' }] }, 'mini-a')!
  assert.equal(ss.channel, 'terminal:sessions')
  assert.deepEqual(ss.payload, {
    projectId: 'p1',
    sessions: [{ id: 'x', hostLocation: 'orchestrator', managedBy: 'local', orchestratorHostId: 'mini-a' }],
    orchestratorHostId: 'mini-a',
  })

  const ex = remoteTerminalEventToRenderer({ type: 'exit', sessionId: 's1' }, 'mini-a')!
  assert.deepEqual(ex, { channel: 'terminal:status', payload: { sessionId: 's1', status: 'done', orchestratorHostId: 'mini-a' } }, 'exit maps to an origin-stamped done status')
  ok('remoteTerminalEventToRenderer: maps data(decoded)/status/sessions/exit to local channels')
}

console.log(`\n${passed} checks passed.`)
