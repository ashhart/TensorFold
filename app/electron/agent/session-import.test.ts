// Run: jiti electron/agent/session-import.test.ts
//
// Session import parse core (VST-033): Codex rollout + Claude session JSONL → ImportableSession
// descriptors, with defensive parsing (mid-write truncation), title/model/cwd extraction, resume
// cursors matching each transport, and merge/dedup/rank for the import picker.
import assert from 'node:assert/strict'
import {
  parseCodexRollout,
  parseClaudeSession,
  parseSessionFile,
  collectImportableSessions,
  resumeCursorFor,
  scanImportableSessions,
  type SessionFileSystem,
  type ImportSessionSource,
} from './session-import'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('session-import tests:')

const jsonl = (...objs: unknown[]) => objs.map((o) => JSON.stringify(o)).join('\n')

// --- Codex rollout ---

{
  const text = jsonl(
    { type: 'session_meta', payload: { session_id: 'cx-1', timestamp: '2026-07-01T14:30:09.423Z', cwd: '/work/repo', cli_version: '0.142.5' } },
    { type: 'event_msg', timestamp: '2026-07-01T14:30:10.000Z', payload: { type: 'task_started', turn_id: 't1' } },
    { type: 'turn_context', payload: { turn_id: 't1', model: 'gpt-5.5' } },
    { type: 'event_msg', timestamp: '2026-07-01T14:30:11.000Z', payload: { type: 'user_message', message: 'Can you connect to my pgx using ssh pgx' } },
    { type: 'response_item', timestamp: '2026-07-01T14:30:20.000Z', payload: { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'Sure.' }] } },
  )
  const session = parseCodexRollout(text)!
  assert.ok(session, 'a valid rollout parses')
  assert.equal(session.sessionId, 'cx-1')
  assert.equal(session.source, 'codex')
  assert.equal(session.cwd, '/work/repo')
  assert.equal(session.model, 'gpt-5.5')
  assert.equal(session.title, 'Can you connect to my pgx using ssh pgx')
  assert.equal(session.messageCount, 2, 'one user + one assistant message')
  assert.deepEqual(session.resumeCursor, { threadId: 'cx-1' }, 'codex cursor is a threadId')
  assert.equal(session.createdAt, Date.parse('2026-07-01T14:30:09.423Z'))
  assert.equal(session.updatedAt, Date.parse('2026-07-01T14:30:20.000Z'), 'updatedAt is the latest timestamp')
  ok('parseCodexRollout: extracts id, cwd, model, title, counts, cursor, timestamps')
}

{
  // No session_meta → not importable.
  assert.equal(parseCodexRollout(jsonl({ type: 'turn_context', payload: { model: 'gpt-5.5' } })), null)
  ok('parseCodexRollout: rollout without session_meta → null')
}

{
  // Mid-write truncation: a corrupt final line must not abort the parse of good earlier lines.
  const text = [
    JSON.stringify({ type: 'session_meta', payload: { session_id: 'cx-2', timestamp: '2026-07-01T10:00:00Z', cwd: '/x' } }),
    JSON.stringify({ type: 'event_msg', payload: { type: 'user_message', message: 'hello' } }),
    '{"type":"response_item","payload":{"type":"messa',  // truncated JSON
  ].join('\n')
  const session = parseCodexRollout(text)!
  assert.equal(session.sessionId, 'cx-2')
  assert.equal(session.messageCount, 1, 'the corrupt line is skipped, the good user line still counts')
  assert.equal(session.title, 'hello')
  ok('parseCodexRollout: skips a truncated trailing line')
}

{
  // Model unknown until turn_context; title falls back when no user message text.
  const text = jsonl(
    { type: 'session_meta', payload: { session_id: 'cx-3', timestamp: '2026-07-01T10:00:00Z' } },
    { type: 'response_item', payload: { type: 'message', role: 'assistant', content: [{ type: 'output_text', text: 'hi' }] } },
  )
  const session = parseCodexRollout(text)!
  assert.equal(session.model, null)
  assert.equal(session.cwd, null)
  assert.equal(session.title, 'Imported Codex session', 'falls back to a generic title')
  ok('parseCodexRollout: null model/cwd + fallback title when absent')
}

// --- Claude session ---

{
  const text = jsonl(
    { type: 'queue-operation', operation: 'enqueue', timestamp: '2026-07-01T19:40:00Z', sessionId: 'cl-1' },
    { type: 'user', timestamp: '2026-07-01T19:40:37.293Z', sessionId: 'cl-1', cwd: '/Volumes/repo', message: { role: 'user', content: 'Get up to speed.' } },
    { type: 'assistant', timestamp: '2026-07-01T19:41:00Z', sessionId: 'cl-1', message: { role: 'assistant', model: 'claude-fable-5', content: [{ type: 'text', text: 'On it.' }] } },
  )
  const session = parseClaudeSession(text)!
  assert.equal(session.sessionId, 'cl-1')
  assert.equal(session.source, 'claude')
  assert.equal(session.cwd, '/Volumes/repo', 'cwd comes from the first line that carries it')
  assert.equal(session.model, 'claude-fable-5')
  assert.equal(session.title, 'Get up to speed.', 'title falls back to the first user message')
  assert.equal(session.messageCount, 2)
  assert.deepEqual(session.resumeCursor, { sessionId: 'cl-1' }, 'claude cursor is a sessionId')
  ok('parseClaudeSession: extracts id, cwd, model, title, counts, cursor')
}

{
  // A custom-title line wins over the first user message.
  const text = jsonl(
    { type: 'user', timestamp: '2026-07-01T19:40:00Z', sessionId: 'cl-2', message: { role: 'user', content: 'first ask that is long' } },
    { type: 'custom-title', timestamp: '2026-07-01T19:40:05Z', sessionId: 'cl-2', content: 'My Renamed Session' },
    { type: 'assistant', timestamp: '2026-07-01T19:40:10Z', sessionId: 'cl-2', message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] } },
  )
  const session = parseClaudeSession(text)!
  assert.equal(session.title, 'My Renamed Session', 'explicit title beats the first user message')
  ok('parseClaudeSession: custom-title wins over first user message')
}

{
  assert.equal(parseClaudeSession(jsonl({ type: 'summary', summary: 'x' })), null, 'no sessionId → null')
  ok('parseClaudeSession: no sessionId → null')
}

// --- dispatch + merge/rank ---

{
  assert.equal(parseSessionFile('codex', jsonl({ type: 'session_meta', payload: { session_id: 'z', timestamp: '2026-07-01T00:00:00Z' }, }, { type: 'event_msg', payload: { type: 'user_message', message: 'hi' } }))?.source, 'codex')
  assert.equal(parseSessionFile('claude', jsonl({ type: 'user', sessionId: 'z', message: { content: 'hi' } }))?.source, 'claude')
  ok('parseSessionFile: dispatches on source')
}

{
  const a = parseCodexRollout(jsonl(
    { type: 'session_meta', payload: { session_id: 'dup', timestamp: '2026-07-01T10:00:00Z' } },
    { type: 'event_msg', timestamp: '2026-07-01T10:00:00Z', payload: { type: 'user_message', message: 'old' } },
  ))
  const b = parseCodexRollout(jsonl(
    { type: 'session_meta', payload: { session_id: 'dup', timestamp: '2026-07-01T12:00:00Z' } },
    { type: 'event_msg', timestamp: '2026-07-01T12:00:00Z', payload: { type: 'user_message', message: 'new' } },
  ))
  const empty = parseCodexRollout(jsonl({ type: 'session_meta', payload: { session_id: 'empty', timestamp: '2026-07-01T13:00:00Z' } }))
  const recent = parseClaudeSession(jsonl(
    { type: 'user', timestamp: '2026-07-01T20:00:00Z', sessionId: 'cl-9', message: { content: 'newest' } },
    { type: 'assistant', timestamp: '2026-07-01T20:00:00Z', sessionId: 'cl-9', message: { content: [{ type: 'text', text: 'x' }] } },
  ))
  const merged = collectImportableSessions([a, b, null, empty, recent])
  assert.equal(merged.length, 2, 'dedup by id + drop 0-message sessions')
  assert.equal(merged[0].sessionId, 'cl-9', 'newest-updated first')
  assert.equal(merged[1].sessionId, 'dup')
  assert.equal(merged[1].title, 'new', 'the newer of two duplicate ids is kept')
  ok('collectImportableSessions: dedups by id, drops empty, ranks newest-first')
}

{
  // The host rebuilds cursors from source+id so a crafted IPC payload can't smuggle an
  // arbitrary cursor object into a persisted thread.
  assert.deepEqual(resumeCursorFor('codex', 'abc'), { threadId: 'abc' })
  assert.deepEqual(resumeCursorFor('claude', 'abc'), { sessionId: 'abc' })
  ok('resumeCursorFor: transport-matched cursor shapes')
}

// --- scan orchestration (injected fs) ---

{
  const files: Record<string, string> = {
    '/codex/a.jsonl': jsonl(
      { type: 'session_meta', payload: { session_id: 'cx-a', timestamp: '2026-07-01T08:00:00Z', cwd: '/a' } },
      { type: 'event_msg', timestamp: '2026-07-01T08:00:01Z', payload: { type: 'user_message', message: 'codex ask' } },
    ),
    '/codex/corrupt.jsonl': '{not json at all',
    '/claude/b.jsonl': jsonl(
      { type: 'user', timestamp: '2026-07-01T09:00:00Z', sessionId: 'cl-b', cwd: '/b', message: { content: 'claude ask' } },
      { type: 'assistant', timestamp: '2026-07-01T09:00:01Z', sessionId: 'cl-b', message: { model: 'claude-fable-5', content: [{ type: 'text', text: 'y' }] } },
    ),
    '/claude/missing.jsonl': '', // read returns empty → skipped
  }
  const fakeFs: SessionFileSystem = {
    listFiles: async (source: ImportSessionSource) => source === 'codex' ? ['/codex/a.jsonl', '/codex/corrupt.jsonl'] : ['/claude/b.jsonl', '/claude/missing.jsonl'],
    readFile: async (p: string) => files[p] ?? '',
  }
  const sessions = await scanImportableSessions(fakeFs)
  assert.equal(sessions.length, 2, 'a valid codex + valid claude session; corrupt/empty dropped')
  assert.deepEqual(sessions.map((s) => s.sessionId).sort(), ['cl-b', 'cx-a'])
  assert.ok(sessions.every((s) => s.messageCount > 0))
  ok('scanImportableSessions: lists → reads → parses across sources, drops corrupt/empty')
}

{
  // A listFiles rejection for one source must not sink the other.
  const fakeFs: SessionFileSystem = {
    listFiles: async (source: ImportSessionSource) => { if (source === 'codex') throw new Error('permission denied'); return ['/c.jsonl'] },
    readFile: async () => jsonl(
      { type: 'user', timestamp: '2026-07-01T09:00:00Z', sessionId: 'cl-only', message: { content: 'hi' } },
      { type: 'assistant', timestamp: '2026-07-01T09:00:01Z', sessionId: 'cl-only', message: { content: [{ type: 'text', text: 'y' }] } },
    ),
  }
  const sessions = await scanImportableSessions(fakeFs)
  assert.deepEqual(sessions.map((s) => s.sessionId), ['cl-only'], 'a failed source is isolated')
  ok('scanImportableSessions: a failing source does not sink the scan')
}

{
  // maxFilesPerSource caps the scan.
  const fakeFs: SessionFileSystem = {
    listFiles: async () => ['/1.jsonl', '/2.jsonl', '/3.jsonl'],
    readFile: async (p: string) => jsonl(
      { type: 'user', timestamp: '2026-07-01T09:00:00Z', sessionId: `s${p}`, message: { content: 'x' } },
      { type: 'assistant', timestamp: '2026-07-01T09:00:01Z', sessionId: `s${p}`, message: { content: [{ type: 'text', text: 'y' }] } },
    ),
  }
  const sessions = await scanImportableSessions(fakeFs, { sources: ['claude'], maxFilesPerSource: 2 })
  assert.equal(sessions.length, 2, 'only the first 2 files per source are parsed')
  ok('scanImportableSessions: honours maxFilesPerSource')
}

console.log(`${passed} passed`)
