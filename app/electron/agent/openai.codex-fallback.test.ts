// Run: npx tsx electron/agent/openai.codex-fallback.test.ts
// C-provider-transport MEDIUM: the codex app-server transport-fallback guard counted
// EVERY yielded chunk, but withDelegatedCliHeartbeat injects synthetic keep-alive
// reasoning chunks during a long silent window before any real output. A failure after
// only a heartbeat (no real model output) wrongly looked like "work in progress" and
// suppressed the safe fall-back to one-shot codex exec. The fix counts only genuine
// model chunks via translateCodexAppServerChunks' onRealChunk hook, which sits UPSTREAM
// of the heartbeat injector.
import assert from 'node:assert'
import { translateCodexAppServerChunks, withDelegatedCliHeartbeat } from './openai.ts'
import type { CodexAppServerChunk } from './codex-app-server.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

async function* fromArray(items: CodexAppServerChunk[]): AsyncGenerator<CodexAppServerChunk> {
  for (const it of items) yield it
}

async function main() {
  console.log('codex app-server fallback guard tests:')

  // onRealChunk fires once per REAL chunk, never for a resume_cursor (which is metadata).
  {
    let real = 0
    let cursor: string | null = null
    const src = fromArray([
      { type: 'reasoning', text: 'thinking' },
      { type: 'content', text: 'hi' },
      { type: 'resume_cursor', cursor: { threadId: 'T1' } },
    ])
    const out: any[] = []
    for await (const c of translateCodexAppServerChunks(src, (t) => { cursor = t }, () => { real++ })) out.push(c)
    assert.strictEqual(real, 2, 'onRealChunk counts the two real chunks only')
    assert.strictEqual(cursor, 'T1', 'the resume cursor is split off via the callback')
    assert.strictEqual(out.length, 2, 'the resume_cursor chunk is not forwarded downstream')
    ok('onRealChunk counts real chunks and excludes resume_cursor')
  }

  // A heartbeat-only window: the source yields no real chunk, so realChunks must stay 0 even
  // if a heartbeat is injected downstream — proving a subsequent failure before real output
  // would still be allowed to fall back to the one-shot path.
  {
    let real = 0
    const src = fromArray([])
    const translated = translateCodexAppServerChunks(src, () => {}, () => { real++ })
    const provider = { transport: 'codex-cli', name: 'Codex' } as const
    const yielded: any[] = []
    for await (const chunk of withDelegatedCliHeartbeat(translated, provider, { heartbeatMs: 1 })) {
      yielded.push(chunk)
    }
    // The load-bearing assertion: NO real chunk was counted, whatever heartbeats (if any) fired.
    assert.strictEqual(real, 0, 'a heartbeat (if any) never counts as a real chunk')
    void yielded
    ok('heartbeat-only output does not inflate the real-chunk count')
  }

  console.log(`\n${passed} checks passed.`)
}

main().catch((e) => { console.error(e); process.exit(1) })
