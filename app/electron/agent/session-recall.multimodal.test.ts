// Run: npx jiti electron/agent/session-recall.multimodal.test.ts
//
// Regression: session_search / session_read must not drop the typed text of a multimodal
// (ContentBlock[]) message. Before the fix, messageText() returned '' for any array-content
// message, so a user searching prior sessions for text they typed alongside an attachment got
// zero hits and session_read showed a blank body for those turns.
import assert from 'node:assert/strict'
import { readSessionThread, searchSessionThreads } from './session-recall'
import type { Thread } from './types'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

const thread: Thread = {
  id: 'multimodal',
  title: 'Screenshot review',
  kind: 'task',
  projectId: 'p1',
  cwd: '/repo',
  providerId: 'provider',
  model: 'model',
  createdAt: 1,
  updatedAt: 30,
  status: 'idle',
  // Content arrays are what buildUserContent produces when an image is attached.
  apiMessages: [
    {
      role: 'user',
      content: [
        { type: 'text', text: 'Please review this dashboard kpi screenshot for layout bugs.' },
        { type: 'image_url', image_url: { url: 'data:image/png;base64,AAAA' } },
      ],
    } as any,
    { role: 'assistant', content: 'The kpi cards overflow on narrow widths.' },
  ],
  timeline: [],
}

async function run() {
  // --- session_search matches a term that lives only in the array-content message -----
  {
    const result = searchSessionThreads([thread], { query: 'dashboard kpi screenshot', includeCurrent: true })
    assert.equal(result.matches.length, 1, 'the multimodal message text should be searchable')
    assert.equal(result.matches[0].threadId, 'multimodal')
    assert.match(result.matches[0].excerpt, /dashboard kpi screenshot/)
    ok('session_search matches terms inside a multimodal (ContentBlock[]) message')
  }

  // --- session_read surfaces the typed text (not a blank body) for the array message ---
  {
    const read = readSessionThread([thread], { threadId: 'multimodal' })
    const userMsg = read.messages.find((m) => m.role === 'user')
    assert.ok(userMsg, 'the user turn should be present, not dropped')
    assert.match(userMsg!.content, /Please review this dashboard kpi screenshot/)
    ok('session_read returns the typed text of a multimodal message instead of a blank body')
  }

  console.log(`\n${passed} checks passed.`)
}

run().catch((e) => {
  console.error(e)
  process.exit(1)
})
