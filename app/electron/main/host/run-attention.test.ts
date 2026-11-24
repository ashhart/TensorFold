// Run: jiti electron/main/host/run-attention.test.ts
//
// Native attention for runs: when to notify, what it says, and the needs-you dock badge.
import assert from 'node:assert/strict'
import { attentionNotice, badgeText, createRunAttention, nextNeedsYou, shouldNotify, type AttentionNotice } from './run-attention'
import { pairedSettingsDefaults } from '../../../src/lib/paired-settings'
import type { AgentEvent } from '../../agent/types'
import { store } from '../state'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('run-attention tests:')

{
  const defaults = pairedSettingsDefaults()
  assert.equal(defaults.completionSound, false, 'a fresh profile does not opt in to completion sounds')
  assert.equal(defaults.notifyRuns, true, 'background notifications keep their existing default')
  ok('completion sound is opt-in and independent of background notifications')
}

{
  assert.equal(shouldNotify({ enabled: true, focused: false, supported: true, knownThread: true }), true)
  assert.equal(shouldNotify({ enabled: true, focused: true, supported: true, knownThread: true }), false)
  assert.equal(shouldNotify({ enabled: false, focused: false, supported: true, knownThread: true }), false)
  assert.equal(shouldNotify({ enabled: true, focused: false, supported: false, knownThread: true }), false)
  assert.equal(shouldNotify({ enabled: true, focused: false, supported: true, knownThread: false }), false)
  ok('notifies only in the background, when enabled and supported, for a known thread')
}

{
  assert.deepEqual(attentionNotice('finished', 't', 'Fix login'), { kind: 'finished', threadId: 't', title: 'Run finished', body: 'Fix login' })
  assert.deepEqual(attentionNotice('approval', 't', 'Fix login', 'run_command'), { kind: 'approval', threadId: 't', title: 'Needs your approval', body: 'Fix login: run run_command?' })
  assert.equal(attentionNotice('failed', 't', null, 'boom').body, 'A thread: boom')
  assert.ok(attentionNotice('failed', 't', 'x', 'e'.repeat(400)).body.length <= 140)
  ok('notice copy is short and names the thread')
}

{
  const s = new Set(['a', 'b'])
  assert.deepEqual([...nextNeedsYou(s, { type: 'status', threadId: 'a', status: 'awaiting' })], ['a', 'b'])
  assert.deepEqual([...nextNeedsYou(s, { type: 'status', threadId: 'a', status: 'tool' })], ['b'])
  assert.deepEqual([...nextNeedsYou(s, { type: 'done', threadId: 'b' })], ['a'])
  assert.deepEqual([...nextNeedsYou(s, { type: 'error', threadId: 'a' })], ['b'])
  assert.deepEqual([...nextNeedsYou(s, { type: 'item_add', threadId: 'a' })], ['a', 'b'])
  assert.equal(badgeText(0), '')
  assert.equal(badgeText(3), '3')
  ok('answered approvals and ended runs leave the needs-you set')
}

{
  const notices: AttentionNotice[] = []
  const badges: string[] = []
  const sounds: string[] = []
  let focused = false
  let enabled = true
  let completionSoundEnabled = false
  let supported = true
  let incognito = false
  const titles: Record<string, string> = { t1: 'Fix login', t2: 'Rotate keys' }
  const attention = createRunAttention({
    isIncognito: () => incognito,
    enabled: () => enabled,
    completionSoundEnabled: () => completionSoundEnabled,
    focused: () => focused,
    supported: () => supported,
    threadTitle: (id) => titles[id],
    playCompletionSound: () => sounds.push('beep'),
    notify: (n) => notices.push(n),
    setBadge: (text) => badges.push(text),
  })
  const ev = (e: Record<string, unknown>) => attention.observe({ runId: 'r1', ...e } as AgentEvent)

  attention.approvalPending({ threadId: 't2', tool: 'run_command' })
  assert.equal(attention.needsYouCount(), 1)
  assert.deepEqual(badges, ['1'])
  assert.equal(notices.at(-1)?.kind, 'approval')

  ev({ threadId: 't2', type: 'status', status: 'tool' })
  assert.deepEqual(badges, ['1', ''])

  ev({ threadId: 't1', type: 'done' })
  ev({ threadId: 't1', type: 'done' })
  assert.equal(notices.filter((n) => n.kind === 'finished').length, 1)
  assert.equal(sounds.length, 0, 'background completion uses a native notice and no sound')

  ev({ threadId: 't2', type: 'error', message: 'build failed' })
  assert.equal(notices.at(-1)?.body, 'Rotate keys: build failed')
  assert.equal(sounds.length, 0, 'errors never play the completion sound')

  focused = true
  supported = false
  ev({ runId: 'r2', threadId: 't1', type: 'done' })
  ev({ runId: 'r2', threadId: 't1', type: 'done' })
  assert.equal(sounds.length, 0, 'completion sound stays off by default even with notifications enabled')
  assert.equal(notices.filter((n) => n.kind === 'finished').length, 1, 'focused completion has no native notice')
  completionSoundEnabled = true
  ev({ runId: 'r3', threadId: 't1', type: 'done' })
  ev({ runId: 'r3', threadId: 't1', type: 'done' })
  assert.equal(sounds.length, 1, 'an opted-in focused completion plays once without native notification support')
  enabled = false
  ev({ runId: 'r4', threadId: 't1', type: 'done' })
  assert.equal(sounds.length, 2, 'run notifications do not disable an explicitly enabled completion sound')
  completionSoundEnabled = false
  ev({ runId: 'r5', threadId: 't1', type: 'done' })
  completionSoundEnabled = true
  ev({ runId: 'r6', threadId: 'unknown', type: 'done' })
  incognito = true
  ev({ runId: 'r7', threadId: 't1', type: 'done' })
  assert.equal(sounds.length, 2, 'disabled, unknown and incognito threads never play a sound')
  assert.equal(notices.length, 3)
  ok('completion sound is opt-in, deduplicated and independent of notifications, with foreground and privacy gates unchanged')
}

{
  const previous = store.settings
  const notices: AttentionNotice[] = [], badges: string[] = [], sounds: string[] = []
  let focused = false
  const attention = createRunAttention({
    enabled: () => true,
    focused: () => focused,
    supported: () => true,
    threadTitle: () => 'Ordinary chat',
    playCompletionSound: () => sounds.push('beep'),
    notify: (notice) => notices.push(notice),
    setBadge: (text) => badges.push(text),
  })
  try {
    store.settings = {} as typeof store.settings
    attention.observe({ threadId: 'ordinary', runId: 'background', type: 'done' })
    focused = true
    attention.observe({ threadId: 'ordinary', runId: 'absent-setting', type: 'done' })
    assert.equal(sounds.length, 0, 'a legacy fixture and absent preference stay silent without a callback')
    assert.deepEqual(notices.map((notice) => notice.kind), ['finished'], 'the ordinary background done path still notifies')
    store.settings = { completionSound: true } as typeof store.settings
    attention.observe({ threadId: 'ordinary', runId: 'opted-in', type: 'done' })
    assert.equal(sounds.length, 1, 'the compliant composition boundary reads an explicit opt-in')
    ok('legacy consumers need no sound callback; live settings remain off when absent and honor opt-in')
  } finally {
    store.settings = previous
  }
}

{
  const notices: AttentionNotice[] = [], badges: string[] = []
  const attention = createRunAttention({ enabled: () => true, focused: () => false, supported: () => true, threadTitle: () => 'Example chat', notify: (notice) => notices.push(notice), setBadge: (text) => badges.push(text) })
  const question: any = { id: 'question-attention', state: 'pending', question: 'Synthetic private question content', choices: ['Amber'] }
  const event: any = { type: 'item_update', threadId: 'thread-question', runId: 'run-question', id: 'tool-question', patch: { question } }
  attention.observe(event)
  attention.observe(event)
  assert.equal(notices.length, 1, 'duplicate pending updates never notify twice for one question')
  assert.equal(notices[0].title, 'Needs your answer')
  assert.equal(notices[0].body, 'Example chat', 'question text and answer choices stay out of lock-screen notification bodies')
  assert.deepEqual(badges, ['1'])
  attention.observe({ ...event, runId: 'older-run', patch: { question: { ...question, id: 'older-question', state: 'interrupted' } } })
  assert.equal(attention.needsYouCount(), 1, 'late recovery of an older question cannot clear the current question badge')
  attention.observe({ ...event, patch: { question: { ...question, state: 'answered', answer: 'Amber' } } })
  assert.equal(attention.needsYouCount(), 0)
  assert.deepEqual(badges, ['1', ''])
  attention.approvalPending({ threadId: event.threadId, tool: 'write_file' })
  attention.observe({ ...event, patch: { question: { ...question, state: 'interrupted' } } })
  assert.equal(attention.needsYouCount(), 1, 'an older question result cannot clear a new mutation approval badge')
  attention.observe({ type: 'status', threadId: event.threadId, runId: event.runId, status: 'tool' })
  assert.equal(attention.needsYouCount(), 0)
  ok('questions get one private needs-you notice and settle the badge after the durable answer')
}

console.log(`\n${passed} run-attention tests passed`)
