// Run: jiti electron/agent/channel-approvals.test.ts
import assert from 'node:assert/strict'
import {
  parseApprovalReply,
  plainTextApprovalPrompt,
  requestApprovalViaChannel,
  inlineApprovalPlatforms,
  type ChannelAdapter,
} from './channel-adapter.ts'
import {
  createChannelApprovalRegistry,
  slackApprovalBlocks,
  discordApprovalComponents,
  telegramApprovalKeyboard,
} from './channel-approvals.ts'
import { selectChannelPrompt, renderChannelPrompt, withEphemeralChannelPrompt } from './channel-prompt.ts'
import type { ApiMessage } from './types/base.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('channel-approvals tests:')

// ---- parseApprovalReply: the in-channel decision parser ----
{
  // Inline button payloads, slash commands, plain words, and emoji all map to a decision.
  for (const yes of ['approve', '/approve', 'APPROVED', 'approve:ca_ab12', '/approve:ca_ab12', 'yes', 'y', 'ok', '👍', '✅']) {
    assert.equal(parseApprovalReply(yes), 'approve', `"${yes}" -> approve`)
  }
  for (const no of ['deny', '/deny', 'Denied', 'deny:ca_ab12', '/deny:ca_ab12', 'no', 'n', 'cancel', '👎', '❌']) {
    assert.equal(parseApprovalReply(no), 'deny', `"${no}" -> deny`)
  }
  // An ordinary chat message is NOT a decision (so we never auto-approve real conversation).
  for (const neither of ['', '   ', 'can you deploy this?', 'approval', 'noted', 'yesterday', 'okay so then']) {
    assert.equal(parseApprovalReply(neither), null, `"${neither}" -> null`)
  }
  ok('parseApprovalReply maps approve/deny payloads and ignores ordinary text')
}

// ---- requestApprovalViaChannel: inline path resolves the decision ----
{
  let prompted: { title: string } | null = null
  const inline: ChannelAdapter = {
    platform: 'slack',
    supportsInlineApproval: true,
    send: async () => ({ ok: true }),
    requestApproval: async (_addr, request) => { prompted = { title: request.title }; return 'approve' },
  }
  const r = await requestApprovalViaChannel([inline], 'slack:T1:C2:169.1', { title: 'Send reply', summary: 'to #launch' })
  assert.equal(r.ok, true)
  if (r.ok) assert.equal(r.decision, 'approve')
  assert.equal(prompted!.title, 'Send reply', 'the adapter saw the request')
  ok('requestApprovalViaChannel: inline adapter resolves the decision in-channel')
}

// ---- requestApprovalViaChannel: plain-text channels post a prompt + fall back to the app ----
{
  const sent: string[] = []
  const plain: ChannelAdapter = {
    platform: 'imessage',
    send: async (_addr, reply) => { sent.push(reply.text); return { ok: true } },
  }
  const r = await requestApprovalViaChannel([plain], 'imessage:+447', { title: 'Send reply', summary: 'to Maya', command: 'imsg send' })
  assert.equal(r.ok, false)
  if (!r.ok) assert.equal(r.fallbackToApp, true, 'caller is told to use the in-app gate')
  assert.equal(sent.length, 1, 'a plain-text prompt was posted to the channel')
  assert.match(sent[0], /\/approve/, 'the prompt tells the user how to answer')
  assert.match(sent[0], /imsg send/, 'the gated action is shown')
  ok('requestApprovalViaChannel: plain-text channel posts prompt and defers to the app')
}

// ---- requestApprovalViaChannel: unknown platform / inline adapter throw ----
{
  const unknown = await requestApprovalViaChannel([], 'slack:a:b', { title: 't', summary: 's' })
  assert.equal(unknown.ok, false)
  if (!unknown.ok) assert.equal(unknown.fallbackToApp, true)

  const boom: ChannelAdapter = { platform: 'discord', supportsInlineApproval: true, send: async () => ({ ok: true }), requestApproval: async () => { throw new Error('rate limited') } }
  const thrown = await requestApprovalViaChannel([boom], 'discord:s:c', { title: 't', summary: 's' })
  assert.equal(thrown.ok, false)
  if (!thrown.ok) { assert.match(thrown.error || '', /rate limited/); assert.equal(thrown.fallbackToApp, true) }
  ok('requestApprovalViaChannel: unknown platform + adapter throw fall back, never reject')
}

// ---- inlineApprovalPlatforms capability listing ----
{
  const adapters: ChannelAdapter[] = [
    { platform: 'imessage', send: async () => ({ ok: true }) },
    { platform: 'slack', supportsInlineApproval: true, send: async () => ({ ok: true }), requestApproval: async () => 'approve' },
    { platform: 'discord', supportsInlineApproval: true, send: async () => ({ ok: true }), requestApproval: async () => 'deny' },
  ]
  assert.deepEqual(inlineApprovalPlatforms(adapters).sort(), ['discord', 'slack'])
  ok('inlineApprovalPlatforms lists only adapters that can resolve in-channel')
}

// ---- the pending-approval registry: block until answered, by token or by thread ----
{
  const reg = createChannelApprovalRegistry(0) // 0 = no auto-deny timeout, for the test
  const a = reg.create('slack', 'slack:s:c:1')
  assert.equal(reg.size(), 1)
  // Resolve by token (a button callback) settles the exact parked promise.
  assert.equal(reg.resolve(a.token, 'approve'), true)
  assert.equal(await a.promise, 'approve')
  assert.equal(reg.size(), 0)
  // A second resolve of the same token is a no-op (already settled).
  assert.equal(reg.resolve(a.token, 'deny'), false)

  // Resolve by threadKey (a plain "/approve" reply with no token) hits the newest for it.
  const b = reg.create('telegram', 'telegram:123')
  assert.equal(reg.resolveByThreadKey('telegram:123', 'deny'), true)
  assert.equal(await b.promise, 'deny')
  // Nothing parked for an unknown thread.
  assert.equal(reg.resolveByThreadKey('telegram:999', 'approve'), false)
  ok('approval registry resolves by token and by threadKey, idempotently')
}

// ---- inline-control markup carries the token so the callback can resolve it ----
{
  const blocks = slackApprovalBlocks({ title: 'Send', summary: 'to #x', command: 'slack send' }, 'ca_tok1') as any[]
  const actions = blocks.find((b) => b.type === 'actions')
  assert.equal(actions.block_id, 'vontra_approval:ca_tok1')
  assert.deepEqual(actions.elements.map((e: any) => e.value), ['approve:ca_tok1', 'deny:ca_tok1'])

  const components = discordApprovalComponents('ca_tok2') as any[]
  assert.deepEqual(components[0].components.map((c: any) => c.custom_id), ['vontra_approve:ca_tok2', 'vontra_deny:ca_tok2'])

  const kb = telegramApprovalKeyboard('ca_tok3')
  assert.deepEqual(kb.inline_keyboard[0].map((b) => b.callback_data), ['/approve:ca_tok3', '/deny:ca_tok3'])
  // The token round-trips through parseApprovalReply on the way back in.
  assert.equal(parseApprovalReply(kb.inline_keyboard[0][0].callback_data), 'approve')
  assert.equal(parseApprovalReply(kb.inline_keyboard[0][1].callback_data), 'deny')
  ok('Slack/Discord/Telegram markup embeds the token and parses back to a decision')
}

// ---- plainTextApprovalPrompt formats title/summary/action + instructions ----
{
  const text = plainTextApprovalPrompt({ title: 'Send email', summary: 'to Maya about renewal', command: 'email send' })
  assert.match(text, /Send email/)
  assert.match(text, /to Maya about renewal/)
  assert.match(text, /email send/)
  assert.match(text, /\/approve/)
  assert.match(text, /\/deny/)
  ok('plainTextApprovalPrompt includes the action and how to answer')
}

// ---- Feature B: selectChannelPrompt picks the right source's prompt by threadKey ----
{
  const sources = {
    slack: { sources: [
      { id: 'sl_a', channelPrompt: 'Be terse and link the PR.' } as any,
      { id: 'sl_b', channelPrompt: '' } as any,
    ] },
    discord: { sources: [{ id: 'dc_a', channelPrompt: 'Use casual tone.' } as any] },
    email: { sources: [{ id: 'em_a', enabled: true, channelPrompt: 'Formal sign-off required.' } as any] },
  }
  // Slack key: account segment = source id (slack:<src>:<chan>:<ts>).
  assert.equal(selectChannelPrompt('slack:sl_a:C1:1.2', sources), 'Be terse and link the PR.')
  // A source with a blank prompt yields null (not an empty block).
  assert.equal(selectChannelPrompt('slack:sl_b:C1:1.2', sources), null)
  // Discord key: account segment = source id (discord:<src>:<chan>).
  assert.equal(selectChannelPrompt('discord:dc_a:C9', sources), 'Use casual tone.')
  // Email keys carry no account segment; fall back to the first enabled source with a prompt.
  assert.equal(selectChannelPrompt('email:root-123', sources), 'Formal sign-off required.')
  // Unknown platform / source / empty key -> null.
  assert.equal(selectChannelPrompt('whatsapp:acct:jid', sources), null)
  assert.equal(selectChannelPrompt('slack:nope:C1:1.2', sources), null)
  assert.equal(selectChannelPrompt('', sources), null)
  ok('selectChannelPrompt resolves per-source prompts and ignores channels without one')
}

// ---- renderChannelPrompt wraps a prompt and blanks an empty one ----
{
  assert.equal(renderChannelPrompt(''), '')
  assert.equal(renderChannelPrompt('   '), '')
  const rendered = renderChannelPrompt('Keep replies under 2 lines.')
  assert.match(rendered, /CHANNEL GUIDANCE/)
  assert.match(rendered, /current channel/)
  assert.match(rendered, /Keep replies under 2 lines\./)
  ok('renderChannelPrompt labels the override and skips empty prompts')
}

// ---- Activation: a connector run's sourceThreadKey resolves the right source prompt ----
// This is the end-to-end selection the startThreadRun bridge now performs: it passes the
// originating threadKey (StartRunPayload.sourceThreadKey) + the live settings to
// selectChannelPrompt. Confirm a real run's key picks its source's prompt, and an
// app-launched run (no key) gets none.
{
  const settings = {
    slack: { sources: [{ id: 'launch', enabled: true, channelPrompt: 'Link the PR and keep it terse.' } as any] },
    discord: { sources: [{ id: 'guild', enabled: true, channelPrompt: 'Casual tone; emoji ok.' } as any] },
    email: { sources: [{ id: 'inbox', enabled: true, channelPrompt: 'Formal sign-off required.' } as any] },
  }
  // Slack run on slack:launch:C123:170.5 -> that source's prompt.
  assert.equal(selectChannelPrompt('slack:launch:C123:170.5', settings), 'Link the PR and keep it terse.')
  // Discord run on discord:guild:9001 -> that source's prompt.
  assert.equal(selectChannelPrompt('discord:guild:9001', settings), 'Casual tone; emoji ok.')
  // Email run (no account segment) -> first enabled source with a prompt.
  assert.equal(selectChannelPrompt('email:root-7', settings), 'Formal sign-off required.')
  // A normal in-app run carries NO sourceThreadKey -> no channel prompt at all.
  assert.equal(selectChannelPrompt(undefined, settings), null)
  assert.equal(selectChannelPrompt(null, settings), null)
  ok('connector run sourceThreadKey selects its source prompt; an in-app run gets none')
}

// ---- withEphemeralChannelPrompt injects after the snapshot AND never mutates the base ----
// The injected channel guidance must be a SEPARATE ephemeral system message that never lands
// in the persisted history. Guard the exact invariant the loop relies on.
{
  const base: ApiMessage[] = [
    { role: 'system', content: 'SNAPSHOT system prompt.' },
    { role: 'user', content: 'hello' },
  ]
  const snapshotBefore = JSON.stringify(base)

  const out = withEphemeralChannelPrompt(base, 'Be terse.')
  // The base array is untouched (same length + identical content) — nothing persisted.
  assert.equal(base.length, 2)
  assert.equal(JSON.stringify(base), snapshotBefore)
  assert.notEqual(out, base) // a new array, not the same reference
  // The ephemeral block is inserted right AFTER the first system message (index 1).
  assert.equal(out.length, 3)
  assert.equal(out[0].role, 'system')
  assert.equal(out[0].content, 'SNAPSHOT system prompt.')
  assert.equal(out[1].role, 'system')
  assert.match(String(out[1].content), /CHANNEL GUIDANCE/)
  assert.match(String(out[1].content), /Be terse\./)
  assert.equal(out[2].role, 'user')

  // An empty/blank prompt is a no-op: the SAME base reference comes back (no allocation, no
  // stray ephemeral system message).
  assert.equal(withEphemeralChannelPrompt(base, ''), base)
  assert.equal(withEphemeralChannelPrompt(base, '   '), base)
  assert.equal(withEphemeralChannelPrompt(base, undefined), base)

  // With no system message present, the ephemeral block goes first (still a fresh array).
  const noSystem: ApiMessage[] = [{ role: 'user', content: 'hi' }]
  const prepended = withEphemeralChannelPrompt(noSystem, 'Use casual tone.')
  assert.equal(noSystem.length, 1)
  assert.equal(prepended.length, 2)
  assert.equal(prepended[0].role, 'system')
  assert.match(String(prepended[0].content), /Use casual tone\./)
  ok('withEphemeralChannelPrompt injects after the snapshot and never mutates/persists the base')
}

console.log(`\n${passed} passed`)
