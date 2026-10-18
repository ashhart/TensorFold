import assert from 'node:assert/strict'
import { fetchSlackMessages, isSlackChannelAllowed, postSlackMessage, readSlackThread, editSlackMessage } from './slack-live'
import { fetchDiscordMessages, isDiscordChannelAllowed, makeDiscordSource, postDiscordMessage, readDiscordMessage, editDiscordMessage } from './discord-connector'
import type { SlackSourceConfig } from './types'

let networkCalls = 0
const deps = { lookup: async () => [{ address: '8.8.8.8', family: 4 }], fetchImpl: (async () => { networkCalls++; throw new Error('unexpected network call') }) as typeof fetch }
const slack: SlackSourceConfig = { id: 'slack_test', kind: 'api_token', label: 'Test workspace', path: '', token: 'xoxb-example-token', channels: '', enabled: true, createdAt: 0, updatedAt: 0, lastSyncAt: null }
const discord = { ...makeDiscordSource('Test server'), enabled: true, token: 'example-token' }
assert.equal(isSlackChannelAllowed('', 'C123'), false)
assert.equal(isSlackChannelAllowed('*', 'C123'), true)
assert.equal(isSlackChannelAllowed('*, C123', 'C999'), false, 'mixed wildcard does not broaden access')
assert.equal(isDiscordChannelAllowed('', '123'), false)
assert.equal(isDiscordChannelAllowed('*', '123'), true)
assert.equal(isDiscordChannelAllowed('*, 123', '999'), false)
assert.deepEqual(await fetchSlackMessages(slack, deps), [])
assert.deepEqual(await fetchDiscordMessages(discord, deps), [])
await assert.rejects(() => postSlackMessage(slack, 'C123', 'no', undefined, undefined, deps), /Choose allowed/)
await assert.rejects(() => editSlackMessage(slack, 'C123', '1.0', 'no', deps), /Choose allowed/)
await assert.rejects(() => readSlackThread(slack, 'C123', '1.0', deps), /Choose allowed/)
await assert.rejects(() => postDiscordMessage(discord, '123', 'no', undefined, undefined, deps), /Choose allowed/)
await assert.rejects(() => editDiscordMessage(discord, '123', '456', 'no', deps), /Choose allowed/)
await assert.rejects(() => readDiscordMessage(discord, '123', '456', deps), /Choose allowed/)
for (const source of [{ ...slack, channels: '*', enabled: false }, { ...slack, channels: 'C123', enabled: false }]) {
  await assert.rejects(() => fetchSlackMessages(source, deps), /disconnected/)
  await assert.rejects(() => postSlackMessage(source, 'C123', 'no', undefined, undefined, deps), /disconnected/)
}
for (const source of [{ ...discord, allowedChannels: '*', enabled: false }, { ...discord, allowedChannels: '123', enabled: false }]) {
  await assert.rejects(() => fetchDiscordMessages(source, deps), /disconnected/)
  await assert.rejects(() => postDiscordMessage(source, '123', 'no', undefined, undefined, deps), /disconnected/)
}
assert.equal(networkCalls, 0, 'empty and disconnected scopes never touch service history or send endpoints')
console.log('Slack/Discord explicit channel access, disconnected read/write rejection and zero-network blank-scope regressions passed')

const slackRequests: string[] = []
const slackDeps = { lookup: deps.lookup, retry: { maxAttempts: 1 }, fetchImpl: (async (input: any) => {
  const url = new URL(typeof input === 'string' ? input : input?.url ?? String(input))
  slackRequests.push(url.pathname + url.search)
  const body = url.pathname.endsWith('/conversations.list') ? { ok: true, channels: [{ id: 'C123', name: 'selected' }, { id: 'C999', name: 'excluded' }] }
    : url.pathname.endsWith('/conversations.history') ? { ok: true, messages: [{ ts: '1.0', text: 'Allowed message', thread_ts: '1.0', reply_count: 2 }] }
    : { ok: true }
  return new Response(JSON.stringify(body), { headers: { 'content-type': 'application/json' } })
}) as typeof fetch }
const messages = await fetchSlackMessages({ ...slack, channels: 'C123' }, slackDeps)
assert.equal(messages.length, 1)
assert.ok(slackRequests.some((request) => request.includes('channel=C123')))
assert.ok(!slackRequests.some((request) => request.includes('C999')), 'selected read never fetches another channel history')
assert.ok(!slackRequests.some((request) => request.includes('conversations.replies')), 'bot setup never pretends unsupported public/private thread retrieval succeeded')
await assert.rejects(() => fetchSlackMessages({ ...slack, channels: 'C_MISSING' }, slackDeps), /None of the selected/)
const deniedDiscord = { lookup: deps.lookup, retry: { maxAttempts: 1 }, fetchImpl: (async () => new Response(JSON.stringify({ message: 'Forbidden' }), { status: 403, headers: { 'content-type': 'application/json' } })) as typeof fetch }
await assert.rejects(() => fetchDiscordMessages({ ...discord, allowedChannels: '*' }, deniedDiscord), /could not list servers/)
console.log('Selected Slack history-only scope and visible missing-channel/Discord permission errors passed')
