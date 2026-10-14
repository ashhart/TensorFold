// Run: jiti electron/agent/slack-connector.test.ts
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { makeSlackSource, readSlackMessage, scanSlackMessages, slackSourceId, slackSourceRef, summarizeSlack, syncSlackConnector } from './slack-connector'
import type { SlackConnectorSettings } from './types'

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-slack-'))
  await fs.writeFile(path.join(root, 'channels.json'), JSON.stringify([{ id: 'C123', name: 'launch-room' }]), 'utf8')
  await fs.writeFile(path.join(root, 'users.json'), JSON.stringify([
    { id: 'U1', name: 'maya', real_name: 'Maya Chen' },
    { id: 'U2', name: 'sam', real_name: 'Sam Patel' },
  ]), 'utf8')
  await fs.mkdir(path.join(root, 'launch-room'), { recursive: true })
  await fs.writeFile(path.join(root, 'launch-room', '2026-06-20.json'), JSON.stringify([
    { type: 'message', user: 'U1', text: 'Auth edge cases need release-note ownership.', ts: '1781978400.000100' },
    { type: 'message', user: 'U2', text: 'Staging data drift is blocking QA.', ts: '1781979000.000200', thread_ts: '1781978400.000100' },
    { type: 'message', subtype: 'channel_join', user: 'U2', text: '<@U2> joined the channel', ts: '1781979300.000300' },
  ]), 'utf8')

  const source = makeSlackSource(root, 1)
  const settings: SlackConnectorSettings = {
    sources: [{ ...source, id: 'slack', label: 'Launch Slack' }],
    maxMessages: 20,
    lastSyncAt: null,
  }

  const { messages, sources } = await scanSlackMessages(settings, { query: 'drift', limit: 10 })
  assert.equal(messages.length, 1)
  assert.equal(messages[0].channelName, 'launch-room')
  assert.equal(messages[0].senderName, 'Sam Patel')
  assert.match(messages[0].text, /blocking QA/)
  assert.equal(sources[0].lastError, null)
  assert.ok(sources[0].lastSyncAt)
  assert.equal(slackSourceId(messages[0]), `slack:slack:${messages[0].id}`)
  assert.equal(slackSourceRef(messages[0]).connectorId, 'slack')

  const read = await readSlackMessage(settings, slackSourceId(messages[0]))
  assert.equal(read.id, messages[0].id)
  assert.match(read.text, /Staging data drift/)
  assert.equal(read.raw.thread_ts, '1781978400.000100')

  const summary = await summarizeSlack(settings, { limit: 10 })
  assert.match(summary.summary, /2 Slack message/)
  assert.match(summary.summary, /#launch-room/)
  assert.match(summary.summary, /Sources:/)
  assert.match(summary.summary, /slack:slack:/)
  assert.equal(summary.sources.length, 2)

  const sync = await syncSlackConnector(settings, [])
  assert.equal(sync.messages, 2)
  assert.equal(sync.events.length, 2)
  assert.equal(sync.events[0].connectorId, 'slack')
  assert.equal(sync.events[0].sources?.[0]?.connectorId, 'slack')
  assert.ok(sync.syncedAt)

  // REPLY-TO CONTEXT: the second message is a threaded reply (thread_ts -> root ts), so
  // its event quotes the root author + text in both the reply field and the summary.
  const replyEvent = sync.events.find((e) => e.reply)
  assert.ok(replyEvent, 'the threaded reply produced an event with reply context')
  assert.match(replyEvent!.summary, /blocking QA/, 'reply event is the drift message')
  assert.equal(replyEvent!.reply?.author, 'Maya Chen')
  assert.match(replyEvent!.reply?.text ?? '', /Auth edge cases/)
  assert.match(replyEvent!.summary, /In reply to Maya Chen: "Auth edge cases/)
  // The root (non-reply) message carries no reply context.
  const rootEvent = sync.events.find((e) => !e.reply)
  assert.ok(rootEvent, 'the root message event has no reply context')
  assert.match(rootEvent!.summary, /release-note ownership/, 'root event is the auth message')

  // INCREMENTAL CURSOR: sync advanced the source cursor to the newest message; a re-sync
  // that feeds the advanced cursor back emits nothing (everything is at/below the cursor).
  const advancedSource = sync.sources.find((s) => s.id === 'slack')
  assert.ok(advancedSource?.cursor, 'the source cursor advanced')
  assert.equal(advancedSource!.cursor!.position, messages.length ? Math.max(1781978400 * 1000, 1781979000 * 1000) : 0)
  const reSync = await syncSlackConnector({ ...settings, sources: [advancedSource!] }, [])
  assert.equal(reSync.events.length, 0, 'a re-sync with the advanced cursor emits no duplicate events')

  await fs.rm(root, { recursive: true, force: true })
  console.log('slack connector tests passed')
}

void main()
