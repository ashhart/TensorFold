import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { store } from '../state'
import type { Thread } from '../../agent/types'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-paired-edit-redaction-'))
const { pairedRemoteArchiveSnapshot, pairedRemoteControlSnapshot, redactRemoteControlEventForPaired, remoteControlSnapshot, remoteArchiveSnapshot } = await import('./remote-snapshot')

const markers = ['SUPERSEDED_USER_PRIVATE', 'SUPERSEDED_ASSISTANT_PRIVATE', 'SUPERSEDED_TOOL_PRIVATE', 'SUPERSEDED_MODEL_HISTORY_PRIVATE', 'SUPERSEDED_OCR_PRIVATE', 'HIDDEN_EDIT_SOURCE_PRIVATE']
const thread: Thread = {
  id: 'edited-visible-task', title: 'Visible worker', kind: 'task', cwd: '', providerId: null, model: null,
  status: 'idle', createdAt: 1, updatedAt: 2, conversationRevision: 1,
  apiMessages: [{ role: 'user', content: 'Current request' }], timeline: [{ id: 'current', kind: 'user', text: 'Current request' }],
  messageEditRevisions: [{id: 'protected-revision', itemId: 'old-user', editedAt: 2, conversationRevision: 0,
    apiMessages: [{ role: 'user', content: markers[3] }],
    timeline: [
      { id: 'old-user', kind: 'user', text: markers[0], attachments: [{ id: 'old-attachment', kind: 'document', name: 'note.txt', mime: 'text/plain', text: markers[4] }] },
      { id: 'old-answer', kind: 'assistant', text: markers[1] },
      { id: 'old-tool', kind: 'tool', output: markers[2], tool: 'read_file', state: 'ok' },
    ],
  }],
  messageEditSource: { threadId: markers[5], itemId: 'hidden-source-item', editedAt: 2 },
}
const original = { settings: store.settings, threads: store.threads }
try {
  store.settings = { remoteControl: { serverName: 'Synthetic host' }, providers: [] } as any
  store.threads = [thread]
  const runtime = { appVersion: '1', headless: false, activeThreadIds: [], activeRoomIds: [], activeMissionIds: [] }
  const owner = remoteControlSnapshot(runtime)
  const archives = { projects: [], threads: [{ ...thread, archivedAt: 3 }], rooms: [], tickets: [] }
  for (const marker of markers) assert.ok(JSON.stringify(owner).includes(marker), 'owner evidence remains intact')
  assert.ok(JSON.stringify(remoteArchiveSnapshot(archives)).includes(markers[0]), 'owner archive still retains superseded history')
  const projections = [
    pairedRemoteControlSnapshot(runtime),
    pairedRemoteArchiveSnapshot(archives),
    redactRemoteControlEventForPaired({ kind: 'snapshot', data: owner }),
    redactRemoteControlEventForPaired({ kind: 'thread', data: thread }),
    redactRemoteControlEventForPaired({ kind: 'thread', data: archives.threads[0] }),
  ]
  for (const projection of projections) {
    const serialized = JSON.stringify(projection)
    for (const marker of markers) assert.ok(!serialized.includes(marker), `paired active/archive/live projection must omit ${marker}`)
    assert.ok(!serialized.includes('messageEditRevisions'))
    assert.ok(!serialized.includes('messageEditSource'))
    assert.ok(serialized.includes('edited-visible-task'), 'ordinary operational task identity remains available')
  }
  assert.equal(thread.messageEditRevisions?.[0].timeline[0].text, markers[0], 'redaction never mutates the owner record')
  console.log('Paired active, archive, snapshot-event and thread-event projections omit protected edit revisions and hidden source identities while owner evidence remains intact')
} finally { Object.assign(store, original) }
