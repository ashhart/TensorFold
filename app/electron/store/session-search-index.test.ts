import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { closeSessionSearchIndexes, searchSessionIndex, syncSessionSearchIndex } from './session-search-index'
import type { Thread } from '../agent/types'

const root = mkdtempSync(path.join(os.tmpdir(), 'vontra-search-index-'))
const file = path.join(root, 'session-search.sqlite')
const blind = (token: string) => createHash('sha256').update(`test-key:${token}`).digest('hex')
const threads: Thread[] = [{
  id: 'thread-1', title: 'Invoice parser', kind: 'task', projectId: 'project-1', cwd: '/repo',
  providerId: 'provider', model: 'model',
  createdAt: 1, updatedAt: 2, status: 'idle',
  apiMessages: [{ role: 'user', content: 'Repair multiline totals.' }],
  timeline: [{ id: 'item-1', kind: 'tool', title: 'Parser tests', text: 'All invoice fixtures passed.', createdAt: 2 }],
}]

try {
  syncSessionSearchIndex(file, threads, blind)
  const result = searchSessionIndex(file, threads, { query: 'invoice fixtures', projectId: 'project-1' }, blind)
  assert.equal(result.matches[0]?.threadId, 'thread-1')
  const prefixResult = searchSessionIndex(file, threads, { query: 'fixture', projectId: 'project-1' }, blind)
  assert.equal(prefixResult.matches[0]?.threadId, 'thread-1', 'prefix terms match common plural variants')
  const bytes = readFileSync(file)
  assert.equal(bytes.includes(Buffer.from('invoice')), false, 'derived index must not store transcript terms in plaintext')

  syncSessionSearchIndex(file, [], blind)
  const removed = searchSessionIndex(file, threads, { query: 'invoice' }, blind)
  assert.equal(removed.matches.length, 0, 'sync removes rows for deleted threads')

  // Re-syncing unchanged threads must not redo per-token blinding work: the skip
  // fingerprint has to be derivable from the searchable text before tokenization.
  syncSessionSearchIndex(file, threads, blind)
  let blindCalls = 0
  const countingBlind = (token: string) => { blindCalls++; return blind(token) }
  syncSessionSearchIndex(file, threads, countingBlind)
  const before = blindCalls
  syncSessionSearchIndex(file, threads, countingBlind)
  assert.equal(blindCalls, before, 'unchanged threads must not re-run per-token blind hashing on re-sync')

  // The per-thread token cap must keep the RECENT tail of long threads searchable:
  // insertion-order capping indexed only the beginning of the conversation.
  const filler = Array.from({ length: 6000 }, (_, i) => `fillerword${i}x`).join(' ')
  const longThreads: Thread[] = [{
    ...threads[0],
    id: 'thread-long',
    updatedAt: 3,
    apiMessages: [
      { role: 'user', content: filler },
      { role: 'user', content: 'qqzanzibar marker hunt' },
    ],
  }]
  syncSessionSearchIndex(file, longThreads, blind)
  const tail = searchSessionIndex(file, longThreads, { query: 'qqzanzibar' }, blind)
  assert.equal(tail.matches.some((match) => match.threadId === 'thread-long'), true, 'recent tail content must stay indexed past the token cap')

  // A broken index must never fall back to searching incognito transcripts. A directory
  // in place of the sqlite file forces the open failure that triggers the fallback path.
  const incognitoThreads: Thread[] = [
    threads[0],
    { ...threads[0], id: 'thread-secret', incognito: true, apiMessages: [{ role: 'user', content: 'kiwistrawberryplan' }] },
  ]
  const notADb = path.join(root, 'not-a-db')
  mkdirSync(notADb)
  const leaked = searchSessionIndex(notADb, incognitoThreads, { query: 'kiwistrawberryplan' }, blind)
  assert.equal(leaked.matches.some((match) => match.threadId === 'thread-secret'), false, 'fallback linear scan must skip incognito threads')
  const noTerms = searchSessionIndex(notADb, incognitoThreads, { query: '   ' }, blind)
  assert.equal(noTerms.matches.some((match) => match.threadId === 'thread-secret'), false, 'no-terms listing must skip incognito threads')

  console.log('session search index tests passed')
} finally {
  closeSessionSearchIndexes()
  rmSync(root, { recursive: true, force: true })
}
