import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { searchSessionThreads } from '../agent/session-recall'
import type { Thread } from '../agent/types'
import { indexSearchTokens, matchingTerms, searchExcerpt, searchTerms } from '../../shared/conversation-search-text'
import { closeSessionSearchIndexes, searchSessionIndex, syncSessionSearchIndex } from './session-search-index'

const root = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-compound-search-'))
const blind = (token: string) => createHash('sha256').update(`synthetic-compound-key:${token}`).digest('hex')
const text = 'SYNTHETIC_PRIVATE_GOOGLE_AMBER /synthetic/reports/file-final café_notes ПРИВЕТ_МИР ' + 'x'.repeat(90) + '_LONGTAIL'
const thread: Thread = { id: 'compound', title: 'Synthetic conversation', kind: 'chat', cwd: '', providerId: null, model: null,
  createdAt: 1, updatedAt: 2, status: 'idle', apiMessages: [{ role: 'user', content: text }], timeline: [{ id: 'body', kind: 'user', text }] }
const queries = ['AMBER', 'PRIVATE_GOOGLE', '_GOOGLE', '/reports/file', 'file-final', 'final', 'cafe', 'notes', 'мир', 'LONGTAIL']
try {
  const file = path.join(root, 'synthetic.sqlite')
  syncSessionSearchIndex(file, [thread], blind)
  const failures: string[] = []
  const indexed = new Set(indexSearchTokens(text, 20_000))
  for (const query of queries) {
    const terms = searchTerms(query)
    if (!terms.every((term) => indexed.has(term))) failures.push(`${query}:token`)
    if (!searchSessionThreads([thread], { query }).matches.length) failures.push(`${query}:recall`)
    if (!searchSessionIndex(file, [thread], { query }, blind).matches.length) failures.push(`${query}:index`)
  }
  assert.deepEqual(failures, [], 'compound subtokens and path segments must agree in recall, exact matching and indexed narrowing')
  assert.equal(matchingTerms('unrelated google notes', searchTerms('PRIVATE_GOOGLE')).length, 0, 'compound queries are not broadened into an OR of unrelated components')
  assert.equal(matchingTerms('invoice', searchTerms('voice')).length, 0, 'separator matching does not introduce arbitrary word infixes')
  const snippet = searchExcerpt('😀 SYNTHETIC_cafe\u0301_МИР /reports/file-final 東京検索', searchTerms('cafe мир final 東京検索'))
  assert.deepEqual(snippet.highlights.map((range) => snippet.excerpt.slice(range.start, range.end)), ['cafe\u0301', 'МИР', 'final', '東京検索'])
  assert.equal(indexSearchTokens('segment_'.repeat(10_000), 37).length, 37, 'compound expansion remains bounded')
  const repeated = indexSearchTokens('aa_'.repeat(40_000) + 'AMBER', 20_000)
  assert.ok(repeated.includes('amber'), 'repeated compound prefixes do not consume expansion work or lose their distinct tail')
  assert.ok(repeated.length < 5_000)
  assert.equal(readFileSync(file).includes(Buffer.from('AMBER')), false, 'compound index stores no cleartext content')
  const guarded = [
    { ...thread, id: 'google', googleLocalOnly: true },
    { ...thread, id: 'phone', phoneLocalOnly: true },
    { ...thread, id: 'private', incognito: true },
  ]
  syncSessionSearchIndex(file, guarded, blind)
  assert.equal(searchSessionIndex(file, guarded, { query: 'AMBER' }, blind).matches.length, 0, 'compound boundaries cannot expose global protected history')
  for (const currentThreadId of ['google', 'phone']) {
    const own = searchSessionIndex(file, guarded, { query: 'AMBER', currentThreadId, includeCurrent: true }, blind)
    assert.equal(own.matches.length, 1)
    assert.equal(own.matches[0].threadId, currentThreadId)
  }
  assert.equal(searchSessionThreads(guarded, { query: 'AMBER', currentThreadId: 'private', includeCurrent: true }).matches.length, 0, 'agent recall never reads incognito, including its own identity')

  closeSessionSearchIndexes()
  const legacyFile = path.join(root, 'legacy-v2.sqlite')
  const legacy = new DatabaseSync(legacyFile)
  legacy.exec('CREATE TABLE thread_search (obsolete_hash TEXT); PRAGMA user_version = 2;')
  legacy.close()
  syncSessionSearchIndex(legacyFile, [thread], blind)
  assert.equal(searchSessionIndex(legacyFile, [thread], { query: 'AMBER' }, blind).matches[0]?.threadId, thread.id)
  closeSessionSearchIndexes()
  const migrated = new DatabaseSync(legacyFile)
  assert.equal(migrated.prepare('PRAGMA user_version').get()?.user_version, 3)
  migrated.close()
  console.log('Compound/path recall, exact highlights, hashed candidates and v2 migration tests passed')
} finally {
  closeSessionSearchIndexes()
  rmSync(root, { recursive: true, force: true })
}
