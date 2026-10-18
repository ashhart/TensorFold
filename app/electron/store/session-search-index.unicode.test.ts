import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { mkdtempSync, rmSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { searchSessionThreads } from '../agent/session-recall'
import type { Thread } from '../agent/types'
import { closeSessionSearchIndexes, searchSessionIndex, syncSessionSearchIndex } from './session-search-index'

const root = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-unicode-search-'))
const file = path.join(root, 'synthetic.sqlite')
const blind = (token: string) => createHash('sha256').update(`synthetic:${token}`).digest('hex')
const cases = [
  ['latin', 'Un café à Montréal', 'cafe montreal'],
  ['decomposed', 'Un cafe\u0301 à Montre\u0301al', 'café montréal'],
  ['cyrillic', 'Синтетический привет мир', 'привет'],
  ['cjk', '合成東京検索テスト中文资料', '東京検索'],
] as const
const threads: Thread[] = cases.map(([id, text], index) => ({
  id, title: 'Synthetic conversation', kind: 'chat', cwd: '', providerId: 'local', model: 'synthetic',
  createdAt: 1, updatedAt: index + 2, status: 'idle',
  apiMessages: [{ role: 'user', content: text }],
  timeline: [{ id: `item-${id}`, kind: 'user', text }],
}))

try {
  syncSessionSearchIndex(file, threads, blind)
  const failures: string[] = []
  for (const [id, , query] of cases) {
    if (!searchSessionThreads(threads, { query }).matches.some((match) => match.threadId === id)) failures.push(`${id}:recall`)
    if (!searchSessionIndex(file, threads, { query }, blind).matches.some((match) => match.threadId === id)) failures.push(`${id}:index`)
  }
  assert.deepEqual(failures, [], 'Unicode message content must be searchable through recall and the hashed index')
  closeSessionSearchIndexes()
  const migratedFile = path.join(root, 'legacy-v1.sqlite')
  const legacy = new DatabaseSync(migratedFile)
  legacy.exec('CREATE TABLE thread_search (obsolete_hash TEXT); PRAGMA user_version = 1;')
  legacy.close()
  syncSessionSearchIndex(migratedFile, threads, blind)
  assert.equal(searchSessionIndex(migratedFile, threads, { query: 'привет' }, blind).matches[0]?.threadId, 'cyrillic', 'v1 index is rebuilt with multilingual tokens')
  closeSessionSearchIndexes()
  const inspected = new DatabaseSync(migratedFile)
  assert.equal(inspected.prepare('PRAGMA user_version').get()?.user_version, 3)
  inspected.close()
  console.log('Unicode search regressions passed')
} finally {
  closeSessionSearchIndexes()
  rmSync(root, { recursive: true, force: true })
}
