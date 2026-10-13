import { createHash } from 'node:crypto'
import { chmodSync, mkdirSync, unlinkSync } from 'node:fs'
import path from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { contentText } from '../agent/content'
import { stripIncognitoThreads } from '../agent/incognito'
import { searchSessionThreads, type SessionSearchOptions, type SessionSearchResult } from '../agent/session-recall'
import type { Thread, TimelineItem } from '../agent/types'

export type BlindSearchToken = (token: string) => string

const SCHEMA_VERSION = 1
const MAX_UNIQUE_TOKENS_PER_THREAD = 20_000
const databases = new Map<string, DatabaseSync>()

function openIndex(file: string): DatabaseSync {
  const absolute = path.resolve(file)
  const cached = databases.get(absolute)
  if (cached) return cached
  mkdirSync(path.dirname(absolute), { recursive: true, mode: 0o700 })

  const create = () => {
    let db: DatabaseSync | null = null
    try {
      db = new DatabaseSync(absolute)
      return initSchema(db)
    } catch (error) {
      // A connection that fails mid-DDL must not leak a live handle on the file we
      // are about to unlink and recreate below.
      try { db?.close() } catch { /* already closed */ }
      throw error
    }
  }

  const initSchema = (db: DatabaseSync): DatabaseSync => {
    db.exec('PRAGMA journal_mode = DELETE; PRAGMA synchronous = NORMAL;')
    const version = Number(db.prepare('PRAGMA user_version').get()?.user_version ?? 0)
    if (version !== 0 && version !== SCHEMA_VERSION) {
      db.exec('DROP TABLE IF EXISTS thread_search_fts; DROP TABLE IF EXISTS thread_search;')
    }
    db.exec(`
      CREATE TABLE IF NOT EXISTS thread_search (
        rowid INTEGER PRIMARY KEY,
        thread_id TEXT NOT NULL UNIQUE,
        project_key TEXT NOT NULL,
        archived INTEGER NOT NULL,
        updated_at INTEGER NOT NULL,
        fingerprint TEXT NOT NULL,
        token_hashes TEXT NOT NULL
      );
      CREATE VIRTUAL TABLE IF NOT EXISTS thread_search_fts USING fts5(
        token_hashes,
        content='thread_search',
        content_rowid='rowid'
      );
      CREATE TRIGGER IF NOT EXISTS thread_search_ai AFTER INSERT ON thread_search BEGIN
        INSERT INTO thread_search_fts(rowid, token_hashes) VALUES (new.rowid, new.token_hashes);
      END;
      CREATE TRIGGER IF NOT EXISTS thread_search_ad AFTER DELETE ON thread_search BEGIN
        INSERT INTO thread_search_fts(thread_search_fts, rowid, token_hashes) VALUES ('delete', old.rowid, old.token_hashes);
      END;
      CREATE TRIGGER IF NOT EXISTS thread_search_au AFTER UPDATE ON thread_search BEGIN
        INSERT INTO thread_search_fts(thread_search_fts, rowid, token_hashes) VALUES ('delete', old.rowid, old.token_hashes);
        INSERT INTO thread_search_fts(rowid, token_hashes) VALUES (new.rowid, new.token_hashes);
      END;
      PRAGMA user_version = ${SCHEMA_VERSION};
    `)
    chmodSync(absolute, 0o600)
    return db
  }

  try {
    const db = create()
    databases.set(absolute, db)
    return db
  } catch (error) {
    // This file is derived from the protected canonical thread store. A corrupt or
    // incompatible index is safe to replace and will be rebuilt by the next sync.
    try { unlinkSync(absolute) } catch { /* absent or locked */ }
    const db = create()
    databases.set(absolute, db)
    return db
  }
}

export function searchTerms(value: string): string[] {
  return [...new Set(String(value ?? '')
    .toLowerCase()
    .split(/[^a-z0-9_/-]+/)
    .map((term) => term.trim())
    .filter((term) => term.length >= 2))]
}

function searchableTimelineText(item: TimelineItem): string {
  return [item.title, item.tool, item.text, item.command, item.meta].filter(Boolean).join(' ')
}

function searchableThreadText(thread: Thread): string {
  return [
    thread.title,
    ...(thread.apiMessages ?? []).map((message) => contentText(message.content)),
    ...(thread.timeline ?? []).map(searchableTimelineText),
  ].filter(Boolean).join('\n')
}

// Most-recent-first searchable segments: the per-thread token cap then drops the
// OLDEST material instead of silently unindexing the recent tail of long threads.
function recencyOrderedThreadText(thread: Thread): string {
  const messages = thread.apiMessages ?? []
  const timeline = thread.timeline ?? []
  return [
    thread.title,
    ...messages.slice().reverse().map((message) => contentText(message.content)),
    ...timeline.slice().reverse().map(searchableTimelineText),
  ].filter(Boolean).join('\n')
}

function threadTextFingerprint(thread: Thread): string {
  const archived = thread.archivedAt ? 1 : 0
  const updatedAt = thread.updatedAt || thread.createdAt || 0
  // Cheap skip fingerprint over the raw searchable text (hashed, never stored in the
  // clear): unchanged threads must not pay per-token blinding on every persist.
  return createHash('sha256')
    .update(`${thread.projectId ?? ''}\0${archived}\0${updatedAt}\0${recencyOrderedThreadText(thread)}`)
    .digest('hex')
}

function indexThreadTokens(thread: Thread, blind: BlindSearchToken) {
  // Index prefixes so normal full-text queries tolerate plural/suffix variants such
  // as "fixture" matching "fixtures" without storing a reversible word list.
  const tokens = [...new Set(searchTerms(recencyOrderedThreadText(thread)).flatMap((term) => {
    const capped = term.slice(0, 64)
    return Array.from({ length: Math.max(0, capped.length - 1) }, (_, index) => capped.slice(0, index + 2))
  }))].slice(0, MAX_UNIQUE_TOKENS_PER_THREAD)
  return {
    tokenHashes: tokens.map(blind).join(' '),
    projectKey: thread.projectId ? blind(`project:${thread.projectId}`) : '',
  }
}

export function syncSessionSearchIndex(file: string, threads: Thread[], blind: BlindSearchToken): void {
  const db = openIndex(file)
  const existing = new Map(
    (db.prepare('SELECT thread_id, fingerprint FROM thread_search').all() as Array<{ thread_id: string; fingerprint: string }>)
      .map((row) => [row.thread_id, row.fingerprint]),
  )
  const upsert = db.prepare(`
    INSERT INTO thread_search(thread_id, project_key, archived, updated_at, fingerprint, token_hashes)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(thread_id) DO UPDATE SET
      project_key=excluded.project_key,
      archived=excluded.archived,
      updated_at=excluded.updated_at,
      fingerprint=excluded.fingerprint,
      token_hashes=excluded.token_hashes
  `)
  const remove = db.prepare('DELETE FROM thread_search WHERE thread_id = ?')
  const liveIds = new Set<string>()

  db.exec('BEGIN IMMEDIATE')
  try {
    for (const thread of threads) {
      liveIds.add(thread.id)
      const fingerprint = threadTextFingerprint(thread)
      if (existing.get(thread.id) === fingerprint) continue
      const { tokenHashes, projectKey } = indexThreadTokens(thread, blind)
      const archived = thread.archivedAt ? 1 : 0
      const updatedAt = thread.updatedAt || thread.createdAt || 0
      upsert.run(thread.id, projectKey, archived, updatedAt, fingerprint, tokenHashes)
    }
    for (const id of existing.keys()) if (!liveIds.has(id)) remove.run(id)
    db.exec('COMMIT')
  } catch (error) {
    // A failed ROLLBACK (or SQLite's own auto-rollback) must not strand the cached
    // connection inside an open transaction — every later sync would fail. Evict a
    // connection we could not roll back so the next call reopens cleanly.
    try {
      db.exec('ROLLBACK')
    } catch {
      try { db.close() } catch { /* already closed */ }
      databases.delete(path.resolve(file))
    }
    throw error
  }
}

export function searchSessionIndex(
  file: string,
  threads: Thread[],
  options: SessionSearchOptions,
  blind: BlindSearchToken,
): SessionSearchResult {
  // The index only ever receives incognito-stripped threads; the linear-scan fallback
  // must preserve that posture instead of leaking incognito transcripts exactly when
  // the index is broken.
  const visible = stripIncognitoThreads(threads)
  const terms = searchTerms(options.query)
  if (!terms.length) return searchSessionThreads(visible, options)
  try {
    const db = openIndex(file)
    const match = terms.map((term) => `"${blind(term)}"`).join(' OR ')
    const clauses = ['thread_search_fts MATCH ?']
    const params: Array<string | number> = [match]
    if (options.projectId) {
      clauses.push('s.project_key = ?')
      params.push(blind(`project:${options.projectId}`))
    }
    if (!options.includeArchived) clauses.push('s.archived = 0')
    if (!options.includeCurrent && options.currentThreadId) {
      clauses.push('s.thread_id <> ?')
      params.push(options.currentThreadId)
    }
    params.push(200)
    const rows = db.prepare(`
      SELECT s.thread_id
      FROM thread_search_fts
      JOIN thread_search s ON s.rowid = thread_search_fts.rowid
      WHERE ${clauses.join(' AND ')}
      ORDER BY bm25(thread_search_fts), s.updated_at DESC
      LIMIT ?
    `).all(...params) as Array<{ thread_id: string }>
    const ids = new Set(rows.map((row) => row.thread_id))
    return searchSessionThreads(visible.filter((thread) => ids.has(thread.id)), options)
  } catch {
    return searchSessionThreads(visible, options)
  }
}

export function closeSessionSearchIndexes(): void {
  for (const db of databases.values()) {
    try { db.close() } catch { /* already closed */ }
  }
  databases.clear()
}
