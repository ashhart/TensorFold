// Run: jiti electron/agent/outlook-connector.test.ts
//
// Round-trips a FIXTURE Outlook.sqlite (built here with the system `sqlite3` CLI using
// our ASSUMED schema) through the parser, and checks the absent-DB path. The real
// Outlook-for-Mac column names are proprietary/version-specific and cannot be inspected
// over SSH, so the parser introspects the live schema and maps defensively; this test
// pins the behaviour against the documented assumption (see OUTLOOK_SCHEMA in the module).
import assert from 'node:assert/strict'
import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import {
  OUTLOOK_SCHEMA,
  findOutlookDbPath,
  isOutlookSource,
  makeOutlookSource,
  outlookProfileGlobRoot,
  rowToOutlookEmail,
  scanOutlookMessages,
  withDetectedOutlookSource,
} from './outlook-connector.ts'
import type { EmailSourceConfig } from './types.ts'

const execFileP = promisify(execFile)
let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

// Build a fixture DB with the assumed schema (Mail table: subject/sender/recipient/preview/
// time-as-unix-seconds/folder/read flag/message-id). Uses the system sqlite3 CLI so the test
// needs no native module — the same binary the connector itself shells out to.
async function buildFixtureDb(file: string, rows: Array<Record<string, unknown>>): Promise<void> {
  const t = OUTLOOK_SCHEMA.messageTable
  const c = OUTLOOK_SCHEMA.columns
  const ddl = [
    `CREATE TABLE ${t} (`,
    `  ${c.id} INTEGER PRIMARY KEY,`,
    `  ${c.subject} TEXT,`,
    `  ${c.sender} TEXT,`,
    `  ${c.recipient} TEXT,`,
    `  ${c.preview} TEXT,`,
    `  ${c.date} INTEGER,`,
    `  ${c.folder} TEXT,`,
    `  ${c.read} INTEGER,`,
    `  ${c.messageId} TEXT`,
    `);`,
  ].join('\n')
  const lit = (v: unknown) => v == null ? 'NULL' : typeof v === 'number' ? String(v) : `'${String(v).replace(/'/g, "''")}'`
  const inserts = rows.map((r) =>
    `INSERT INTO ${t} (${c.id}, ${c.subject}, ${c.sender}, ${c.recipient}, ${c.preview}, ${c.date}, ${c.folder}, ${c.read}, ${c.messageId}) VALUES (` +
    [r.id, r.subject, r.sender, r.recipient, r.preview, r.date, r.folder, r.read, r.messageId].map(lit).join(', ') + ');',
  ).join('\n')
  await execFileP('sqlite3', [file, `${ddl}\n${inserts}`])
}

async function main() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-outlook-'))
  const dbPath = path.join(root, 'Outlook.sqlite')
  const source: EmailSourceConfig = {
    id: 'olk', label: 'Outlook (local)', path: dbPath, kind: 'outlook_local',
    enabled: true, createdAt: 1, updatedAt: 1, lastSyncAt: null, lastError: null,
  }

  // 1641038400 = 2022-01-01T12:00:00Z (unix seconds). Newest first after sort.
  await buildFixtureDb(dbPath, [
    { id: 1, subject: 'Renewal question', sender: 'Maya <maya@example.test>', recipient: 'user@example.test', preview: 'Can you confirm pricing and rollout dates?', date: 1641038400, folder: 'Inbox', read: 0, messageId: '<one@example.test>' },
    { id: 2, subject: 'Re: Renewal question', sender: 'user@example.test', recipient: 'Maya <maya@example.test>', preview: 'I will send the details today.', date: 1641038460, folder: 'Inbox', read: 1, messageId: '<two@example.test>' },
    // A row with no subject + no preview must still parse (defensive fallbacks), never throw.
    { id: 3, subject: null, sender: 'noreply@example.test', recipient: null, preview: null, date: 1641038500, folder: 'Junk', read: 1, messageId: null },
  ])

  {
    // FIXTURE ROUND-TRIP: rows map into the EmailMessage shape the email feed already uses.
    const messages = await scanOutlookMessages(source, { limit: 50 })
    assert.equal(messages.length, 3)
    // Sorted newest-first by date (id 3 has the latest timestamp).
    assert.deepEqual(messages.map((m) => m.id), ['olk:3', 'olk:2', 'olk:1'])
    const renewal = messages.find((m) => m.subject === 'Renewal question')
    assert.ok(renewal, 'parsed the subject')
    assert.equal(renewal.sourceId, 'olk')
    assert.equal(renewal.sourceLabel, 'Outlook (local)')
    assert.equal(renewal.from, 'Maya <maya@example.test>')
    assert.equal(renewal.to, 'user@example.test')
    assert.equal(renewal.snippet, 'Can you confirm pricing and rollout dates?')
    assert.equal(renewal.unread, true, 'read=0 -> unread')
    assert.equal(renewal.messageId, '<one@example.test>')
    // Date column is unix SECONDS -> milliseconds epoch.
    assert.equal(renewal.date, 1641038400 * 1000)
    // path points at the local store (provenance), not an imap:// URL.
    assert.equal(renewal.path, dbPath)

    const read = messages.find((m) => m.id === 'olk:2')
    assert.ok(read)
    assert.equal(read.unread, false, 'read=1 -> read')

    // Defensive row: missing subject + preview fall back to safe defaults, still emitted.
    const bare = messages.find((m) => m.id === 'olk:3')
    assert.ok(bare, 'a row with null subject/preview still parses')
    assert.equal(typeof bare.subject, 'string')
    assert.equal(typeof bare.snippet, 'string')
    assert.equal(bare.from, 'noreply@example.test')
    ok('scanOutlookMessages parses fixture rows into the EmailMessage shape (newest-first, unread flag, ms date)')
  }

  {
    // unreadOnly + query are honoured by the SAME options the file/IMAP paths take, so the
    // shared scanEmailMessages filters apply uniformly. (The connector's own limit is enough
    // here; we just prove the scanner returns plain EmailMessages the caller can filter.)
    const unread = (await scanOutlookMessages(source, { limit: 50 })).filter((m) => m.unread)
    assert.equal(unread.length, 1)
    assert.equal(unread[0].subject, 'Renewal question')
    ok('scanned messages expose `unread` so the shared unreadOnly filter works')
  }

  {
    // ABSENT DB: a non-existent store degrades QUIETLY to [] (matches imessage "missing DB" behaviour),
    // it does NOT throw. This is what keeps the connector gated/off when Outlook isn't installed.
    const absent: EmailSourceConfig = { ...source, path: path.join(root, 'does-not-exist.sqlite') }
    const messages = await scanOutlookMessages(absent, { limit: 50 })
    assert.deepEqual(messages, [])
    ok('scanOutlookMessages returns [] for an absent DB (degrade quietly, never throw)')
  }

  {
    // GATE: findOutlookDbPath globs the profile dir and returns the first Outlook.sqlite it finds,
    // or null when the profiles root has none. Build a fake "Outlook 15 Profiles/<name>/Data" tree.
    const fakeProfiles = path.join(root, 'Outlook 15 Profiles')
    const profileData = path.join(fakeProfiles, 'My Custom Profile', 'Data')
    await fs.mkdir(profileData, { recursive: true })
    // No DB yet -> null.
    assert.equal(await findOutlookDbPath(fakeProfiles), null)
    // Drop a DB in and it is discovered regardless of the (arbitrary) profile name.
    const realDb = path.join(profileData, 'Outlook.sqlite')
    await buildFixtureDb(realDb, [{ id: 9, subject: 'Hi', sender: 'a@example.test', recipient: 'c@example.test', preview: 'x', date: 1641038400, folder: 'Inbox', read: 1, messageId: '<x@example.test>' }])
    assert.equal(await findOutlookDbPath(fakeProfiles), realDb)
    // A non-existent profiles root just yields null (never throws) — the off-when-absent gate.
    assert.equal(await findOutlookDbPath(path.join(root, 'no-such-root')), null)
    ok('findOutlookDbPath globs the profile dir defensively (DB found by glob; null when absent)')
  }

  {
    // isOutlookSource recognises the new source kind; the default profiles root is under the
    // Office Group Container (so a one-line correction suffices if a real install differs).
    assert.equal(isOutlookSource(source), true)
    assert.equal(isOutlookSource({ ...source, kind: 'eml_dir' }), false)
    assert.match(outlookProfileGlobRoot(), /Group Containers\/UBF8T346G9\.Office\/Outlook\/Outlook 15 Profiles$/)
    ok('isOutlookSource + outlookProfileGlobRoot expose the gate + default path')
  }

  {
    // PURE PARSER: rowToOutlookEmail maps a single raw row (already aliased to canonical keys by the
    // SELECT) and applies fallbacks. Tested directly so the mapping is locked independent of sqlite.
    const m = rowToOutlookEmail(source, {
      _id: 42, _subject: 'Direct', _sender: 'z@example.test', _recipient: 'q@example.test',
      _preview: 'body text', _date: 1641038400, _folder: 'Inbox', _read: 0, _messageId: '<id@example.test>',
    })
    assert.ok(m)
    assert.equal(m.id, 'olk:42')
    assert.equal(m.subject, 'Direct')
    assert.equal(m.unread, true)
    assert.equal(m.date, 1641038400 * 1000)
    // A row with no id is unusable -> null (defensive).
    assert.equal(rowToOutlookEmail(source, { _subject: 'no id' }), null)
    ok('rowToOutlookEmail maps a raw row with fallbacks and rejects an id-less row')
  }

  if (process.platform === 'darwin') {
    // GATED AUTO-PICKUP via a fake $HOME laid out like a real install. withDetectedOutlookSource
    // prepends an ephemeral outlook_local source ONLY when the DB exists under the Office container.
    // (Detection is darwin-only by design — the path layout is mac-specific — so this block is too.)
    const home = path.join(root, 'home')
    const dataDir = path.join(home, 'Library', 'Group Containers', 'UBF8T346G9.Office', 'Outlook', 'Outlook 15 Profiles', 'Main Profile', 'Data')
    await fs.mkdir(dataDir, { recursive: true })

    // No DB yet -> the user's source list is returned unchanged (connector stays OFF).
    const before = await withDetectedOutlookSource([], Date.now(), home)
    assert.deepEqual(before, [], 'no Outlook DB -> no auto source (gate closed)')
    assert.equal(await makeOutlookSource(Date.now(), home), null, 'makeOutlookSource is null with no DB')

    // Drop the DB in -> an ephemeral outlook_local source is auto-prepended.
    const homeDb = path.join(dataDir, 'Outlook.sqlite')
    await buildFixtureDb(homeDb, [{ id: 7, subject: 'Hello from Outlook', sender: 'boss@example.test', recipient: 'me@example.test', preview: 'See attached', date: 1641038400, folder: 'Inbox', read: 0, messageId: '<auto@example.test>' }])
    const after = await withDetectedOutlookSource([], Date.now(), home)
    assert.equal(after.length, 1, 'DB present -> one auto source prepended (gate open)')
    assert.equal(after[0].kind, 'outlook_local')
    assert.equal(after[0].path, homeDb)
    // ...and that ephemeral source actually scans the local store into EmailMessages.
    const scanned = await scanOutlookMessages(after[0] as EmailSourceConfig, { limit: 10 })
    assert.equal(scanned.length, 1)
    assert.equal(scanned[0].subject, 'Hello from Outlook')

    // If the user already configured an Outlook source, we DON'T add a second (respect their config).
    const userSource: EmailSourceConfig = { ...source, id: 'user_olk' }
    const respected = await withDetectedOutlookSource([userSource], Date.now(), home)
    assert.equal(respected.length, 1, 'existing outlook_local source is respected, not duplicated')
    assert.equal(respected[0].id, 'user_olk')
    ok('withDetectedOutlookSource: gated auto-pickup (off without DB, on with DB, respects a user source)')
  }

  await fs.rm(root, { recursive: true, force: true })
  console.log(`\n${passed} checks passed.`)
}

main().catch((e) => { console.error(e); process.exit(1) })
