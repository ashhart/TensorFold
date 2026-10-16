import { execFile } from 'node:child_process'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
import type { EmailListOptions, EmailMessage, EmailReadResult, EmailSourceConfig } from './types'

const execFileP = promisify(execFile)

// We shell out to the system `sqlite3` CLI exactly like imessage-connector.ts: it is always
// present on macOS, needs NO native module (better-sqlite3 would break the packaged app since
// npmRebuild is off), and `-readonly` guarantees we can never mutate the user's mail store.
const SQLITE3 = 'sqlite3'

const exec = (cmd: string, args: string[]) =>
  execFileP(cmd, args, { maxBuffer: 16_000_000, timeout: 20_000 }).then(({ stdout }) => stdout)

// ---------------------------------------------------------------------------------------------
// SCHEMA MAP — THE ONE PLACE TO CORRECT AGAINST A REAL OUTLOOK INSTALL.
//
// The new Outlook for Mac ("Outlook 15") stores mail in a SQLite file, but the table/column
// names are PROPRIETARY and version-specific, and this build was developed over SSH with NO
// real Outlook install to inspect — so these names are an INFORMED ASSUMPTION, not verified.
//
// To make a real-install correction a one-line change, every name lives here. At runtime we
// ALSO introspect the live DB (introspectOutlook) and, for each logical field, pick the first
// of [configured name, ...aliases] that actually exists — so a near-miss on the exact column
// name still resolves. `messageTable`/`messageTableAliases` work the same way for the table.
//
// VERIFY THIS MAP against a real `~/Library/Group Containers/UBF8T346G9.Office/.../Outlook.sqlite`
// (`sqlite3 -readonly <db> '.schema'`) before trusting the surfaced data.
// The logical fields we map every Outlook row down to (the EmailMessage essentials).
type LogicalColumn = 'id' | 'subject' | 'sender' | 'recipient' | 'preview' | 'date' | 'folder' | 'read' | 'messageId'

interface OutlookSchemaMap {
  messageTable: string
  messageTableAliases: string[]
  columns: Record<LogicalColumn, string>
  columnAliases: Record<LogicalColumn, string[]>
}

export const OUTLOOK_SCHEMA: OutlookSchemaMap = {
  messageTable: 'Mail',
  // Fallback table names tried (in order) if `messageTable` is absent from the live schema.
  messageTableAliases: ['Messages', 'Message', 'Mail_Messages', 'ZMESSAGE', 'Records'],
  columns: {
    id: 'record_id',
    subject: 'subject',
    sender: 'sender',
    recipient: 'recipient',
    preview: 'preview',
    // Stored as unix SECONDS in this assumption; convert to ms in the parser. A real install may
    // use a different epoch/unit — that's the single most likely thing to need correcting.
    date: 'received_date',
    folder: 'folder',
    // 1 = read, 0/null = unread (we surface `unread`).
    read: 'read',
    messageId: 'message_id',
  },
  // Per-field alias fallbacks tried (in order) if the configured column is absent. Keep these
  // broad — they only ever match a column the live schema actually has.
  columnAliases: {
    id: ['ROWID', 'Record_RecordID', 'Message_RecordID', 'id', 'Z_PK', 'OID'],
    subject: ['Subject', 'Message_Subject', 'normalizedSubject', 'ZSUBJECT', 'title'],
    sender: ['from', 'fromAddress', 'Message_SenderList', 'senderAddress', 'ZSENDER', 'sender_address', 'displayFrom'],
    recipient: ['to', 'toAddress', 'Message_ToRecipientList', 'recipients', 'ZTORECIPIENTS', 'displayTo'],
    preview: ['Preview', 'summary', 'Message_Preview', 'snippet', 'ZPREVIEW', 'bodyPreview', 'body'],
    date: ['date', 'Message_TimeReceived', 'time_received', 'timeReceived', 'sentTime', 'ZRECEIVEDTIME', 'modificationDate'],
    folder: ['Folder', 'folderName', 'Message_FolderID', 'mailbox', 'ZFOLDER'],
    read: ['isRead', 'read_flag', 'Message_IsRead', 'seen', 'ZREAD', 'flagStatus'],
    messageId: ['MessageID', 'internetMessageId', 'Message_MessageID', 'rfc822_id', 'ZMESSAGEID', 'globalMessageId'],
  },
}

// ---------------------------------------------------------------------------------------------
// DETECTION — defensive, "absent → off" (mirrors imessage's missing-DB handling).

// Default profiles root. The new Outlook keeps profiles under the Office Group Container; the
// profile NAME varies ("Main Profile", a UUID, a custom name), so callers glob this for the DB.
export function outlookProfileGlobRoot(home = os.homedir()): string {
  return path.join(
    home,
    'Library', 'Group Containers', 'UBF8T346G9.Office',
    'Outlook', 'Outlook 15 Profiles',
  )
}

// Resolve the Outlook SQLite store under a profiles root by globbing `<root>/*/Data/Outlook.sqlite`
// (profile dir name is arbitrary). Returns the first existing match, or null when the root is
// missing / has no DB. Never throws — an unreadable/absent root simply means "Outlook-local off".
export async function findOutlookDbPath(profilesRoot = outlookProfileGlobRoot()): Promise<string | null> {
  let entries: import('node:fs').Dirent[]
  try {
    entries = await fs.readdir(profilesRoot, { withFileTypes: true })
  } catch {
    return null // no profiles root -> not installed / no local profile
  }
  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    if (!entry.isDirectory()) continue
    // Outlook.sqlite typically lives at <profile>/Data/Outlook.sqlite.
    const candidate = path.join(profilesRoot, entry.name, 'Data', 'Outlook.sqlite')
    if (await fileExists(candidate)) return candidate
    // A couple of layouts seen in the wild keep it directly under the profile dir.
    const flat = path.join(profilesRoot, entry.name, 'Outlook.sqlite')
    if (await fileExists(flat)) return flat
  }
  return null
}

// THE GATE: is there a readable local Outlook store on this machine? Resolves to its path when
// the connector should activate, else null. macOS-only (the path layout is mac-specific).
export async function outlookDbPath(home = os.homedir()): Promise<string | null> {
  if (process.platform !== 'darwin') return null
  return findOutlookDbPath(outlookProfileGlobRoot(home))
}

export function isOutlookSource(source: Pick<EmailSourceConfig, 'kind'>): boolean {
  return source.kind === 'outlook_local'
}

// Detection state for the UI, parallel to imessageAccessState. 'ok' carries the resolved path.
export type OutlookAccessState = 'ok' | 'no_macos' | 'missing_db' | 'error'
export async function outlookAccessState(home = os.homedir()): Promise<{ state: OutlookAccessState; detail: string; path?: string }> {
  if (process.platform !== 'darwin') return { state: 'no_macos', detail: 'Outlook local store is only available on macOS.' }
  let dbPath: string | null
  try {
    dbPath = await findOutlookDbPath(outlookProfileGlobRoot(home))
  } catch (e: any) {
    return { state: 'error', detail: String(e?.message ?? e).slice(0, 200) }
  }
  if (!dbPath) return { state: 'missing_db', detail: 'No local Outlook profile database was found.' }
  try {
    // Trivial read-only probe: proves the file is a readable SQLite DB (and that we have access).
    await exec(SQLITE3, ['-readonly', dbPath, 'SELECT 1;'])
    return { state: 'ok', detail: 'Outlook local store ready.', path: dbPath }
  } catch (e: any) {
    const msg = String(e?.stderr || e?.message || e)
    return { state: 'error', detail: msg.replace(/\s+/g, ' ').slice(0, 200) }
  }
}

// Build (or refresh) an Outlook-local email source pointing at the detected DB. Returns null when
// no local store exists, so the runtime can keep the connector OFF until Outlook is installed.
export async function makeOutlookSource(now = Date.now(), home = os.homedir()): Promise<EmailSourceConfig | null> {
  const dbPath = await outlookDbPath(home)
  if (!dbPath) return null
  return {
    id: 'outlook_local',
    label: 'Outlook (local)',
    path: dbPath,
    kind: 'outlook_local',
    enabled: true,
    createdAt: now,
    updatedAt: now,
    lastSyncAt: null,
    lastError: null,
  }
}

// GATED AUTO-PICKUP. Given the user's configured email sources, return the list to actually scan:
// if a local Outlook store EXISTS and no `outlook_local` source is already configured, prepend an
// ephemeral one so its messages flow into the feed automatically — but ONLY when the DB is present
// (the gate). When Outlook isn't installed, or the user already added/curated their own Outlook
// source, the input list is returned unchanged. Never throws; detection failure = no auto-source.
export async function withDetectedOutlookSource<T extends EmailSourceConfig>(sources: T[], now = Date.now(), home = os.homedir()): Promise<Array<T | EmailSourceConfig>> {
  if (sources.some(isOutlookSource)) return sources // user already has an Outlook source — respect it
  let auto: EmailSourceConfig | null = null
  try {
    auto = await makeOutlookSource(now, home)
  } catch {
    auto = null
  }
  return auto ? [auto, ...sources] : sources
}

// ---------------------------------------------------------------------------------------------
// RUNTIME SCHEMA INTROSPECTION — list tables/columns and map each logical field defensively.

interface ResolvedSchema {
  table: string
  // logical field -> actual column name present in the live DB (subset; required ones guaranteed).
  cols: Partial<Record<LogicalColumn, string>>
}

async function listTables(dbPath: string): Promise<string[]> {
  const raw = await exec(SQLITE3, ['-readonly', '-json', dbPath, "SELECT name FROM sqlite_master WHERE type='table';"])
  const rows: Array<{ name?: unknown }> = raw.trim() ? JSON.parse(raw) : []
  return rows.map((r) => String(r.name ?? '')).filter(Boolean)
}

async function listColumns(dbPath: string, table: string): Promise<string[]> {
  // PRAGMA table_info via a SELECT wrapper so `-json` returns clean rows. quote() the identifier.
  const raw = await exec(SQLITE3, ['-readonly', '-json', dbPath, `PRAGMA table_info(${quoteIdent(table)});`])
  const rows: Array<{ name?: unknown }> = raw.trim() ? JSON.parse(raw) : []
  return rows.map((r) => String(r.name ?? '')).filter(Boolean)
}

// Resolve the message table + each logical column against the live schema, preferring the
// configured name then the alias list. Returns null when no plausible message table/id column
// exists (e.g. the file is some unrelated SQLite DB) — caller then surfaces nothing.
export async function introspectOutlook(dbPath: string): Promise<ResolvedSchema | null> {
  let tables: string[]
  try { tables = await listTables(dbPath) } catch { return null }
  if (!tables.length) return null
  const lc = (s: string) => s.toLowerCase()
  const tableSet = new Map(tables.map((t) => [lc(t), t]))
  const tableCandidates = [OUTLOOK_SCHEMA.messageTable, ...OUTLOOK_SCHEMA.messageTableAliases]
  let table: string | null = null
  for (const cand of tableCandidates) {
    const hit = tableSet.get(lc(cand))
    if (hit) { table = hit; break }
  }
  if (!table) return null

  let columns: string[]
  try { columns = await listColumns(dbPath, table) } catch { return null }
  if (!columns.length) return null
  const colSet = new Map(columns.map((c) => [lc(c), c]))
  const resolve = (logical: LogicalColumn): string | undefined => {
    const order = [OUTLOOK_SCHEMA.columns[logical], ...(OUTLOOK_SCHEMA.columnAliases[logical] ?? [])]
    for (const name of order) {
      const hit = colSet.get(lc(name))
      if (hit) return hit
    }
    return undefined
  }
  const cols: Partial<Record<LogicalColumn, string>> = {}
  for (const logical of Object.keys(OUTLOOK_SCHEMA.columns) as LogicalColumn[]) {
    const found = resolve(logical)
    if (found) cols[logical] = found
  }
  // We need at least an id column to produce stable message ids; without it there's nothing usable.
  if (!cols.id) return null
  return { table, cols }
}

// ---------------------------------------------------------------------------------------------
// SCAN — map rows into the SAME EmailMessage shape the email feed already flows through, so
// Outlook-local messages travel the existing email event path (no new event type).

export async function scanOutlookMessages(source: EmailSourceConfig, opts: EmailListOptions = {}): Promise<EmailMessage[]> {
  // ABSENT/UNREADABLE DB -> [] (degrade quietly, exactly like imessage's missing-DB path). This is
  // what keeps the connector inert when Outlook isn't installed or the store path is stale.
  if (!source.path || !(await fileExists(source.path))) return []
  let schema: ResolvedSchema | null
  try {
    schema = await introspectOutlook(source.path)
  } catch {
    return []
  }
  if (!schema) return []

  const limit = Math.max(1, Math.min(opts.limit ?? 300, 2_000))
  // Alias each resolved actual column to a canonical key so rowToOutlookEmail is schema-agnostic.
  const sel: string[] = []
  for (const logical of Object.keys(OUTLOOK_SCHEMA.columns) as LogicalColumn[]) {
    const actual = schema.cols[logical]
    if (actual) sel.push(`${quoteIdent(actual)} AS _${logical}`)
  }
  const dateCol = schema.cols.date
  const orderBy = dateCol ? `ORDER BY ${quoteIdent(dateCol)} DESC` : `ORDER BY ${quoteIdent(schema.cols.id as string)} DESC`
  // Over-fetch a little then sort/slice in JS (date columns can be null/garbage; JS sort is the
  // source of truth for newest-first, matching scanEmailMessages).
  const sql = `SELECT ${sel.join(', ')} FROM ${quoteIdent(schema.table)} ${orderBy} LIMIT ${limit * 2};`

  let raw: string
  try {
    raw = await exec(SQLITE3, ['-readonly', '-json', source.path, sql])
  } catch {
    return [] // a malformed/locked DB degrades to nothing rather than throwing into the feed
  }
  const rows: any[] = raw.trim() ? JSON.parse(raw) : []
  const messages = rows
    .map((row) => rowToOutlookEmail(source, row))
    .filter((m): m is EmailMessage => !!m)
    .sort((a, b) => (b.date ?? 0) - (a.date ?? 0))
  return messages.slice(0, limit)
}

// Pure mapping of one canonical-aliased row into EmailMessage, with defensive fallbacks. Exported
// so the parser is unit-tested independently of sqlite. A row with no id is unusable -> null.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function rowToOutlookEmail(source: Pick<EmailSourceConfig, 'id' | 'label' | 'path'>, row: any): EmailMessage | null {
  if (!row) return null
  const rawId = row._id
  if (rawId == null || String(rawId).trim() === '') return null
  const id = `${source.id}:${String(rawId)}`
  return {
    id,
    sourceId: source.id,
    sourceLabel: source.label,
    // Provenance points at the local store file (not an imap:// URL), so source refs read sensibly.
    path: source.path,
    subject: str(row._subject),
    from: str(row._sender),
    to: str(row._recipient),
    date: toMs(row._date),
    unread: isUnread(row._read),
    snippet: str(row._preview).replace(/\s+/g, ' ').trim().slice(0, 280),
    size: 0,
    messageId: row._messageId == null ? null : (str(row._messageId) || null),
    // Outlook's local store does not expose a clean In-Reply-To for the lightweight scan; leave
    // null so reply-threading falls back gracefully (same as a file source with no In-Reply-To).
    inReplyTo: null,
  }
}

// Promote a scanned Outlook message to an EmailReadResult. The local-store scan is already
// "full" for our purposes (the preview is the only body text we surface), so the body is the
// snippet and headers carry the message-id; this lets the existing readEmailMessage/readEmailThread
// paths return Outlook messages without a separate per-message DB round-trip.
export function outlookReadResult(message: EmailMessage): EmailReadResult {
  return {
    ...message,
    headers: {
      ...(message.messageId ? { 'message-id': message.messageId } : {}),
      ...(message.subject ? { subject: message.subject } : {}),
      ...(message.from ? { from: message.from } : {}),
      ...(message.to ? { to: message.to } : {}),
    },
    body: message.snippet,
  }
}

// ---------------------------------------------------------------------------------------------
// helpers

function str(v: unknown): string {
  return v == null ? '' : String(v)
}

// read flag -> unread. Treat 1/'1'/true as READ; 0/null/'' as unread (Outlook stores read=1).
function isUnread(read: unknown): boolean {
  if (read == null || read === '') return true
  return !(read === 1 || read === '1' || read === true)
}

// Assumed unit is unix SECONDS. Convert to ms epoch; reject non-finite/<=0 as null. If a real
// install turns out to store ms already (value implausibly large for seconds, i.e. year > ~5000),
// pass it through unscaled so we never produce a year-50000 timestamp.
function toMs(value: unknown): number | null {
  const n = typeof value === 'number' ? value : Number(value)
  if (!Number.isFinite(n) || n <= 0) return null
  // > 1e12 (~ year 33658 in seconds) almost certainly already milliseconds.
  return n > 1e12 ? Math.round(n) : Math.round(n * 1000)
}

async function fileExists(p: string): Promise<boolean> {
  try {
    const st = await fs.stat(p)
    return st.isFile()
  } catch {
    return false
  }
}

// Quote a SQL identifier (table/column) by wrapping in double quotes and doubling embedded ones.
// All identifiers passed here come from the live schema (introspected) or the OUTLOOK_SCHEMA map,
// never from user input, but quoting keeps odd-but-real column names (spaces, reserved words) safe.
function quoteIdent(name: string): string {
  return '"' + String(name).replace(/"/g, '""') + '"'
}
