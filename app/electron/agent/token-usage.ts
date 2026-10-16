import os from 'node:os'
import path from 'node:path'
import { promises as fs } from 'node:fs'
import { execFile } from 'node:child_process'
import { promisify } from 'node:util'
import type {
  TokenUsageAccount,
  TokenUsageAccountId,
  TokenUsageBreakdown,
  TokenUsageModel,
  TokenUsagePoint,
  TokenUsageSnapshot,
  TokenUsageWindow,
  TokenUsageWindowKey,
} from './types'

const execFileAsync = promisify(execFile)

const DAILY_DAYS = 45
const HOURLY_HOURS = 50
const ACCOUNTS: Array<{ id: TokenUsageAccountId; name: string }> = [
  { id: 'claude', name: 'Claude' },
  { id: 'codex', name: 'Codex' },
  { id: 'opencode', name: 'OpenCode' },
]
const WINDOW_ORDER: Array<{ key: TokenUsageWindowKey; label: string; start: (now: Date) => number }> = [
  { key: 'five_hour', label: '5h', start: (now) => now.getTime() - 5 * 60 * 60_000 },
  { key: 'today', label: 'Today', start: (now) => new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime() },
  {
    key: 'week',
    label: 'Week',
    start: (now) => {
      const midnight = new Date(now.getFullYear(), now.getMonth(), now.getDate())
      const mondayOffset = (midnight.getDay() + 6) % 7
      return midnight.getTime() - mondayOffset * 24 * 60 * 60_000
    },
  },
  { key: 'month', label: 'Month', start: (now) => new Date(now.getFullYear(), now.getMonth(), 1).getTime() },
  { key: 'all', label: 'All', start: () => 0 },
]

interface UsageEvent {
  ts: number
  account: TokenUsageAccountId
  model: string
  tokens: TokenUsageBreakdown
}

interface ClaudeCachedEvent {
  key: string
  ts: number
  model: string
  input: number
  output: number
  cacheCreation: number
  cacheRead: number
}

interface ClaudeCache {
  v: 1
  files: Record<string, { mtimeMs: number; size: number; events: ClaudeCachedEvent[] }>
}

export async function collectTokenUsage(cacheDir: string): Promise<TokenUsageSnapshot> {
  const now = new Date()
  const events = [
    ...(await claudeEvents(cacheDir)),
    ...(await codexEvents()),
    ...(await opencodeEvents()),
  ]
  return aggregateEvents(events, now)
}

function emptyCounts(): TokenUsageBreakdown {
  return { total: 0, input: 0, output: 0, cacheCreation: 0, cacheRead: 0, cached: 0, reasoning: 0 }
}

function normalizeCounts(p: Partial<TokenUsageBreakdown>): TokenUsageBreakdown {
  const out = emptyCounts()
  for (const key of Object.keys(out) as Array<keyof TokenUsageBreakdown>) {
    out[key] = Math.max(0, Number(p[key] ?? 0) || 0)
  }
  if (!out.total) out.total = out.input + out.output + out.cacheCreation + out.cacheRead + out.cached + out.reasoning
  return out
}

function addCounts(target: TokenUsageBreakdown, next: Partial<TokenUsageBreakdown>) {
  target.total += Math.max(0, Number(next.total ?? 0) || 0)
  target.input += Math.max(0, Number(next.input ?? 0) || 0)
  target.output += Math.max(0, Number(next.output ?? 0) || 0)
  target.cacheCreation += Math.max(0, Number(next.cacheCreation ?? 0) || 0)
  target.cacheRead += Math.max(0, Number(next.cacheRead ?? 0) || 0)
  target.cached += Math.max(0, Number(next.cached ?? 0) || 0)
  target.reasoning += Math.max(0, Number(next.reasoning ?? 0) || 0)
}

function aggregateEvents(events: UsageEvent[], now: Date): TokenUsageSnapshot {
  const starts = Object.fromEntries(WINDOW_ORDER.map((w) => [w.key, w.start(now)])) as Record<TokenUsageWindowKey, number>
  const windowMaps = Object.fromEntries(WINDOW_ORDER.map((w) => [
    w.key,
    { accounts: new Map<TokenUsageAccountId, TokenUsageAccount>(), models: new Map<string, TokenUsageModel>(), total: 0 },
  ])) as Record<TokenUsageWindowKey, { accounts: Map<TokenUsageAccountId, TokenUsageAccount>; models: Map<string, TokenUsageModel>; total: number }>
  const daily = new Map<string, TokenUsagePoint>()
  const hourly = new Map<number, TokenUsagePoint>()
  let latestAt: number | null = null
  let active = false
  const fiveMinutesAgo = now.getTime() - 5 * 60_000

  for (const event of events) {
    if (!Number.isFinite(event.ts) || event.tokens.total <= 0) continue
    if (latestAt == null || event.ts > latestAt) latestAt = event.ts
    if (event.ts >= fiveMinutesAgo) active = true
    for (const { key } of WINDOW_ORDER) {
      if (event.ts < starts[key]) continue
      const bucket = windowMaps[key]
      bucket.total += event.tokens.total
      const account = getAccount(bucket.accounts, event.account)
      account.latestAt = account.latestAt == null ? event.ts : Math.max(account.latestAt, event.ts)
      addCounts(account, event.tokens)
      const accountModel = getModel(new Map(account.models.map((m) => [m.model, m])), event.model)
      if (!account.models.find((m) => m.model === accountModel.model)) account.models.push(accountModel)
      addCounts(accountModel, event.tokens)
      const model = getModel(bucket.models, event.model)
      addCounts(model, event.tokens)
    }

    const dayKey = new Date(event.ts).toISOString().slice(0, 10)
    const day = getPoint(daily, dayKey, new Date(dayKey + 'T00:00:00.000Z').getTime())
    day.total += event.tokens.total
    day.accounts[event.account] += event.tokens.total

    const hourTs = Math.floor(event.ts / 3_600_000) * 3_600_000
    const hour = getPoint(hourly, hourTs, hourTs)
    hour.total += event.tokens.total
    hour.accounts[event.account] += event.tokens.total
  }

  const windows = Object.fromEntries(WINDOW_ORDER.map(({ key, label }) => {
    const bucket = windowMaps[key]
    const accounts = [...bucket.accounts.values()].map((account) => ({
      ...account,
      models: [...account.models].sort(byTotal),
    })).sort(byTotal)
    const models = [...bucket.models.values()].sort(byTotal)
    return [key, { key, label, startAt: starts[key], total: bucket.total, accounts, models } satisfies TokenUsageWindow]
  })) as Record<TokenUsageWindowKey, TokenUsageWindow>

  const minDay = now.getTime() - (DAILY_DAYS - 1) * 24 * 60 * 60_000
  const minHour = now.getTime() - (HOURLY_HOURS - 1) * 60 * 60_000

  return {
    generatedAt: now.getTime(),
    latestAt,
    active,
    windows,
    daily: [...daily.values()].filter((p) => p.ts >= minDay).sort((a, b) => a.ts - b.ts),
    hourly: [...hourly.values()].filter((p) => p.ts >= minHour).sort((a, b) => a.ts - b.ts),
  }
}

function byTotal(a: TokenUsageBreakdown, b: TokenUsageBreakdown) {
  return b.total - a.total
}

function getAccount(map: Map<TokenUsageAccountId, TokenUsageAccount>, id: TokenUsageAccountId): TokenUsageAccount {
  const existing = map.get(id)
  if (existing) return existing
  const meta = ACCOUNTS.find((a) => a.id === id)
  const account: TokenUsageAccount = { id, name: meta?.name ?? id, latestAt: null, models: [], ...emptyCounts() }
  map.set(id, account)
  return account
}

function getModel(map: Map<string, TokenUsageModel>, modelName: string): TokenUsageModel {
  const model = modelName || 'unknown'
  const existing = map.get(model)
  if (existing) return existing
  const record: TokenUsageModel = { model, ...emptyCounts() }
  map.set(model, record)
  return record
}

function getPoint(map: Map<string | number, TokenUsagePoint>, key: string | number, ts: number): TokenUsagePoint {
  const existing = map.get(key)
  if (existing) return existing
  const point: TokenUsagePoint = {
    key: String(key),
    ts,
    total: 0,
    accounts: { claude: 0, codex: 0, opencode: 0 },
  }
  map.set(key, point)
  return point
}

async function claudeEvents(cacheDir: string): Promise<UsageEvent[]> {
  const home = os.homedir()
  const root = path.join(home, '.claude', 'projects')
  const files = await collectFiles(root, (file) => file.endsWith('.jsonl'))
  const cachePath = path.join(cacheDir, 'usage-cache-claude.json')
  const old = await readClaudeCache(cachePath)
  const next: ClaudeCache = { v: 1, files: {} }

  for (const file of files) {
    try {
      const st = await fs.stat(file)
      const cached = old.files[file]
      if (cached && cached.mtimeMs === st.mtimeMs && cached.size === st.size) {
        next.files[file] = cached
      } else {
        next.files[file] = { mtimeMs: st.mtimeMs, size: st.size, events: await parseClaudeFile(file) }
      }
    } catch {
      // Ignore files racing with log rotation.
    }
  }

  await writeClaudeCache(cachePath, next)
  const seen = new Set<string>()
  const out: UsageEvent[] = []
  for (const file of Object.values(next.files)) {
    for (const event of file.events) {
      if (seen.has(event.key)) continue
      seen.add(event.key)
      out.push({
        ts: event.ts,
        account: 'claude',
        model: event.model,
        tokens: normalizeCounts({
          input: event.input,
          output: event.output,
          cacheCreation: event.cacheCreation,
          cacheRead: event.cacheRead,
        }),
      })
    }
  }
  return out
}

async function parseClaudeFile(file: string): Promise<ClaudeCachedEvent[]> {
  const events: ClaudeCachedEvent[] = []
  const text = await readText(file)
  for (const line of text.split('\n')) {
    if (!line.includes('"usage"') || !line.includes('"model"')) continue
    try {
      const raw = JSON.parse(line)
      const message = raw?.message ?? {}
      const usage = message?.usage
      const model = String(message?.model ?? '')
      const ts = isoEpoch(raw?.timestamp)
      if (!usage || typeof usage !== 'object' || !model || model === '<synthetic>' || ts == null) continue
      events.push({
        key: `${message?.id ?? ''}:${raw?.requestId ?? ''}`,
        ts,
        model,
        input: number(usage.input_tokens),
        output: number(usage.output_tokens),
        cacheCreation: number(usage.cache_creation_input_tokens),
        cacheRead: number(usage.cache_read_input_tokens),
      })
    } catch {
      // JSONL logs can include partial/corrupt lines while a process is writing.
    }
  }
  return events
}

async function codexEvents(): Promise<UsageEvent[]> {
  const root = path.join(os.homedir(), '.codex', 'sessions')
  const files = await collectFiles(root, (file) => path.basename(file).startsWith('rollout-') && file.endsWith('.jsonl'))
  const out: UsageEvent[] = []
  for (const file of files) {
    // Dedup is scoped PER FILE: two concurrent codex sessions (two rollout files) can each
    // record a token_count at the same millisecond timestamp, and those are distinct events —
    // a global timestamp set would drop the second and undercount. Within a single rollout
    // (one session) a repeated timestamp does mean a duplicated line, so per-file dedup is right.
    const seenTs = new Set<string>()
    // Model is unknown until the first turn_context line parses successfully. A
    // rollout file can be read mid-write, so an early turn_context line may be
    // truncated/corrupt and get silently skipped below; any token_count events
    // seen before a model is known are buffered here and retroactively attributed
    // once a later turn_context in the SAME file/stream does parse successfully,
    // instead of being stamped 'unknown' forever.
    let currentModel: string | null = null
    const pendingUnknownModel: UsageEvent[] = []
    const text = await readText(file)
    for (const line of text.split('\n')) {
      if (!line.includes('"model"') && !line.includes('"token_count"')) continue
      try {
        const raw = JSON.parse(line)
        if (raw?.type === 'turn_context') {
          const model = raw?.payload?.model
          if (model) {
            currentModel = String(model)
            if (pendingUnknownModel.length) {
              for (const pending of pendingUnknownModel) pending.model = currentModel
              pendingUnknownModel.length = 0
            }
          }
          continue
        }
        if (raw?.type !== 'event_msg' || raw?.payload?.type !== 'token_count') continue
        const usage = raw?.payload?.info?.last_token_usage ?? {}
        const total = number(usage.total_tokens)
        const tsRaw = String(raw?.timestamp ?? '')
        const ts = isoEpoch(tsRaw)
        if (!total || !tsRaw || seenTs.has(tsRaw) || ts == null) continue
        seenTs.add(tsRaw)
        const event: UsageEvent = {
          ts,
          account: 'codex',
          model: currentModel ?? 'unknown',
          tokens: normalizeCounts({
            total,
            input: number(usage.input_tokens),
            output: number(usage.output_tokens),
            cached: number(usage.cached_input_tokens),
            reasoning: number(usage.reasoning_output_tokens),
          }),
        }
        out.push(event)
        if (currentModel == null) pendingUnknownModel.push(event)
      } catch {
        // Ignore partial/corrupt JSONL lines.
      }
    }
  }
  return out
}

async function opencodeEvents(): Promise<UsageEvent[]> {
  const db = path.join(os.homedir(), '.local', 'share', 'opencode', 'opencode.db')
  try {
    await fs.access(db)
  } catch {
    return []
  }
  try {
    const query = 'select time_updated, model, tokens_input, tokens_output, tokens_reasoning, tokens_cache_read, tokens_cache_write from session'
    const { stdout } = await execFileAsync('sqlite3', ['-readonly', '-json', db, query], { timeout: 5000, maxBuffer: 8_000_000 })
    const rows = JSON.parse(stdout || '[]') as Array<Record<string, unknown>>
    return rows.flatMap((row) => {
      const ts = number(row.time_updated) // OpenCode stores ms
      const input = number(row.tokens_input)
      const output = number(row.tokens_output)
      const reasoning = number(row.tokens_reasoning)
      const cacheRead = number(row.tokens_cache_read)
      const cacheCreation = number(row.tokens_cache_write)
      const total = input + output + reasoning + cacheRead + cacheCreation
      if (!ts || total <= 0) return []
      return [{
        ts,
        account: 'opencode' as const,
        model: opencodeModel(row.model),
        tokens: normalizeCounts({ total, input, output, reasoning, cacheRead, cacheCreation }),
      }]
    })
  } catch {
    return []
  }
}

function opencodeModel(raw: unknown): string {
  const text = String(raw ?? '').trim()
  if (!text) return 'unknown'
  try {
    const parsed = JSON.parse(text)
    const id = String(parsed?.id ?? text)
    return id.split('/').pop() || 'unknown'
  } catch {
    return text.split('/').pop() || 'unknown'
  }
}

async function readClaudeCache(file: string): Promise<ClaudeCache> {
  try {
    const cache = JSON.parse(await fs.readFile(file, 'utf8'))
    if (cache?.v === 1 && cache?.files && typeof cache.files === 'object') return cache
  } catch {
    // Missing or corrupt cache just triggers a rebuild.
  }
  return { v: 1, files: {} }
}

async function writeClaudeCache(file: string, cache: ClaudeCache) {
  try {
    await fs.mkdir(path.dirname(file), { recursive: true })
    const tmp = file + '.tmp'
    await fs.writeFile(tmp, JSON.stringify(cache), 'utf8')
    await fs.rename(tmp, file)
  } catch {
    // Cache failures should never hide usage data.
  }
}

async function collectFiles(root: string, accept: (file: string) => boolean): Promise<string[]> {
  const out: string[] = []
  const stack = [root]
  while (stack.length) {
    const dir = stack.pop()!
    let entries: Array<{ name: string; isDirectory(): boolean; isFile(): boolean }> = []
    try {
      entries = await fs.readdir(dir, { withFileTypes: true })
    } catch {
      continue
    }
    for (const entry of entries) {
      const full = path.join(dir, entry.name)
      if (entry.isDirectory()) stack.push(full)
      else if (entry.isFile() && accept(full)) out.push(full)
    }
  }
  return out
}

async function readText(file: string): Promise<string> {
  try {
    return await fs.readFile(file, 'utf8')
  } catch {
    return ''
  }
}

function isoEpoch(value: unknown): number | null {
  const ms = Date.parse(String(value ?? ''))
  return Number.isFinite(ms) ? ms : null
}

function number(value: unknown): number {
  const n = Number(value ?? 0)
  return Number.isFinite(n) ? n : 0
}
