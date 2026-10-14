import { readdir, readFile, stat } from 'node:fs/promises'
import path from 'node:path'

// User-authored file-based slash commands / recipes (pure loader).
//
// A command is a single markdown file in a commands directory (default `.vontra/commands/*.md`).
// Optional YAML front matter supplies metadata; the markdown BODY (or a `prompt:` front-matter
// field) is the prompt template. At invoke time the template's argument placeholders are expanded
// against the caller-supplied argument string.
//
// This module is intentionally pure/dependency-injected: all filesystem access is injected via
// `CommandLoaderIO` so it is unit-testable without touching disk, mirroring the repo's DI pattern
// (ssh-connection clientFactory, connector-retry fetchImpl, pr-publish runner). It is NOT wired
// into the loop, tools, or IPC — the parent applies that wiring.
//
// Placeholder grammar (git/Claude-style), expanded in expandCommandArgs():
//   $ARGUMENTS   -> the entire raw argument string (verbatim, trimmed of outer whitespace)
//   $1 .. $N     -> positional args (shell-style whitespace split, quotes honoured)
//   ${1} ..      -> same, brace form (lets you write `$1x` vs `${1}x`)
//   $NAME/${NAME}-> named args declared in front-matter `args:` (case-insensitive), falling back
//                   to positional order when the caller passes bare positional values
//   $$           -> a literal '$'
// Unknown placeholders are left untouched (so a prompt can legitimately contain `$foo`).

export const DEFAULT_COMMANDS_DIR = '.vontra/commands'
export const DEFAULT_COMMAND_MAX_BYTES = 256 * 1024

export interface CommandArgSpec {
  /** Declared argument name (lower-cased, safe identifier). */
  name: string
  /** Optional human description shown in UI/help. */
  description?: string
  /** Whether the arg must be supplied for the command to run. */
  required: boolean
}

export interface CommandSpec {
  /** Stable id derived from the file name (slug), used to look the command up by `/id`. */
  id: string
  /** Display name (front-matter `name:` or humanised file name). */
  name: string
  /** Optional one-line description. */
  description: string
  /** The raw prompt template (front-matter `prompt:` if present, else the markdown body). */
  template: string
  /** Declared positional/named arguments, in order. */
  args: CommandArgSpec[]
  /** Absolute path to the source file. */
  sourcePath: string
  /** File mtime (ms), for cache-busting / sort-by-recent. */
  updatedAt: number
}

export interface LoadCommandsOptions {
  /** Cap on a single command file's size; larger files are skipped. */
  maxFileBytes?: number
}

export interface CommandLoaderIO {
  readdir(dir: string): Promise<string[]>
  readFile(filePath: string): Promise<string>
  stat(filePath: string): Promise<{ size: number; mtimeMs: number; isFile: boolean }>
}

/** Default IO backed by node:fs/promises. Inject a fake in tests. */
export const nodeCommandLoaderIO: CommandLoaderIO = {
  async readdir(dir: string) {
    return await readdir(dir)
  },
  async readFile(filePath: string) {
    return await readFile(filePath, 'utf8')
  },
  async stat(filePath: string) {
    const s = await stat(filePath)
    return { size: s.size, mtimeMs: s.mtimeMs, isFile: s.isFile() }
  },
}

export interface LoadCommandsResult {
  commands: CommandSpec[]
  /** Non-fatal per-file problems (unreadable, too large, duplicate id, invalid). */
  errors: Array<{ file: string; error: string }>
}

/**
 * Load every `*.md` command file from `dir`. Never throws for a missing dir or a broken file —
 * a missing dir yields an empty result and per-file problems land in `errors`. Commands are
 * returned sorted by id; on duplicate ids the first (alphabetically) wins and the rest are errors.
 */
export async function loadCommands(
  dir: string,
  io: CommandLoaderIO = nodeCommandLoaderIO,
  options: LoadCommandsOptions = {},
): Promise<LoadCommandsResult> {
  const maxBytes = normalizedMaxBytes(options.maxFileBytes)
  const errors: LoadCommandsResult['errors'] = []

  let names: string[]
  try {
    names = await io.readdir(dir)
  } catch {
    return { commands: [], errors }
  }

  const mdFiles = names
    .filter((n) => n.toLowerCase().endsWith('.md'))
    .sort((a, b) => a.localeCompare(b))

  const byId = new Map<string, CommandSpec>()

  for (const fileName of mdFiles) {
    const filePath = path.join(dir, fileName)
    let info: { size: number; mtimeMs: number; isFile: boolean }
    try {
      info = await io.stat(filePath)
    } catch (e: any) {
      errors.push({ file: fileName, error: `Could not stat: ${e?.message ?? e}` })
      continue
    }
    if (!info.isFile) continue
    if (info.size > maxBytes) {
      errors.push({ file: fileName, error: `Command file exceeds ${maxBytes} bytes; skipped.` })
      continue
    }

    let text: string
    try {
      text = await io.readFile(filePath)
    } catch (e: any) {
      errors.push({ file: fileName, error: `Could not read: ${e?.message ?? e}` })
      continue
    }

    const parsed = parseCommandDocument(text, fileName)
    if (!parsed.template.trim() && !parsed.description) {
      errors.push({ file: fileName, error: 'Command has no prompt body and no description; skipped.' })
      continue
    }

    const spec: CommandSpec = {
      ...parsed,
      sourcePath: filePath,
      updatedAt: Math.floor(info.mtimeMs),
    }

    if (byId.has(spec.id)) {
      errors.push({ file: fileName, error: `Duplicate command id "${spec.id}" (already defined by an earlier file); skipped.` })
      continue
    }
    byId.set(spec.id, spec)
  }

  const commands = [...byId.values()].sort((a, b) => a.id.localeCompare(b.id))
  return { commands, errors }
}

/**
 * Parse a single command markdown document into a CommandSpec (minus filesystem metadata).
 * Pure — safe to call on in-memory strings.
 */
export function parseCommandDocument(text: string, fileName = 'command.md'): Omit<CommandSpec, 'sourcePath' | 'updatedAt'> {
  const normalized = text.replace(/^﻿/, '').replace(/\r\n?/g, '\n')
  const { frontMatter, body } = splitFrontMatter(normalized)
  const fields = frontMatter ? parseYamlishFields(frontMatter) : {}

  const baseName = fileName.replace(/\.md$/i, '')
  const id = slugify(baseName)
  const name = compact(fields.name ?? humanize(baseName), 120) || 'Untitled Command'
  const description = compact(fields.description ?? '', 400)
  const template = (fields.prompt ?? body).replace(/^\n+/, '').replace(/\s+$/, '')
  const args = parseArgSpecs(fields.args)

  return { id, name, description, template, args }
}

/**
 * Expand a command template against a raw argument string, producing the final prompt.
 * The declared `args` order is used to name positional inputs.
 */
export function expandCommandArgs(spec: Pick<CommandSpec, 'template' | 'args'>, rawArgs: string): { prompt: string; missing: string[] } {
  const raw = (rawArgs ?? '').trim()
  const positional = tokenizeArgs(raw)

  // Build a name -> value map from declared args (positional order) so $NAME resolves.
  const named = new Map<string, string>()
  spec.args.forEach((arg, index) => {
    if (index < positional.length) named.set(arg.name.toLowerCase(), positional[index])
  })

  const missing = spec.args
    .filter((arg, index) => arg.required && index >= positional.length)
    .map((arg) => arg.name)

  const prompt = spec.template.replace(
    /\$\$|\$\{([A-Za-z0-9_]+)\}|\$([A-Za-z0-9_]+)/g,
    (match, braced: string | undefined, bare: string | undefined) => {
      if (match === '$$') return '$'
      const key = (braced ?? bare) as string
      if (/^ARGUMENTS$/i.test(key)) return raw
      const posMatch = /^(\d+)$/.exec(key)
      if (posMatch) {
        const idx = Number(posMatch[1]) - 1
        return idx >= 0 && idx < positional.length ? positional[idx] : ''
      }
      const namedVal = named.get(key.toLowerCase())
      if (namedVal !== undefined) return namedVal
      // Unknown placeholder: leave verbatim so prompts can contain literal `$foo`.
      return match
    },
  )

  return { prompt, missing }
}

// ---- argument tokenizer (shell-lite: whitespace split honouring "double" and 'single' quotes) ----

export function tokenizeArgs(raw: string): string[] {
  const tokens: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let hasToken = false

  for (let i = 0; i < raw.length; i++) {
    const ch = raw[i]
    if (quote) {
      if (ch === quote) {
        quote = null
      } else if (ch === '\\' && quote === '"' && i + 1 < raw.length) {
        current += raw[++i]
      } else {
        current += ch
      }
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      hasToken = true
      continue
    }
    if (/\s/.test(ch)) {
      if (hasToken) {
        tokens.push(current)
        current = ''
        hasToken = false
      }
      continue
    }
    current += ch
    hasToken = true
  }
  if (hasToken) tokens.push(current)
  return tokens
}

// ---- front-matter + yamlish parsing (self-contained; mirrors agent-skills.ts approach) ----

function splitFrontMatter(text: string): { frontMatter: string | null; body: string } {
  const lines = text.split('\n')
  if (lines[0]?.trim() !== '---') return { frontMatter: null, body: text }
  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '---') {
      return { frontMatter: lines.slice(1, i).join('\n'), body: lines.slice(i + 1).join('\n') }
    }
  }
  return { frontMatter: null, body: text }
}

type CommandFields = Partial<Record<'name' | 'description' | 'prompt', string>> & { args?: unknown }

function parseYamlishFields(text: string): CommandFields {
  const fields: CommandFields = {}
  const lines = text.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const match = /^([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$/.exec(lines[i])
    if (!match) continue
    const key = match[1].toLowerCase()
    let value = match[2].trim()

    // Block scalars (| literal, > folded) for multi-line prompt bodies.
    if (value === '|' || value === '>') {
      const folded = value === '>'
      const block: string[] = []
      for (let j = i + 1; j < lines.length; j++) {
        if (/^[A-Za-z][A-Za-z0-9_-]*\s*:/.test(lines[j]) && !/^\s/.test(lines[j])) break
        block.push(lines[j].replace(/^\s{1,4}/, ''))
        i = j
      }
      value = folded ? block.join(' ') : block.join('\n')
      if (key === 'name' || key === 'description' || key === 'prompt') fields[key] = value
      continue
    }

    // Inline / nested list for `args:`.
    if (key === 'args') {
      // Inline JSON-ish array on the same line, e.g. args: [file, mode]
      if (value.startsWith('[')) {
        fields.args = parseInlineList(value)
        continue
      }
      // Nested block list on following lines. Each item is a `- ...` line optionally followed
      // by indented `key: value` continuation lines (YAML block-mapping form):
      //   - name: target
      //     required: true
      //     description: the file to edit
      const items: unknown[] = []
      let current: string[] | null = null
      const flush = () => {
        if (current) { items.push(parseArgItem(current)); current = null }
      }
      for (let j = i + 1; j < lines.length; j++) {
        const line = lines[j]
        if (line.trim() === '') { i = j; continue }
        const listMatch = /^\s*-\s+(.*)$/.exec(line)
        if (listMatch) {
          flush()
          current = [listMatch[1].trim()]
          i = j
          continue
        }
        // A dedented (non-indented) line ends the args block.
        if (!/^\s/.test(line)) break
        // Indented continuation of the current item (e.g. `    required: true`).
        if (current) { current.push(line.trim()); i = j; continue }
        break
      }
      flush()
      if (items.length) fields.args = items
      continue
    }

    if (key === 'name' || key === 'description' || key === 'prompt') {
      fields[key] = stripQuotes(value)
    }
  }

  return fields
}

function parseArgItem(itemLines: string[]): unknown {
  // itemLines[0] is the `- ...` content; the rest are indented `key: value` continuation lines.
  const [firstRaw, ...rest] = itemLines
  const first = firstRaw.trim()

  // Bare scalar form: `- file` (no colon, no continuation lines).
  const flowMap = /^\{(.*)\}$/.exec(first)
  const hasKvOnFirst = /^[A-Za-z0-9_-]+\s*:/.test(first) || !!flowMap
  if (!hasKvOnFirst && rest.length === 0) return stripQuotes(first)

  const obj: Record<string, string> = {}
  const applyKv = (segment: string) => {
    const kv = /^([A-Za-z0-9_-]+)\s*:\s*(.*)$/.exec(segment.trim())
    if (kv) obj[kv[1].toLowerCase()] = stripQuotes(kv[2].trim())
  }
  // Inline flow map `- {name: x, required: true}` or comma-separated `- name: x, required: true`.
  for (const part of splitTopLevel(flowMap ? flowMap[1] : first)) applyKv(part)
  for (const line of rest) applyKv(line)
  return obj
}

function parseInlineList(value: string): string[] {
  const inner = value.replace(/^\[/, '').replace(/\]$/, '')
  return splitTopLevel(inner).map((s) => stripQuotes(s.trim())).filter(Boolean)
}

function splitTopLevel(src: string): string[] {
  const out: string[] = []
  let cur = ''
  let quote: string | null = null
  for (const ch of src) {
    if (quote) {
      cur += ch
      if (ch === quote) quote = null
      continue
    }
    if (ch === '"' || ch === "'") { quote = ch; cur += ch; continue }
    if (ch === ',') { out.push(cur); cur = ''; continue }
    cur += ch
  }
  if (cur.trim()) out.push(cur)
  return out
}

function parseArgSpecs(rawArgs: unknown): CommandArgSpec[] {
  if (!Array.isArray(rawArgs)) return []
  const specs: CommandArgSpec[] = []
  const seen = new Set<string>()
  for (const item of rawArgs) {
    let name: string | undefined
    let description: string | undefined
    let required = false
    if (typeof item === 'string') {
      name = item
    } else if (item && typeof item === 'object') {
      const rec = item as Record<string, unknown>
      name = typeof rec.name === 'string' ? rec.name : undefined
      description = typeof rec.description === 'string' ? rec.description : undefined
      required = rec.required === true || rec.required === 'true'
    }
    const clean = sanitizeArgName(name)
    if (!clean || seen.has(clean)) continue
    seen.add(clean)
    specs.push({ name: clean, description, required })
  }
  return specs
}

function sanitizeArgName(name: string | undefined): string {
  if (!name) return ''
  return name.trim().toLowerCase().replace(/[^a-z0-9_]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 48)
}

function stripQuotes(value: string): string {
  const t = value.trim()
  if (t.length >= 2 && t.startsWith('"') && t.endsWith('"')) return t.slice(1, -1).replace(/\\"/g, '"')
  if (t.length >= 2 && t.startsWith("'") && t.endsWith("'")) return t.slice(1, -1).replace(/''/g, "'")
  return t
}

function slugify(name: string): string {
  const id = name.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 96)
  return id || 'command'
}

function humanize(name: string): string {
  const cleaned = name.replace(/[._-]+/g, ' ').replace(/\s+/g, ' ').trim()
  return cleaned ? cleaned.replace(/\b[a-z]/g, (c) => c.toUpperCase()) : 'Untitled Command'
}

function compact(text: string, limit: number): string {
  const c = text.replace(/\s+/g, ' ').trim()
  if (c.length <= limit) return c
  return c.slice(0, Math.max(0, limit - 3)).trimEnd() + '...'
}

function normalizedMaxBytes(input: number | undefined): number {
  if (typeof input !== 'number' || !Number.isFinite(input)) return DEFAULT_COMMAND_MAX_BYTES
  return Math.max(0, Math.floor(input))
}
