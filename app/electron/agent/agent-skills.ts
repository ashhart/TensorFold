import { open, readdir, realpath, stat } from 'node:fs/promises'
import path from 'node:path'

export const DEFAULT_AGENT_SKILL_MAX_BYTES = 256 * 1024

export interface AgentSkillMetadata {
  id: string
  name: string
  description: string
  rootPath: string
  skillPath: string
  hasScripts: boolean
  hasReferences: boolean
  hasAssets: boolean
  updatedAt: number
}

export interface AgentSkillScanOptions {
  maxSkillFileBytes?: number
}

export interface ParsedAgentSkillDocument {
  name: string
  description: string
}

export async function scanAgentSkills(rootPath: string, options: AgentSkillScanOptions = {}): Promise<AgentSkillMetadata[]> {
  if (!rootPath.trim()) return []

  const rootRealPath = await realDirectoryPath(rootPath)
  if (!rootRealPath) return []

  const maxSkillFileBytes = normalizedMaxBytes(options.maxSkillFileBytes)
  const entries = await directoryEntries(rootRealPath)
  const skills: AgentSkillMetadata[] = []
  const seenSkillRoots = new Set<string>()

  for (const entry of entries) {
    if (!entry.isDirectory() && !entry.isSymbolicLink()) continue

    const entryPath = path.join(rootRealPath, entry.name)
    const skillRoot = await realDirectoryPath(entryPath)
    if (!skillRoot || !isStrictChildPath(rootRealPath, skillRoot) || seenSkillRoots.has(skillRoot)) continue
    seenSkillRoots.add(skillRoot)

    const skillPath = await realFilePath(path.join(skillRoot, 'SKILL.md'), skillRoot)
    if (!skillPath) continue

    const text = await readBoundedText(skillPath, maxSkillFileBytes)
    if (text == null) continue

    const skillStat = await fileStat(skillPath)
    if (!skillStat) continue

    const parsed = parseAgentSkillDocument(text, entry.name)
    const [hasScripts, hasReferences, hasAssets] = await Promise.all([
      hasSafeDirectory(skillRoot, 'scripts'),
      hasSafeDirectory(skillRoot, 'references'),
      hasSafeDirectory(skillRoot, 'assets'),
    ])

    skills.push({
      id: skillIdFromFolderName(entry.name),
      name: parsed.name,
      description: parsed.description,
      rootPath: skillRoot,
      skillPath,
      hasScripts,
      hasReferences,
      hasAssets,
      updatedAt: Math.floor(skillStat.mtimeMs),
    })
  }

  return skills
}

export function parseAgentSkillDocument(text: string, folderName = 'skill'): ParsedAgentSkillDocument {
  const normalized = text.replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n')
  const { frontMatter, body } = splitFrontMatter(normalized)
  const fields = frontMatter ? parseYamlishFields(frontMatter) : {}
  const title = firstMarkdownHeading(body)
  const description = firstMarkdownParagraph(body)

  return {
    name: compactPlainText(fields.name ?? title ?? humanizeFolderName(folderName), 120) || 'Untitled Skill',
    description: compactPlainText(fields.description ?? description ?? '', 600),
  }
}

async function directoryEntries(rootPath: string) {
  try {
    return (await readdir(rootPath, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))
  } catch {
    return []
  }
}

async function realDirectoryPath(inputPath: string): Promise<string | null> {
  try {
    const resolved = await realpath(path.resolve(inputPath))
    const resolvedStat = await stat(resolved)
    return resolvedStat.isDirectory() ? resolved : null
  } catch {
    return null
  }
}

async function realFilePath(inputPath: string, safeRoot: string): Promise<string | null> {
  try {
    const resolved = await realpath(path.resolve(inputPath))
    if (!isInsideOrSamePath(safeRoot, resolved)) return null
    const resolvedStat = await stat(resolved)
    return resolvedStat.isFile() ? resolved : null
  } catch {
    return null
  }
}

async function fileStat(inputPath: string) {
  try {
    return await stat(inputPath)
  } catch {
    return null
  }
}

async function hasSafeDirectory(skillRoot: string, name: string): Promise<boolean> {
  const childRoot = await realDirectoryPath(path.join(skillRoot, name))
  return !!childRoot && isStrictChildPath(skillRoot, childRoot)
}

export async function readBoundedText(filePath: string, maxBytes: number): Promise<string | null> {
  let handle: Awaited<ReturnType<typeof open>> | null = null
  try {
    handle = await open(filePath, 'r')
    const openedStat = await handle.stat()
    if (!openedStat.isFile() || openedStat.size > maxBytes) return null
    if (openedStat.size === 0) return ''

    const buffer = Buffer.alloc(openedStat.size)
    const { bytesRead } = await handle.read(buffer, 0, buffer.length, 0)
    return buffer.subarray(0, bytesRead).toString('utf8')
  } catch {
    return null
  } finally {
    await handle?.close().catch(() => undefined)
  }
}

function normalizedMaxBytes(input: number | undefined): number {
  if (typeof input !== 'number' || !Number.isFinite(input)) return DEFAULT_AGENT_SKILL_MAX_BYTES
  return Math.max(0, Math.floor(input))
}

function isInsideOrSamePath(parentPath: string, candidatePath: string): boolean {
  const relative = path.relative(parentPath, candidatePath)
  return relative === '' || (!!relative && !relative.startsWith('..') && !path.isAbsolute(relative))
}

function isStrictChildPath(parentPath: string, candidatePath: string): boolean {
  const relative = path.relative(parentPath, candidatePath)
  return !!relative && !relative.startsWith('..') && !path.isAbsolute(relative)
}

function splitFrontMatter(text: string): { frontMatter: string | null; body: string } {
  const lines = text.split('\n')
  if (lines[0]?.trim() !== '---') return { frontMatter: null, body: text }

  for (let i = 1; i < lines.length; i++) {
    if (lines[i].trim() === '---') {
      return {
        frontMatter: lines.slice(1, i).join('\n'),
        body: lines.slice(i + 1).join('\n'),
      }
    }
  }

  const firstHeading = lines.findIndex((line, index) => index > 0 && /^\s{0,3}#{1,6}\s+/.test(line))
  return { frontMatter: null, body: lines.slice(firstHeading > 0 ? firstHeading : 1).join('\n') }
}

function parseYamlishFields(text: string): Partial<Record<'name' | 'description', string>> {
  const fields: Partial<Record<'name' | 'description', string>> = {}
  const lines = text.split('\n')

  for (let i = 0; i < lines.length; i++) {
    const match = /^([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$/.exec(lines[i])
    if (!match) continue

    const key = match[1].toLowerCase()
    if (key !== 'name' && key !== 'description') continue

    let value = match[2].trim()
    if (value === '|' || value === '>') {
      const folded = value === '>'
      const block: string[] = []
      for (let j = i + 1; j < lines.length; j++) {
        if (/^[A-Za-z][A-Za-z0-9_-]*\s*:/.test(lines[j])) break
        block.push(lines[j].replace(/^\s{1,4}/, ''))
        i = j
      }
      value = folded ? block.join(' ') : block.join('\n')
    }

    fields[key] = stripYamlishQuotes(value)
  }

  return fields
}

function stripYamlishQuotes(value: string): string {
  const trimmed = value.trim()
  if (trimmed.length >= 2 && trimmed.startsWith('"') && trimmed.endsWith('"')) {
    return trimmed.slice(1, -1).replace(/\\"/g, '"')
  }
  if (trimmed.length >= 2 && trimmed.startsWith("'") && trimmed.endsWith("'")) {
    return trimmed.slice(1, -1).replace(/''/g, "'")
  }
  return trimmed
}

function firstMarkdownHeading(text: string): string | null {
  for (const line of text.split('\n')) {
    const match = /^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$/.exec(line)
    if (match) return cleanMarkdownInline(match[1])
  }
  return null
}

function firstMarkdownParagraph(text: string): string | null {
  const paragraph: string[] = []
  let inFence = false

  for (const rawLine of text.split('\n')) {
    const trimmed = rawLine.trim()
    if (/^(```|~~~)/.test(trimmed)) {
      inFence = !inFence
      continue
    }
    if (inFence) continue
    if (!trimmed) {
      if (paragraph.length) return cleanMarkdownInline(paragraph.join(' '))
      continue
    }
    if (/^\s{0,3}#{1,6}\s+/.test(rawLine)) {
      if (paragraph.length) return cleanMarkdownInline(paragraph.join(' '))
      continue
    }
    if (/^\s{0,3}[-*_]{3,}\s*$/.test(rawLine)) continue

    paragraph.push(trimmed.replace(/^>\s?/, '').replace(/^[-*+]\s+/, '').replace(/^\d+\.\s+/, ''))
  }

  return paragraph.length ? cleanMarkdownInline(paragraph.join(' ')) : null
}

function cleanMarkdownInline(text: string): string {
  return text
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/<[^>]+>/g, ' ')
    .replace(/[`*_~]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

function compactPlainText(text: string, limit: number): string {
  const compacted = cleanMarkdownInline(text).replace(/\s+/g, ' ').trim()
  if (compacted.length <= limit) return compacted
  return compacted.slice(0, Math.max(0, limit - 3)).trimEnd() + '...'
}

function humanizeFolderName(folderName: string): string {
  const cleaned = folderName.replace(/[._-]+/g, ' ').replace(/\s+/g, ' ').trim()
  return cleaned ? cleaned.replace(/\b[a-z]/g, (char) => char.toUpperCase()) : 'Untitled Skill'
}

function skillIdFromFolderName(folderName: string): string {
  const id = folderName.trim().toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 96)
  return id || `skill-${hashString(folderName)}`
}

function hashString(value: string): string {
  let hash = 0x811c9dc5
  for (let i = 0; i < value.length; i++) {
    hash ^= value.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(36)
}
