import { promises as fs } from 'node:fs'
import { existsSync, realpathSync } from 'node:fs'
import path from 'node:path'
import * as ts from 'typescript'
import { computeDiff, diffStat } from './diff'
import type { DiffLine } from './types'

const SKIP_DIRS = new Set(['node_modules', '.git', 'dist', 'dist-electron', '.next', 'build', '.cache', 'vendor', '.venv'])
const CODE_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.jsx', '.mjs', '.cjs', '.mts', '.cts', '.py', '.rb', '.go', '.rs', '.java', '.kt', '.swift', '.cs'])
const TYPESCRIPT_SYMBOL_EXTENSIONS = new Set(['.ts', '.tsx', '.js', '.jsx', '.mts', '.cts', '.mjs', '.cjs'])
const DEFAULT_MAX_FILES = 500
const DEFAULT_MAX_MATCHES = 100
const MAX_READ_BYTES = 1_500_000

export interface StructuralMatch {
  sourceId: string
  path: string
  line: number
  column: number
  excerpt: string
  variables: Record<string, string>
}

export interface StructuralSearchResult {
  query: string
  engine: 'ast-grep-compatible'
  matches: StructuralMatch[]
  truncated: boolean
}

export interface RewritePreviewResult {
  engine: 'ast-grep-compatible'
  files: Array<{ path: string; replacements: number; diff: DiffLine[] }>
  diff: DiffLine[]
  added: number
  removed: number
  applied: false
}

export interface SecurityFinding {
  sourceId: string
  ruleId: string
  severity: 'info' | 'warning' | 'error'
  path: string
  line: number
  message: string
  excerpt: string
  blocking: boolean
}

export interface SecurityScanResult {
  engine: 'semgrep-compatible'
  findings: SecurityFinding[]
  proposedFixes: Array<{ path: string; ruleId: string; diff: DiffLine[] }>
  diff: DiffLine[]
  applied: false
}

export interface CodebaseMapResult {
  generatedAt: number
  filesScanned: number
  entryPoints: string[]
  packageScripts: string[]
  packageDependencies: string[]
  imports: Array<{ from: string; to: string }>
  symbols: Array<{ sourceId: string; path: string; line: number; kind: string; name: string }>
}

export interface SymbolIndexEntry {
  sourceId: string
  path: string
  line: number
  column: number
  kind: string
  name: string
  container: string | null
  exported: boolean
}

export interface SymbolIndexResult {
  generatedAt: number
  engine: 'typescript-language-service'
  filesScanned: number
  symbols: SymbolIndexEntry[]
  truncated: boolean
}

export interface SafeRefactorPreviewResult {
  search: StructuralSearchResult
  rewrite: RewritePreviewResult
  scan: SecurityScanResult
  applied: false
}

interface CompiledPattern {
  regex: RegExp
  variableGroups: Map<string, string>
}

interface CodeFile {
  path: string
  abs: string
}

export async function structuralSearch(cwd: string, input: { pattern: string; path?: string; maxMatches?: number }): Promise<StructuralSearchResult> {
  const pattern = String(input.pattern ?? '').trim()
  if (!pattern) throw new Error('Structural pattern is required.')
  const maxMatches = boundedInt(input.maxMatches, DEFAULT_MAX_MATCHES, 1, 250)
  const compiled = compileStructuralPattern(pattern)
  const files = await collectCodeFiles(cwd, input.path, DEFAULT_MAX_FILES)
  const matches: StructuralMatch[] = []
  let truncated = false

  for (const file of files) {
    if (matches.length >= maxMatches) {
      truncated = true
      break
    }
    const content = await readSmallText(file.abs)
    if (content == null) continue
    for (const match of content.matchAll(compiled.regex)) {
      if (matches.length >= maxMatches) {
        truncated = true
        break
      }
      const index = match.index ?? 0
      const line = lineForIndex(content, index)
      const column = columnForIndex(content, index)
      matches.push({
        sourceId: `code:${file.path}:${line}`,
        path: file.path,
        line,
        column,
        excerpt: compactSnippet(match[0]),
        variables: variablesFromMatch(match.groups ?? {}, compiled.variableGroups),
      })
    }
  }

  return { query: pattern, engine: 'ast-grep-compatible', matches, truncated }
}

export async function rewritePreview(cwd: string, input: { pattern: string; rewrite: string; path?: string; maxReplacements?: number }): Promise<RewritePreviewResult> {
  const pattern = String(input.pattern ?? '').trim()
  const rewrite = String(input.rewrite ?? '')
  if (!pattern) throw new Error('Structural pattern is required.')
  if (!rewrite) throw new Error('Rewrite template is required.')
  const maxReplacements = boundedInt(input.maxReplacements, 20, 1, 100)
  const compiled = compileStructuralPattern(pattern)
  const files = await collectCodeFiles(cwd, input.path, DEFAULT_MAX_FILES)
  const changed: RewritePreviewResult['files'] = []
  const allDiff: DiffLine[] = []
  let remaining = maxReplacements

  for (const file of files) {
    if (remaining <= 0) break
    const before = await readSmallText(file.abs)
    if (before == null) continue
    let count = 0
    const after = before.replace(compiled.regex, (full, ...rest) => {
      if (remaining <= 0) return full
      const groups = rest.at(-1) && typeof rest.at(-1) === 'object' ? rest.at(-1) as Record<string, string> : {}
      remaining--
      count++
      return materializeRewrite(rewrite, groups, compiled.variableGroups)
    })
    if (!count || after === before) continue
    const diff = computeDiff(before, after)
    if (allDiff.length) allDiff.push({ type: 'hunk', text: '' })
    allDiff.push({ type: 'hunk', text: `*** ${file.path}` })
    allDiff.push(...diff)
    changed.push({ path: file.path, replacements: count, diff })
  }

  const stat = diffStat(allDiff)
  return { engine: 'ast-grep-compatible', files: changed, diff: allDiff, added: stat.added, removed: stat.removed, applied: false }
}

export async function semgrepCompatibleScan(cwd: string, input: { path?: string; autofix?: boolean; maxFindings?: number }): Promise<SecurityScanResult> {
  const maxFindings = boundedInt(input.maxFindings, DEFAULT_MAX_MATCHES, 1, 250)
  const files = await collectCodeFiles(cwd, input.path, DEFAULT_MAX_FILES)
  const findings: SecurityFinding[] = []
  const fixes: SecurityScanResult['proposedFixes'] = []
  const allDiff: DiffLine[] = []

  for (const file of files) {
    if (findings.length >= maxFindings) break
    const content = await readSmallText(file.abs)
    if (content == null) continue
    const lines = content.split(/\r?\n/)
    for (let index = 0; index < lines.length && findings.length < maxFindings; index++) {
      const line = lines[index]
      for (const rule of SEMGREP_RULES) {
        if (!rule.pattern.test(line)) continue
        findings.push({
          sourceId: `semgrep:${rule.id}:${file.path}:${index + 1}`,
          ruleId: rule.id,
          severity: rule.severity,
          path: file.path,
          line: index + 1,
          message: rule.message,
          excerpt: line.trim().slice(0, 240),
          blocking: rule.severity === 'error',
        })
      }
    }

    if (input.autofix && /\.innerHTML\b/.test(content)) {
      const after = content.replace(/\.innerHTML(\s*=)/g, '.textContent$1')
      if (after !== content) {
        const diff = computeDiff(content, after)
        if (allDiff.length) allDiff.push({ type: 'hunk', text: '' })
        allDiff.push({ type: 'hunk', text: `*** ${file.path}` })
        allDiff.push(...diff)
        fixes.push({ path: file.path, ruleId: 'js-inner-html-assignment', diff })
      }
    }
  }

  return { engine: 'semgrep-compatible', findings, proposedFixes: fixes, diff: allDiff, applied: false }
}

export async function buildCodebaseMap(cwd: string, input: { path?: string; maxFiles?: number } = {}): Promise<CodebaseMapResult> {
  const files = await collectCodeFiles(cwd, input.path, boundedInt(input.maxFiles, DEFAULT_MAX_FILES, 25, 1000))
  const entryPoints = await presentFiles(cwd, [
    'package.json',
    'index.html',
    'src/main.tsx',
    'src/main.ts',
    'src/App.tsx',
    'electron/main.ts',
    'electron/preload.ts',
    'vite.config.ts',
    'tsconfig.json',
    'AGENTS.md',
    'CLAUDE.md',
  ])
  const packageScripts: string[] = []
  const packageDependencies: string[] = []
  try {
    const pkg = JSON.parse(await fs.readFile(path.join(cwd, 'package.json'), 'utf8'))
    for (const [name, script] of Object.entries(pkg.scripts ?? {})) packageScripts.push(`${name}: ${String(script)}`)
    packageDependencies.push(...Object.keys({ ...(pkg.dependencies ?? {}), ...(pkg.devDependencies ?? {}) }).sort())
  } catch { /* no package */ }

  const imports: CodebaseMapResult['imports'] = []
  const symbols: CodebaseMapResult['symbols'] = []
  for (const file of files) {
    const content = await readSmallText(file.abs)
    if (content == null) continue
    collectImports(file.path, content, imports)
    collectSymbols(file.path, content, symbols)
  }

  return {
    generatedAt: Date.now(),
    filesScanned: files.length,
    entryPoints,
    packageScripts: packageScripts.slice(0, 40),
    packageDependencies: packageDependencies.slice(0, 80),
    imports: imports.slice(0, 300),
    symbols: symbols.slice(0, 300),
  }
}

export async function buildSymbolIndex(cwd: string, input: { path?: string; query?: string; maxFiles?: number; maxSymbols?: number } = {}): Promise<SymbolIndexResult> {
  const files = (await collectCodeFiles(cwd, input.path, boundedInt(input.maxFiles, DEFAULT_MAX_FILES, 25, 1000)))
    .filter((file) => TYPESCRIPT_SYMBOL_EXTENSIONS.has(path.extname(file.path).toLowerCase()))
  const maxSymbols = boundedInt(input.maxSymbols, 300, 1, 1000)
  const contents = new Map<string, string>()
  for (const file of files) {
    const content = await readSmallText(file.abs)
    if (content != null) contents.set(file.abs, content)
  }

  const service = ts.createLanguageService({
    getCompilationSettings: () => ({
      allowJs: true,
      checkJs: false,
      jsx: ts.JsxEmit.ReactJSX,
      module: ts.ModuleKind.ESNext,
      moduleResolution: ts.ModuleResolutionKind.Bundler,
      target: ts.ScriptTarget.ES2022,
    }),
    getCurrentDirectory: () => path.resolve(cwd),
    getDefaultLibFileName: (options) => ts.getDefaultLibFilePath(options),
    getScriptFileNames: () => [...contents.keys()],
    getScriptSnapshot: (fileName) => {
      const content = contents.get(fileName) ?? ts.sys.readFile(fileName)
      return content == null ? undefined : ts.ScriptSnapshot.fromString(content)
    },
    getScriptVersion: () => '1',
    fileExists: ts.sys.fileExists,
    readFile: ts.sys.readFile,
    readDirectory: ts.sys.readDirectory,
    directoryExists: ts.sys.directoryExists,
    getDirectories: ts.sys.getDirectories,
  })

  const query = String(input.query ?? '').trim().toLowerCase()
  const symbols: SymbolIndexEntry[] = []
  let truncated = false

  for (const file of files) {
    if (symbols.length >= maxSymbols) {
      truncated = true
      break
    }
    const content = contents.get(file.abs)
    if (content == null) continue
    const tree = service.getNavigationTree(file.abs)
    collectNavigationSymbols(file.path, content, tree.childItems ?? [], symbols, [], query, maxSymbols)
    if (symbols.length >= maxSymbols) truncated = true
  }

  return { generatedAt: Date.now(), engine: 'typescript-language-service', filesScanned: contents.size, symbols: symbols.slice(0, maxSymbols), truncated }
}

export async function safeRefactorPreview(cwd: string, input: { pattern: string; rewrite: string; path?: string; maxReplacements?: number }): Promise<SafeRefactorPreviewResult> {
  const search = await structuralSearch(cwd, { pattern: input.pattern, path: input.path })
  const rewrite = await rewritePreview(cwd, input)
  const scan = await semgrepCompatibleScan(cwd, { path: input.path, autofix: false })
  return { search, rewrite, scan, applied: false }
}

export function formatStructuralSearch(result: StructuralSearchResult): string {
  if (!result.matches.length) return `No structural matches for: ${result.query}`
  return [
    `Engine: ${result.engine}`,
    `Pattern: ${result.query}`,
    '',
    ...result.matches.map((match) => {
      const vars = Object.keys(match.variables).length
        ? ` | ${Object.entries(match.variables).map(([k, v]) => `${k}=${compactSnippet(v, 80)}`).join(', ')}`
        : ''
      return `- ${match.sourceId} | ${match.excerpt}${vars}`
    }),
    result.truncated ? '\nResults truncated.' : '',
  ].filter(Boolean).join('\n')
}

export function formatRewritePreview(result: RewritePreviewResult): string {
  if (!result.files.length) return 'No rewrite matches. No files were changed.'
  return [
    `Engine: ${result.engine}`,
    `Applied: ${result.applied}`,
    `Preview: ${result.files.length} file${result.files.length === 1 ? '' : 's'} · +${result.added} -${result.removed}`,
    '',
    ...result.files.map((file) => `- ${file.path}: ${file.replacements} replacement${file.replacements === 1 ? '' : 's'}`),
    '',
    'This is a proposed diff only. Apply it separately after approval.',
  ].join('\n')
}

export function formatSecurityScan(result: SecurityScanResult): string {
  const lines = [
    `Engine: ${result.engine}`,
    `Applied fixes: ${result.applied}`,
    `Findings: ${result.findings.length}`,
  ]
  if (result.findings.length) {
    lines.push('')
    for (const finding of result.findings) {
      lines.push(`- ${finding.sourceId} [${finding.severity}] ${finding.message}`)
      lines.push(`  ${finding.excerpt}`)
    }
  }
  if (result.proposedFixes.length) {
    lines.push('', `Proposed autofix patches: ${result.proposedFixes.length}`)
    for (const fix of result.proposedFixes) lines.push(`- ${fix.path} (${fix.ruleId})`)
    lines.push('Autofix output is a proposed patch only. No files were mutated.')
  }
  return lines.join('\n')
}

export function formatCodebaseMap(result: CodebaseMapResult): string {
  return [
    'Artifact: codebase-map',
    `Generated: ${new Date(result.generatedAt).toISOString()}`,
    `Files scanned: ${result.filesScanned}`,
    '',
    '## Entry points',
    result.entryPoints.length ? result.entryPoints.map((entry) => `- ${entry}`).join('\n') : '- none detected',
    '',
    '## Scripts',
    result.packageScripts.length ? result.packageScripts.slice(0, 20).map((script) => `- ${script}`).join('\n') : '- none detected',
    '',
    '## Dependency graph',
    result.imports.length ? result.imports.slice(0, 80).map((edge) => `- ${edge.from} -> ${edge.to}`).join('\n') : '- no imports detected',
    '',
    '## Symbols',
    result.symbols.length ? result.symbols.slice(0, 80).map((symbol) => `- ${symbol.sourceId} ${symbol.kind} ${symbol.name}`).join('\n') : '- no symbols detected',
  ].join('\n')
}

export function formatSymbolIndex(result: SymbolIndexResult): string {
  return [
    'Artifact: symbol-index',
    `Engine: ${result.engine}`,
    `Generated: ${new Date(result.generatedAt).toISOString()}`,
    `Files scanned: ${result.filesScanned}`,
    `Symbols: ${result.symbols.length}${result.truncated ? '+' : ''}`,
    '',
    result.symbols.length
      ? result.symbols.slice(0, 120).map((symbol) => {
        const container = symbol.container ? ` in ${symbol.container}` : ''
        const exported = symbol.exported ? ' exported' : ''
        return `- ${symbol.sourceId} ${symbol.kind}${exported} ${symbol.name}${container}`
      }).join('\n')
      : '- no symbols detected',
  ].join('\n')
}

export function formatSafeRefactorPreview(result: SafeRefactorPreviewResult): string {
  const blocking = result.scan.findings.filter((finding) => finding.blocking).length
  return [
    'Workflow: safe mechanical refactor',
    `Applied: ${result.applied}`,
    `Search: ${result.search.matches.length}${result.search.truncated ? '+' : ''} structural matches`,
    `Preview: ${result.rewrite.files.length} file${result.rewrite.files.length === 1 ? '' : 's'} · +${result.rewrite.added} -${result.rewrite.removed}`,
    `Guardrail scan: ${result.scan.findings.length} findings · ${blocking} blocking`,
    '',
    result.rewrite.files.length
      ? result.rewrite.files.map((file) => `- ${file.path}: ${file.replacements} replacement${file.replacements === 1 ? '' : 's'}`).join('\n')
      : 'No rewrite diff was produced.',
    '',
    'This workflow only searches, previews, and scans. Apply the diff separately after approval.',
  ].join('\n')
}

function compileStructuralPattern(pattern: string): CompiledPattern {
  const variableGroups = new Map<string, string>()
  let regex = ''
  let groupIndex = 0

  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i]
    if (/\s/.test(ch)) {
      while (i + 1 < pattern.length && /\s/.test(pattern[i + 1])) i++
      regex += '\\s+'
      continue
    }
    if (ch === '$') {
      const triple = pattern.slice(i, i + 3) === '$$$'
      const start = i + (triple ? 3 : 1)
      const nameMatch = /^[A-Za-z_][A-Za-z0-9_]*/.exec(pattern.slice(start))
      if (nameMatch) {
        const token = pattern.slice(i, start + nameMatch[0].length)
        const group = `v${groupIndex++}`
        variableGroups.set(token, group)
        regex += triple ? `(?<${group}>[\\s\\S]*?)` : `(?<${group}>[A-Za-z_$][\\w$]*|[^\\s,;(){}]+)`
        i = start + nameMatch[0].length - 1
        continue
      }
    }
    regex += escapeRegex(ch)
  }

  return { regex: new RegExp(regex, 'gm'), variableGroups }
}

function materializeRewrite(template: string, groups: Record<string, string>, variableGroups: Map<string, string>): string {
  return template.replace(/\$\$\$[A-Za-z_][A-Za-z0-9_]*|\$[A-Za-z_][A-Za-z0-9_]*/g, (token) => {
    const group = variableGroups.get(token)
    return group ? groups[group] ?? '' : token
  })
}

function variablesFromMatch(groups: Record<string, string>, variableGroups: Map<string, string>): Record<string, string> {
  const out: Record<string, string> = {}
  for (const [token, group] of variableGroups) out[token.replace(/^\$+/, '')] = groups[group] ?? ''
  return out
}

async function collectCodeFiles(cwd: string, relPath: string | undefined, maxFiles: number): Promise<CodeFile[]> {
  const base = safeResolve(cwd, relPath || '.')
  const out: CodeFile[] = []
  try {
    const stat = await fs.stat(base)
    if (stat.isFile()) {
      return CODE_EXTENSIONS.has(path.extname(base))
        ? [{ path: normalizeSlash(path.relative(path.resolve(cwd), base)), abs: base }]
        : []
    }
  } catch {
    return []
  }
  const walk = async (dir: string, depth: number) => {
    if (depth > 20 || out.length >= maxFiles) return
    let entries: any[] = []
    try { entries = await fs.readdir(dir, { withFileTypes: true }) } catch { return }
    for (const entry of entries.sort((a, b) => Number(b.isDirectory()) - Number(a.isDirectory()) || a.name.localeCompare(b.name))) {
      if (out.length >= maxFiles) break
      if (entry.name.startsWith('.') || SKIP_DIRS.has(entry.name)) continue
      if (entry.isSymbolicLink()) continue
      const abs = path.join(dir, entry.name)
      if (entry.isDirectory()) {
        await walk(abs, depth + 1)
        continue
      }
      if (!CODE_EXTENSIONS.has(path.extname(entry.name))) continue
      out.push({ path: normalizeSlash(path.relative(path.resolve(cwd), abs)), abs })
    }
  }
  await walk(base, 0)
  return out.sort((a, b) => a.path.localeCompare(b.path))
}

function safeResolve(cwd: string, relPath: string): string {
  const root = path.resolve(cwd)
  const realRoot = realpathSync.native(root)
  const target = path.resolve(root, relPath || '.')
  if (target !== root && !target.startsWith(root + path.sep)) throw new Error(`Path "${relPath}" escapes the workspace.`)
  let probe = target
  while (!existsSync(probe)) {
    const parent = path.dirname(probe)
    if (parent === probe) break
    probe = parent
  }
  const realProbe = realpathSync.native(probe)
  if (realProbe !== realRoot && !realProbe.startsWith(realRoot + path.sep)) throw new Error(`Path "${relPath}" escapes the workspace through a symlink.`)
  return target
}

async function readSmallText(abs: string): Promise<string | null> {
  try {
    const stat = await fs.stat(abs)
    if (stat.size > MAX_READ_BYTES) return null
    return await fs.readFile(abs, 'utf8')
  } catch {
    return null
  }
}

function collectImports(filePath: string, content: string, imports: CodebaseMapResult['imports']) {
  const re = /\bimport\s+(?:type\s+)?(?:[^'"]+\s+from\s+)?['"]([^'"]+)['"]|\brequire\s*\(\s*['"]([^'"]+)['"]\s*\)/g
  for (const match of content.matchAll(re)) {
    imports.push({ from: filePath, to: match[1] || match[2] })
  }
}

function collectSymbols(filePath: string, content: string, symbols: CodebaseMapResult['symbols']) {
  const re = /^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_$][\w$]*)|^\s*(?:export\s+)?class\s+([A-Za-z_$][\w$]*)|^\s*export\s+(?:const|let|var)\s+([A-Za-z_$][\w$]*)|^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>/gm
  for (const match of content.matchAll(re)) {
    const name = match[1] || match[2] || match[3] || match[4]
    const kind = match[1] ? 'function' : match[2] ? 'class' : match[3] ? 'export' : 'function'
    const line = lineForIndex(content, match.index ?? 0)
    symbols.push({ sourceId: `symbol:${filePath}:${line}`, path: filePath, line, kind, name })
  }
}

function collectNavigationSymbols(filePath: string, content: string, items: ts.NavigationTree[], out: SymbolIndexEntry[], stack: string[], query: string, maxSymbols: number) {
  for (const item of items) {
    if (out.length >= maxSymbols) return
    const spans = item.spans ?? []
    const span = spans[0]
    const name = cleanSymbolName(item.text)
    const kind = normalizeSymbolKind(item.kind)
    const isContainer = Boolean(item.childItems?.length)
    const searchable = `${name} ${kind} ${stack.join(' ')}`.toLowerCase()
    if (name && kind !== 'script' && (!query || searchable.includes(query))) {
      const start = Math.max(0, span?.start ?? 0)
      const line = lineForIndex(content, start)
      const column = columnForIndex(content, start)
      out.push({
        sourceId: `symbol:${filePath}:${line}:${slugSymbolName(name)}`,
        path: filePath,
        line,
        column,
        kind,
        name,
        container: stack.at(-1) ?? null,
        exported: /\bexport\b/.test(item.kindModifiers ?? ''),
      })
    }
    if (item.childItems?.length) {
      collectNavigationSymbols(filePath, content, item.childItems, out, isContainer && name ? [...stack, name] : stack, query, maxSymbols)
    }
  }
}

function cleanSymbolName(value: string): string {
  const name = String(value || '').trim()
  return name === '<function>' || name === '<class>' || name === '<interface>' ? '' : name
}

function normalizeSymbolKind(kind: string): string {
  return String(kind || 'unknown').replace(/\s+/g, '-').toLowerCase()
}

function slugSymbolName(name: string): string {
  return name.replace(/[^A-Za-z0-9_$.-]+/g, '_').slice(0, 80) || 'anonymous'
}

async function presentFiles(cwd: string, files: string[]): Promise<string[]> {
  const out: string[] = []
  for (const file of files) {
    try {
      await fs.access(path.join(cwd, file))
      out.push(file)
    } catch { /* absent */ }
  }
  return out
}

function lineForIndex(content: string, index: number): number {
  return content.slice(0, index).split('\n').length
}

function columnForIndex(content: string, index: number): number {
  const prev = content.lastIndexOf('\n', index - 1)
  return index - prev
}

function compactSnippet(value: string, max = 180): string {
  return value.replace(/\s+/g, ' ').trim().slice(0, max)
}

function normalizeSlash(value: string): string {
  return value.replace(/\\/g, '/')
}

function escapeRegex(value: string): string {
  return value.replace(/[\\^$.*+?()[\]{}|]/g, '\\$&')
}

function boundedInt(value: unknown, fallback: number, min: number, max: number): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return fallback
  return Math.max(min, Math.min(max, Math.floor(n)))
}

const SEMGREP_RULES = [
  {
    id: 'js-eval',
    severity: 'error' as const,
    pattern: /\beval\s*\(/,
    message: 'Avoid eval; it executes strings as code.',
  },
  {
    id: 'js-inner-html-assignment',
    severity: 'warning' as const,
    pattern: /\.innerHTML\s*=/,
    message: 'Avoid direct innerHTML assignment; prefer textContent or a sanitizer.',
  },
  {
    id: 'react-dangerously-set-inner-html',
    severity: 'warning' as const,
    pattern: /\bdangerouslySetInnerHTML\b/,
    message: 'dangerouslySetInnerHTML needs an explicit sanitization boundary.',
  },
  {
    id: 'node-child-process-exec',
    severity: 'warning' as const,
    pattern: /\bexec\s*\(/,
    message: 'child_process.exec-style calls should be policy-gated and argument-safe.',
  },
]
