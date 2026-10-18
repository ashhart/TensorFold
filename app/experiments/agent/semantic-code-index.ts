// Semantic / symbol codebase index — the CONTAINED, pure core.
//
// This is the embedding-based counterpart to the lexical `retrieval-index.ts`
// and the AST-based `code-intelligence.ts`. It has three pure pieces:
//
//   1. chunkCodeFile / chunkCodeFiles — files → semantic chunks. A chunk is a
//      symbol-ish span (function / class / interface / export block) with a
//      declaration heuristic, falling back to a sliding window so no file is lost.
//   2. buildSemanticIndex + searchSemanticIndex — a dependency-injected embedder
//      (Embedder) turns chunk text into vectors; cosine similarity ranks them.
//      No network, no fs — you inject the embedder (hash-embedder for tests,
//      a real provider embedder in production).
//   3. buildSymbolGraph — a symbol adjacency graph: nodes are declared symbols,
//      edges are "file A references symbol declared in file B" derived from the
//      chunks' referenced identifiers. Pure adjacency, no TypeScript program.
//
// Everything is synchronous and side-effect free except the injected embedder,
// which is the only async seam. Full auto-sync (watching the fs, persisting the
// vectors to disk, incremental re-embedding) is DEFERRED and lives outside this
// module — see the wire report.

export interface SemanticCodeFile {
  path: string
  content: string
  modifiedAt?: number | null
}

export type SymbolChunkKind =
  | 'function'
  | 'class'
  | 'interface'
  | 'type'
  | 'enum'
  | 'const'
  | 'method'
  | 'block'

export interface SemanticChunk {
  id: string
  path: string
  kind: SymbolChunkKind
  /** Declared symbol name, or null for anonymous / window fallback chunks. */
  symbol: string | null
  exported: boolean
  startLine: number
  endLine: number
  text: string
  /** Identifiers referenced inside the chunk (deduped, minus the chunk's own symbol). */
  references: string[]
}

/**
 * The injected embedding seam. Given N texts, returns N vectors of equal
 * dimension. Real providers batch a single HTTP call; tests inject a
 * deterministic hash embedder. Vectors need NOT be pre-normalized — cosine
 * similarity normalizes at compare time.
 */
export type Embedder = (texts: string[]) => Promise<number[][]>

export interface SemanticIndexEntry {
  chunk: SemanticChunk
  vector: number[]
}

export interface SemanticIndex {
  dim: number
  generatedAt: number
  entries: SemanticIndexEntry[]
}

export interface SemanticSearchHit {
  chunk: SemanticChunk
  score: number
}

export interface SemanticSearchResult {
  query: string
  hits: SemanticSearchHit[]
  total: number
}

export interface SymbolGraphNode {
  symbol: string
  path: string
  kind: SymbolChunkKind
  exported: boolean
  startLine: number
}

export interface SymbolGraphEdge {
  /** Referencing symbol id (path#symbol) — or a file-level pseudo-node. */
  from: string
  /** Referenced declared symbol id (path#symbol). */
  to: string
  /** How many chunks in `from`'s file reference `to`. */
  weight: number
}

export interface SymbolGraph {
  nodes: SymbolGraphNode[]
  edges: SymbolGraphEdge[]
  /** symbolId -> ids it references (outbound adjacency). */
  adjacency: Record<string, string[]>
  /** symbolId -> ids that reference it (inbound adjacency). */
  reverse: Record<string, string[]>
}

// ---------------------------------------------------------------------------
// 1. Chunking
// ---------------------------------------------------------------------------

const WINDOW_LINES = 40
const MAX_CHUNK_LINES = 200
const MAX_CHUNK_CHARS = 8_000

// Matches a top-of-declaration line and captures kind + name.
const DECL_RE =
  /^(?<indent>\s*)(?<exp>export\s+(?:default\s+)?)?(?:(?<abstract>abstract\s+)?(?:async\s+)?(?<kind>function|class|interface|type|enum)\s+(?<name1>[A-Za-z_$][\w$]*)|(?<constkw>const|let|var)\s+(?<name2>[A-Za-z_$][\w$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:function\b|\([^)]*\)\s*(?::[^=>]+)?=>|[A-Za-z_$][\w$]*\s*=>))/

const IDENT_RE = /[A-Za-z_$][\w$]*/g

// Reserved words we never treat as referenced symbols.
const STOPWORDS = new Set([
  'const', 'let', 'var', 'function', 'return', 'if', 'else', 'for', 'while', 'do',
  'switch', 'case', 'break', 'continue', 'new', 'this', 'super', 'class', 'extends',
  'implements', 'interface', 'type', 'enum', 'export', 'import', 'from', 'as', 'default',
  'async', 'await', 'yield', 'typeof', 'instanceof', 'in', 'of', 'void', 'delete',
  'true', 'false', 'null', 'undefined', 'try', 'catch', 'finally', 'throw', 'public',
  'private', 'protected', 'readonly', 'static', 'abstract', 'get', 'set', 'string',
  'number', 'boolean', 'any', 'unknown', 'never', 'object', 'Array', 'Promise', 'Record',
  'then', 'map', 'filter', 'reduce', 'forEach', 'push', 'slice', 'length',
])

export function chunkCodeFile(file: SemanticCodeFile): SemanticChunk[] {
  const path = normalizeSlash(file.path)
  const lines = file.content.split('\n')
  const chunks: SemanticChunk[] = []

  // Find declaration boundaries at the top level (indent-based heuristic).
  const decls: Array<{ line: number; kind: SymbolChunkKind; name: string; exported: boolean }> = []
  for (let i = 0; i < lines.length; i++) {
    const m = DECL_RE.exec(lines[i])
    if (!m || !m.groups) continue
    const name = m.groups.name1 || m.groups.name2
    if (!name) continue
    const rawKind = m.groups.kind
    const kind: SymbolChunkKind = rawKind
      ? (rawKind as SymbolChunkKind)
      : 'const'
    decls.push({ line: i, kind, name, exported: Boolean(m.groups.exp) })
  }

  if (decls.length) {
    // Any preamble before the first declaration becomes a block chunk.
    if (decls[0].line > 0) {
      pushWindowed(chunks, path, 0, decls[0].line - 1, lines)
    }
    for (let d = 0; d < decls.length; d++) {
      const start = decls[d].line
      const end = (d + 1 < decls.length ? decls[d + 1].line : lines.length) - 1
      pushSymbolChunk(chunks, path, decls[d], start, end, lines)
    }
  } else {
    // No detected declarations — sliding-window the whole file.
    pushWindowed(chunks, path, 0, lines.length - 1, lines)
  }

  return chunks
}

export function chunkCodeFiles(files: SemanticCodeFile[]): SemanticChunk[] {
  const out: SemanticChunk[] = []
  for (const file of files) out.push(...chunkCodeFile(file))
  return out
}

function pushSymbolChunk(
  chunks: SemanticChunk[],
  path: string,
  decl: { line: number; kind: SymbolChunkKind; name: string; exported: boolean },
  start: number,
  end: number,
  lines: string[],
): void {
  // A single symbol span may be huge; split into windows but keep symbol metadata.
  let cursor = start
  let part = 0
  while (cursor <= end) {
    const stop = Math.min(cursor + MAX_CHUNK_LINES - 1, end)
    const text = clampText(lines.slice(cursor, stop + 1).join('\n'))
    const idSuffix = part === 0 ? '' : `:${part}`
    chunks.push({
      id: `${path}#${decl.name}@${cursor + 1}${idSuffix}`,
      path,
      kind: decl.kind,
      symbol: decl.name,
      exported: decl.exported,
      startLine: cursor + 1,
      endLine: stop + 1,
      text,
      references: extractReferences(text, decl.name),
    })
    cursor = stop + 1
    part++
  }
}

function pushWindowed(chunks: SemanticChunk[], path: string, start: number, end: number, lines: string[]): void {
  for (let cursor = start; cursor <= end; cursor += WINDOW_LINES) {
    const stop = Math.min(cursor + WINDOW_LINES - 1, end)
    const text = clampText(lines.slice(cursor, stop + 1).join('\n'))
    if (!text.trim()) continue
    chunks.push({
      id: `${path}#block@${cursor + 1}`,
      path,
      kind: 'block',
      symbol: null,
      exported: false,
      startLine: cursor + 1,
      endLine: stop + 1,
      text,
      references: extractReferences(text, null),
    })
  }
}

function extractReferences(text: string, own: string | null): string[] {
  const seen = new Set<string>()
  for (const m of text.matchAll(IDENT_RE)) {
    const id = m[0]
    if (id === own) continue
    if (STOPWORDS.has(id)) continue
    if (id.length < 2) continue
    seen.add(id)
  }
  return [...seen]
}

function clampText(text: string): string {
  return text.length > MAX_CHUNK_CHARS ? text.slice(0, MAX_CHUNK_CHARS) : text
}

// ---------------------------------------------------------------------------
// 2. Embedding index + cosine search
// ---------------------------------------------------------------------------

export async function buildSemanticIndex(
  chunks: SemanticChunk[],
  embed: Embedder,
  now = Date.now(),
): Promise<SemanticIndex> {
  if (!chunks.length) return { dim: 0, generatedAt: now, entries: [] }
  const vectors = await embed(chunks.map((chunk) => embeddingText(chunk)))
  if (vectors.length !== chunks.length) {
    throw new Error(`Embedder returned ${vectors.length} vectors for ${chunks.length} chunks.`)
  }
  const dim = vectors[0]?.length ?? 0
  const entries: SemanticIndexEntry[] = []
  for (let i = 0; i < chunks.length; i++) {
    const vector = vectors[i]
    if (!Array.isArray(vector) || vector.length !== dim) {
      throw new Error(`Embedding vector ${i} has wrong dimension (expected ${dim}, got ${vector?.length}).`)
    }
    entries.push({ chunk: chunks[i], vector })
  }
  return { dim, generatedAt: now, entries }
}

export async function searchSemanticIndex(
  index: SemanticIndex,
  query: string,
  embed: Embedder,
  options: { limit?: number; minScore?: number } = {},
): Promise<SemanticSearchResult> {
  const trimmed = String(query ?? '').trim()
  const limit = clampInt(options.limit, 8, 1, 100)
  const minScore = Number.isFinite(options.minScore) ? Number(options.minScore) : -Infinity
  if (!trimmed || !index.entries.length) {
    return { query: trimmed, hits: [], total: index.entries.length }
  }
  const [queryVector] = await embed([trimmed])
  if (!queryVector || queryVector.length !== index.dim) {
    throw new Error(`Query embedding dimension (${queryVector?.length}) does not match index dimension (${index.dim}).`)
  }
  const qNorm = norm(queryVector)
  const hits: SemanticSearchHit[] = []
  for (const entry of index.entries) {
    const score = cosine(queryVector, qNorm, entry.vector)
    if (score >= minScore) hits.push({ chunk: entry.chunk, score })
  }
  hits.sort((a, b) => b.score - a.score || a.chunk.id.localeCompare(b.chunk.id))
  return { query: trimmed, hits: hits.slice(0, limit), total: index.entries.length }
}

/** The text we embed for a chunk: symbol + path context + body. */
export function embeddingText(chunk: SemanticChunk): string {
  const header = [chunk.exported ? 'exported' : '', chunk.kind, chunk.symbol ?? '', chunk.path]
    .filter(Boolean)
    .join(' ')
  return `${header}\n${chunk.text}`
}

/**
 * A deterministic, network-free embedder for tests and offline fallback:
 * token-hashing into a fixed-dim bag-of-words vector. NOT semantic-quality —
 * it exists so the index/search machinery is unit-testable without a provider.
 */
export function hashEmbedder(dim = 64): Embedder {
  return async (texts: string[]) =>
    texts.map((text) => {
      const vec = new Array(dim).fill(0)
      for (const m of text.toLowerCase().matchAll(IDENT_RE)) {
        const token = m[0]
        if (token.length < 2) continue
        vec[hashToken(token) % dim] += 1
      }
      return vec
    })
}

export function cosineSimilarity(a: number[], b: number[]): number {
  return cosine(a, norm(a), b)
}

function cosine(a: number[], aNorm: number, b: number[]): number {
  if (!aNorm) return 0
  const bNorm = norm(b)
  if (!bNorm) return 0
  const n = Math.min(a.length, b.length)
  let dot = 0
  for (let i = 0; i < n; i++) dot += a[i] * b[i]
  return dot / (aNorm * bNorm)
}

function norm(v: number[]): number {
  let sum = 0
  for (const x of v) sum += x * x
  return Math.sqrt(sum)
}

function hashToken(token: string): number {
  let h = 2166136261
  for (let i = 0; i < token.length; i++) {
    h ^= token.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

// ---------------------------------------------------------------------------
// 3. Symbol adjacency graph
// ---------------------------------------------------------------------------

export function buildSymbolGraph(chunks: SemanticChunk[]): SymbolGraph {
  // Declared symbols become nodes (keyed by path#symbol so same-name symbols in
  // different files stay distinct). Name -> canonical node id lets references
  // resolve across files.
  const nodes: SymbolGraphNode[] = []
  const nodeIdBySymbol = new Map<string, string>() // symbol name -> node id
  const nodeIds = new Set<string>()

  for (const chunk of chunks) {
    if (!chunk.symbol) continue
    const id = symbolId(chunk.path, chunk.symbol)
    if (nodeIds.has(id)) continue
    nodeIds.add(id)
    nodes.push({
      symbol: chunk.symbol,
      path: chunk.path,
      kind: chunk.kind,
      exported: chunk.exported,
      startLine: chunk.startLine,
    })
    nodeIdBySymbol.set(chunk.symbol, id)
  }

  // Edges: for each chunk, every referenced identifier that resolves to a
  // declared symbol becomes an edge from the chunk's own symbol (or a file
  // pseudo-node) to that declaration. Self-edges are dropped.
  const edgeWeights = new Map<string, number>() // `${from} ${to}` -> weight
  for (const chunk of chunks) {
    const from = chunk.symbol ? symbolId(chunk.path, chunk.symbol) : filePseudoNode(chunk.path)
    for (const ref of chunk.references) {
      const to = nodeIdBySymbol.get(ref)
      if (!to || to === from) continue
      const key = `${from} ${to}`
      edgeWeights.set(key, (edgeWeights.get(key) ?? 0) + 1)
    }
  }

  const edges: SymbolGraphEdge[] = []
  const adjacency: Record<string, string[]> = {}
  const reverse: Record<string, string[]> = {}
  for (const [key, weight] of edgeWeights) {
    const [from, to] = key.split(' ')
    edges.push({ from, to, weight })
    ;(adjacency[from] ??= []).push(to)
    ;(reverse[to] ??= []).push(from)
  }
  edges.sort((a, b) => b.weight - a.weight || a.from.localeCompare(b.from) || a.to.localeCompare(b.to))
  for (const list of Object.values(adjacency)) list.sort()
  for (const list of Object.values(reverse)) list.sort()

  return { nodes, edges, adjacency, reverse }
}

/** Declared symbols with no inbound edges — candidate dead code / entry points. */
export function findUnreferencedSymbols(graph: SymbolGraph): SymbolGraphNode[] {
  return graph.nodes.filter((node) => !(graph.reverse[symbolId(node.path, node.symbol)]?.length))
}

export function symbolId(path: string, symbol: string): string {
  return `${normalizeSlash(path)}#${symbol}`
}

function filePseudoNode(path: string): string {
  return `${normalizeSlash(path)}#<file>`
}

function normalizeSlash(value: string): string {
  return value.replace(/\\/g, '/').replace(/^\.\/+/, '')
}

function clampInt(value: unknown, fallback: number, min: number, max: number): number {
  const n = Number(value)
  if (!Number.isFinite(n)) return fallback
  return Math.max(min, Math.min(max, Math.floor(n)))
}
