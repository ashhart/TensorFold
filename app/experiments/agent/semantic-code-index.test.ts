import assert from 'node:assert/strict'
import {
  buildSemanticIndex,
  buildSymbolGraph,
  chunkCodeFile,
  chunkCodeFiles,
  cosineSimilarity,
  findUnreferencedSymbols,
  hashEmbedder,
  searchSemanticIndex,
  symbolId,
  type Embedder,
} from './semantic-code-index'

let passed = 0
function ok(name: string) {
  passed++
  console.log(`  ✓ ${name}`)
}

console.log('semantic-code-index tests:')

// --- 1. Chunking -----------------------------------------------------------
{
  const chunks = chunkCodeFile({
    path: './src/math.ts',
    content: [
      'import { add } from "../../electron/agent/add"',
      '',
      'export function multiply(a: number, b: number) {',
      '  return a * b',
      '}',
      '',
      'const helper = () => multiply(2, 3)',
    ].join('\n'),
  })
  const symbols = chunks.map((c) => c.symbol)
  assert.ok(symbols.includes('multiply'), 'captures exported function')
  assert.ok(symbols.includes('helper'), 'captures arrow const')
  const mul = chunks.find((c) => c.symbol === 'multiply')!
  assert.equal(mul.kind, 'function')
  assert.equal(mul.exported, true)
  assert.equal(mul.path, 'src/math.ts', 'leading ./ stripped from path')
  assert.ok(mul.startLine >= 1 && mul.endLine >= mul.startLine)
  ok('chunkCodeFile detects symbols, kind, exported, normalized path')
}

{
  // A file with no detectable declarations falls back to sliding windows.
  const content = Array.from({ length: 95 }, (_, i) => `line ${i} token${i}`).join('\n')
  const chunks = chunkCodeFile({ path: 'data.txt', content })
  assert.ok(chunks.length >= 2, 'long undecorated file split into multiple window chunks')
  assert.ok(chunks.every((c) => c.kind === 'block' && c.symbol === null))
  ok('chunkCodeFile windows files with no declarations')
}

{
  // References exclude the chunk's own symbol and stopwords.
  const [chunk] = chunkCodeFile({
    path: 'x.ts',
    content: 'export function alpha() { return beta(gamma) }',
  })
  assert.ok(chunk.references.includes('beta'))
  assert.ok(chunk.references.includes('gamma'))
  assert.ok(!chunk.references.includes('alpha'), 'own symbol excluded from refs')
  assert.ok(!chunk.references.includes('return'), 'stopword excluded from refs')
  ok('extractReferences excludes own symbol + stopwords')
}

// --- 2. Embedding index + cosine search ------------------------------------
{
  const files = [
    { path: 'auth.ts', content: 'export function login(user, password) { return session }' },
    { path: 'geometry.ts', content: 'export function circleArea(radius) { return radius * radius * pi }' },
  ]
  const chunks = chunkCodeFiles(files)
  const embed = hashEmbedder(48)
  const index = await buildSemanticIndex(chunks, embed, 111)
  assert.equal(index.generatedAt, 111)
  assert.equal(index.dim, 48)
  assert.equal(index.entries.length, chunks.length)
  assert.ok(index.entries.every((e) => e.vector.length === 48))

  const result = await searchSemanticIndex(index, 'login password session user', embed, { limit: 1 })
  assert.equal(result.total, chunks.length)
  assert.equal(result.hits.length, 1)
  assert.equal(result.hits[0].chunk.symbol, 'login', 'semantically nearest chunk ranked first')
  assert.ok(result.hits[0].score > 0.3, 'cosine score is meaningful')
  ok('buildSemanticIndex + searchSemanticIndex rank the relevant chunk first')
}

{
  // Empty query and empty index return empty hits, not crashes.
  const index = await buildSemanticIndex([], hashEmbedder(), 5)
  assert.equal(index.dim, 0)
  const empty = await searchSemanticIndex(index, 'anything', hashEmbedder())
  assert.equal(empty.hits.length, 0)
  const blank = await searchSemanticIndex(
    await buildSemanticIndex(chunkCodeFile({ path: 'a.ts', content: 'export const x = 1' }), hashEmbedder()),
    '   ',
    hashEmbedder(),
  )
  assert.equal(blank.hits.length, 0, 'blank query yields no hits')
  ok('empty index and blank query are handled safely')
}

{
  // DI seam: a bogus embedder returning wrong-dim vectors is rejected.
  const badEmbed: Embedder = async (texts) => texts.map(() => [1, 2, 3])
  const goodChunks = chunkCodeFile({ path: 'a.ts', content: 'export const x = 1\nexport const y = 2' })
  const index = await buildSemanticIndex(goodChunks, hashEmbedder(16))
  await assert.rejects(() => searchSemanticIndex(index, 'x', badEmbed), /dimension/)
  ok('search rejects a query embedder with mismatched dimension')
}

{
  assert.equal(cosineSimilarity([1, 0], [1, 0]), 1)
  assert.equal(cosineSimilarity([1, 0], [0, 1]), 0)
  assert.equal(cosineSimilarity([0, 0], [1, 1]), 0, 'zero vector similarity is 0 not NaN')
  const partial = cosineSimilarity([1, 1, 0], [1, 0, 0])
  assert.ok(partial > 0 && partial < 1)
  ok('cosineSimilarity is correct on identical/orthogonal/zero vectors')
}

// --- 3. Symbol adjacency graph ---------------------------------------------
{
  const chunks = chunkCodeFiles([
    { path: 'util.ts', content: 'export function helper() { return 1 }' },
    { path: 'main.ts', content: 'export function run() { return helper() }' },
    { path: 'orphan.ts', content: 'export function lonely() { return 42 }' },
  ])
  const graph = buildSymbolGraph(chunks)

  const helperId = symbolId('util.ts', 'helper')
  const runId = symbolId('main.ts', 'run')
  assert.ok(graph.nodes.some((n) => n.symbol === 'helper' && n.path === 'util.ts'))
  assert.ok(graph.edges.some((e) => e.from === runId && e.to === helperId), 'run -> helper edge exists')
  assert.deepEqual(graph.adjacency[runId], [helperId], 'outbound adjacency')
  assert.deepEqual(graph.reverse[helperId], [runId], 'inbound adjacency')

  const unref = findUnreferencedSymbols(graph).map((n) => n.symbol).sort()
  // `run` and `lonely` have no inbound edges; `helper` does.
  assert.deepEqual(unref, ['lonely', 'run'])
  assert.ok(!unref.includes('helper'))
  ok('buildSymbolGraph resolves cross-file edges + unreferenced symbols')
}

{
  // Edge weights aggregate repeated references within a file.
  const chunks = chunkCodeFiles([
    { path: 'lib.ts', content: 'export function target() { return 0 }' },
    {
      path: 'caller.ts',
      content: [
        'export function a() { return target() }',
        'export function b() { return target() }',
      ].join('\n'),
    },
  ])
  const graph = buildSymbolGraph(chunks)
  const targetId = symbolId('lib.ts', 'target')
  const inbound = graph.reverse[targetId] ?? []
  assert.equal(inbound.length, 2, 'both callers reference target')
  const totalWeight = graph.edges.filter((e) => e.to === targetId).reduce((s, e) => s + e.weight, 0)
  assert.equal(totalWeight, 2)
  ok('symbol graph aggregates edge weights and inbound references')
}

console.log(`\n${passed} assertions passed.`)
