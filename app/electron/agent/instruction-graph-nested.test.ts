// Run: jiti electron/agent/instruction-graph-nested.test.ts
// Nested / directory-scoped AGENTS.md resolution (the Codex/Cursor/Amp parity feature).
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { buildInstructionGraph } from './instruction-graph.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

console.log('instruction-graph nested tests:')

async function makeWorkspace(files: Record<string, string>): Promise<string> {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-instr-'))
  for (const [rel, content] of Object.entries(files)) {
    const abs = path.join(dir, rel)
    await fs.mkdir(path.dirname(abs), { recursive: true })
    await fs.writeFile(abs, content)
  }
  return dir
}

// ---- nested files are discovered, ordered shallow→deep, and node_modules is skipped ----
{
  const ws = await makeWorkspace({
    'AGENTS.md': 'ROOT_RULES_MARKER',
    'packages/api/AGENTS.md': 'API_RULES_MARKER',
    'packages/api/deep/CLAUDE.md': 'DEEP_RULES_MARKER',
    'node_modules/pkg/AGENTS.md': 'NODE_MODULES_SHOULD_NOT_APPEAR',
    'dist/AGENTS.md': 'DIST_SHOULD_NOT_APPEAR',
  })
  try {
    const g = await buildInstructionGraph(ws)
    const c = g.resolvedContent
    assert.ok(c.includes('ROOT_RULES_MARKER'), 'root AGENTS.md included')
    assert.ok(c.includes('API_RULES_MARKER'), 'nested packages/api/AGENTS.md included')
    assert.ok(c.includes('DEEP_RULES_MARKER'), 'deeper CLAUDE.md included')
    assert.ok(!c.includes('NODE_MODULES_SHOULD_NOT_APPEAR'), 'node_modules is skipped')
    assert.ok(!c.includes('DIST_SHOULD_NOT_APPEAR'), 'dist is skipped')
    // shallow → deep ordering (root before api before deep)
    assert.ok(c.indexOf('ROOT_RULES_MARKER') < c.indexOf('API_RULES_MARKER'), 'root before nested')
    assert.ok(c.indexOf('API_RULES_MARKER') < c.indexOf('DEEP_RULES_MARKER'), 'shallower nested before deeper')
    // scoped provenance header names the directory
    assert.ok(/scoped instructions for packages\/api\/ /.test(c), 'scoped header names the directory')
    ok('nested discovery: included shallow→deep, node_modules/dist skipped, scoped headers')
  } finally { await fs.rm(ws, { recursive: true, force: true }) }
}

// ---- AGENTS.md wins over CLAUDE.md in the same directory ----
{
  const ws = await makeWorkspace({
    'AGENTS.md': 'ROOT',
    'sub/AGENTS.md': 'SUB_AGENTS_WINS',
    'sub/CLAUDE.md': 'SUB_CLAUDE_LOSES',
  })
  try {
    const g = await buildInstructionGraph(ws)
    assert.ok(g.resolvedContent.includes('SUB_AGENTS_WINS'), 'AGENTS.md is used')
    assert.ok(!g.resolvedContent.includes('SUB_CLAUDE_LOSES'), 'CLAUDE.md is not double-included when AGENTS.md exists')
    ok('AGENTS.md wins over CLAUDE.md in the same directory')
  } finally { await fs.rm(ws, { recursive: true, force: true }) }
}

// ---- a workspace with ONLY nested files (no root) still surfaces them ----
{
  const ws = await makeWorkspace({ 'service/AGENTS.md': 'ONLY_NESTED_MARKER' })
  try {
    const g = await buildInstructionGraph(ws)
    assert.ok(g.resolvedContent.includes('ONLY_NESTED_MARKER'), 'nested file surfaces with no root present')
    ok('nested files surface even without a root AGENTS.md')
  } finally { await fs.rm(ws, { recursive: true, force: true }) }
}

// ---- no instruction files anywhere → empty graph (unchanged behavior) ----
{
  const ws = await makeWorkspace({ 'src/index.ts': 'export const x = 1' })
  try {
    const g = await buildInstructionGraph(ws)
    assert.equal(g.resolvedContent.trim(), '', 'no instruction files → empty resolved content')
    ok('a workspace with no instruction files yields an empty graph')
  } finally { await fs.rm(ws, { recursive: true, force: true }) }
}

console.log(`\n${passed} checks passed.`)
