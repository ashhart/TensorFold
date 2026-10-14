// Run: npx tsx electron/agent/instruction-graph.test.ts
// (imports ./instruction-graph.ts → ./tools which uses extensionless imports, so tsx is required)
import assert from 'node:assert'
import { promises as fs } from 'node:fs'
import { existsSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { resolveImports, buildInstructionGraph } from './instruction-graph.ts'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

async function ws(seed: Record<string, string>): Promise<string> {
  const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-instr-'))
  for (const [rel, content] of Object.entries(seed)) {
    const abs = path.join(cwd, rel)
    await fs.mkdir(path.dirname(abs), { recursive: true })
    await fs.writeFile(abs, content, 'utf8')
  }
  return cwd
}

async function main() {
  console.log('instruction-graph tests:')

  // 1) imports resolve + dedup (diamond: AGENTS imports child twice → child once)
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\n@docs/child.md\n@docs/child.md\n', 'docs/child.md': 'child body\n' })
    const g = await buildInstructionGraph(cwd)
    assert.match(g.resolvedContent, /child body/)
    assert.strictEqual(g.nodes.filter((n) => n.path === 'docs/child.md').length, 1, 'child deduped to one node')
    ok('resolves @imports and dedups diamond references')
  }

  // 2) provider overlay appears (kind: provider) after shared root
  {
    const cwd = await ws({ 'AGENTS.md': '# Shared root\n', '.vontra/openai.md': 'openai overlay rules\n' })
    const g = await buildInstructionGraph(cwd, 'openai')
    assert.ok(g.nodes.some((n) => n.kind === 'provider'), 'has a provider node')
    assert.match(g.resolvedContent, /openai overlay rules/)
    assert.ok(g.resolvedContent.indexOf('Shared root') < g.resolvedContent.indexOf('openai overlay'), 'overlay comes after shared')
    ok('applies provider overlay after shared root')
  }

  // 3) ignored local override — node kept, body excluded from resolvedContent
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\n@docs/child.md\n', 'docs/child.md': 'IGNORED BODY\n', '.vontra/instructions-ignore': 'docs/child.md\n' })
    const g = await buildInstructionGraph(cwd)
    const child = g.nodes.find((n) => n.path === 'docs/child.md')
    assert.ok(child?.ignored === true, 'child marked ignored')
    assert.doesNotMatch(g.resolvedContent, /IGNORED BODY/)
    ok('honors ignored local override (kept in graph, out of content)')
  }

  // 4) missing import — no throw, root survives, placeholder recorded
  {
    const cwd = await ws({ 'AGENTS.md': '# Root body\n@docs/nope.md\n' })
    const g = await buildInstructionGraph(cwd)
    assert.match(g.resolvedContent, /Root body/)
    assert.ok(g.nodes.some((n) => n.path === 'docs/nope.md'), 'missing import recorded as a node')
    ok('tolerates a missing import without crashing')
  }

  // 5) cycle detection terminates (a→b→a)
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\n@a.md\n', 'a.md': '@b.md\n', 'b.md': '@a.md\n' })
    const g = await buildInstructionGraph(cwd)
    assert.ok(g.cycles.length > 0, 'cycle reported')
    assert.strictEqual(g.nodes.filter((n) => n.path === 'a.md').length, 1)
    assert.strictEqual(g.nodes.filter((n) => n.path === 'b.md').length, 1)
    ok('detects cycles and terminates')
  }

  // 6) jail escape rejected — ../ and absolute, no read of /etc/passwd
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\n' })
    const rel = await resolveImports('@../../../../../../etc/passwd', { cwd })
    assert.match(rel, /escapes the workspace/i)
    assert.doesNotMatch(rel, /root:.*:0:0:/)
    const absSpec = await resolveImports('@/etc/passwd', { cwd })
    assert.doesNotMatch(absSpec, /root:.*:0:0:/)
    ok('rejects ../ and absolute path escapes')
  }

  // 7) credential path rejected before any read (.env)
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\n', '.env': 'SECRET=shh\n' })
    const out = await resolveImports('@.env', { cwd })
    assert.match(out, /safety policy|blocked/i)
    assert.doesNotMatch(out, /SECRET=shh/)
    ok('blocks @.env via the credential denylist (no read)')
  }

  // 8) token estimates present
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\nSome guidance here.\n@docs/child.md\n', 'docs/child.md': 'child body\n' })
    const g = await buildInstructionGraph(cwd)
    assert.ok(g.tokenEstimate > 0, 'graph token estimate > 0')
    assert.ok(g.nodes.every((n) => n.tokenEstimate > 0), 'every (non-blocked) node has a token estimate')
    ok('produces token estimates')
  }

  // 9) symlink escape rejected (skip where symlinks unavailable)
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\n' })
    const outside = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-outside-'))
    await fs.writeFile(path.join(outside, 'secret.txt'), 'leaked-symlink-body\n', 'utf8')
    const link = path.join(cwd, 'linked-outside')
    try {
      await fs.symlink(outside, link, 'dir')
      const out = await resolveImports('@linked-outside/secret.txt', { cwd })
      assert.match(out, /escapes the workspace/i)
      assert.doesNotMatch(out, /leaked-symlink-body/)
      ok('rejects symlink escapes')
    } catch (e: any) {
      if (existsSync(link)) throw e
      ok('symlink escape test skipped where symlinks are unavailable')
    }
  }

  // 10) never inlines in-workspace secret carriers the narrow denylist used to miss
  {
    const npmToken = `npm_${['SUPER', 'SECRET', 'TOKEN'].join('')}`
    const privateKeyHeader = ['-----BEGIN', 'PRIVATE KEY-----'].join(' ')
    const privateKeyFooter = ['-----END', 'PRIVATE KEY-----'].join(' ')
    const privateKeyBody = ['LEAKED', 'KEY', 'BODY'].join('')
    const dbPassword = ['hunter2', 'prod'].join('-')
    const cwd = await ws({
      'AGENTS.md': '# Root\n@.npmrc\n@keys/deploy.pem\n@infra/prod.tfvars\n',
      '.npmrc': `//registry.npmjs.org/:_authToken=${npmToken}\n`,
      'keys/deploy.pem': `${privateKeyHeader}\n${privateKeyBody}\n${privateKeyFooter}\n`,
      'infra/prod.tfvars': `db_password = "${dbPassword}"\n`,
    })
    const g = await buildInstructionGraph(cwd)
    assert.ok(!g.resolvedContent.includes(npmToken))
    assert.ok(!g.resolvedContent.includes(privateKeyBody))
    assert.ok(!g.resolvedContent.includes(privateKeyHeader))
    assert.ok(!g.resolvedContent.includes(dbPassword))
    ok('never inlines in-workspace secret carriers (.npmrc / *.pem / *.tfvars)')
  }

  // 11) only documentation files are importable — a non-doc extension is skipped
  {
    const configSecret = ['secret', 'config', 'value'].join('-')
    const cwd = await ws({ 'AGENTS.md': '# Root body kept\n@config.json\n', 'config.json': `{"token":"${configSecret}"}\n` })
    const g = await buildInstructionGraph(cwd)
    assert.ok(!g.resolvedContent.includes(configSecret))
    assert.match(g.resolvedContent, /Root body kept/)
    ok('restricts @import to doc/text files (.json skipped)')
  }

  // 12) trailing `# comment` on an @import line still resolves
  {
    const cwd = await ws({ 'AGENTS.md': '# Root\n@docs/child.md # optional note\n', 'docs/child.md': 'child body via comment\n' })
    const g = await buildInstructionGraph(cwd)
    assert.match(g.resolvedContent, /child body via comment/)
    ok('supports a trailing # comment on @import lines')
  }

  // 13) hostile fan-out is bounded (per-file import cap + node cap)
  {
    const many = Array.from({ length: 500 }, (_, i) => `@ghost${i}.md`).join('\n')
    const cwd = await ws({ 'AGENTS.md': `# Root\n${many}\n` })
    const g = await buildInstructionGraph(cwd)
    assert.ok(g.nodes.length <= 64, `node count bounded under hostile fan-out (got ${g.nodes.length})`)
    ok('bounds node count under hostile fan-out')
  }

  // 14) a huge repository-authored instruction file is prefix-read and model-bounded.
  {
    const cwd = await ws({ 'AGENTS.md': `HEAD_MARKER\n${'x'.repeat(2 * 1024 * 1024)}\nTAIL_MUST_NOT_LOAD` })
    const g = await buildInstructionGraph(cwd)
    assert.match(g.resolvedContent, /HEAD_MARKER/)
    assert.doesNotMatch(g.resolvedContent, /TAIL_MUST_NOT_LOAD/)
    assert.ok(g.nodes[0].content.length <= 16 * 1024)
    ok('prefix-reads oversized instruction files instead of materializing them')
  }

  console.log(`\n${passed} checks passed.`)
}

main().catch((error) => { console.error(error); process.exitCode = 1 })
