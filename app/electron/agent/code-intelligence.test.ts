// Run: npx tsx electron/agent/code-intelligence.test.ts
import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { buildCodebaseMap, buildSymbolIndex, formatSymbolIndex, rewritePreview, safeRefactorPreview, semgrepCompatibleScan, structuralSearch } from './code-intelligence'

let passed = 0
const ok = (name: string) => { passed++; console.log(`  ✓ ${name}`) }

async function makeWorkspace(): Promise<string> {
  const cwd = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-code-intel-'))
  await fs.mkdir(path.join(cwd, 'src'), { recursive: true })
  await fs.writeFile(path.join(cwd, 'package.json'), JSON.stringify({
    scripts: { test: 'vitest' },
    dependencies: { react: '^18.0.0' },
  }, null, 2), 'utf8')
  await fs.writeFile(path.join(cwd, 'src/app.ts'), [
    "import { helper } from './helper'",
    'export const alpha = 1;',
    'export const beta = 2;',
    'export function run(input: string) {',
    '  return helper(input)',
    '}',
    'document.body.innerHTML = "<b>x</b>"',
    '',
  ].join('\n'), 'utf8')
  await fs.writeFile(path.join(cwd, 'src/helper.ts'), [
    'export function helper(value: string) {',
    '  return value.trim()',
    '}',
    '',
  ].join('\n'), 'utf8')
  return cwd
}

async function main() {
  console.log('code intelligence tests:')
  const cwd = await makeWorkspace()

  {
    const result = await structuralSearch(cwd, { pattern: 'export const $NAME = $VALUE', path: 'src' })
    assert.equal(result.engine, 'ast-grep-compatible')
    assert.equal(result.matches.length, 2)
    assert.deepEqual(result.matches.map((match) => match.variables.NAME), ['alpha', 'beta'])
    assert.ok(result.matches.every((match) => match.sourceId.startsWith('code:src/app.ts:')))
    ok('structural search finds equivalent declarations with source ids')
  }

  {
    const before = await fs.readFile(path.join(cwd, 'src/app.ts'), 'utf8')
    const result = await rewritePreview(cwd, {
      pattern: 'export const $NAME = $VALUE',
      rewrite: 'export let $NAME = $VALUE',
      path: 'src/app.ts',
    })
    assert.equal(result.applied, false)
    assert.equal(result.files.length, 1)
    assert.equal(result.files[0].replacements, 2)
    assert.ok(result.diff.some((line) => line.type === 'del' && line.text.includes('export const alpha')))
    assert.ok(result.diff.some((line) => line.type === 'add' && line.text.includes('export let alpha')))
    assert.equal(await fs.readFile(path.join(cwd, 'src/app.ts'), 'utf8'), before)
    ok('rewrite preview returns a reviewable diff without mutating files')
  }

  {
    const before = await fs.readFile(path.join(cwd, 'src/app.ts'), 'utf8')
    const result = await semgrepCompatibleScan(cwd, { path: 'src', autofix: true })
    assert.equal(result.engine, 'semgrep-compatible')
    assert.equal(result.applied, false)
    assert.ok(result.findings.some((finding) => finding.ruleId === 'js-inner-html-assignment'))
    assert.ok(result.proposedFixes.some((fix) => fix.path === 'src/app.ts'))
    assert.ok(result.diff.some((line) => line.type === 'add' && line.text.includes('textContent')))
    assert.equal(await fs.readFile(path.join(cwd, 'src/app.ts'), 'utf8'), before)
    ok('semgrep-compatible scan returns findings and proposed autofix patches only')
  }

  {
    const result = await buildCodebaseMap(cwd)
    assert.ok(result.entryPoints.includes('package.json'))
    assert.ok(result.packageScripts.some((script) => script.startsWith('test:')))
    assert.ok(result.packageDependencies.includes('react'))
    assert.ok(result.imports.some((edge) => edge.from === 'src/app.ts' && edge.to === './helper'))
    assert.ok(result.symbols.some((symbol) => symbol.sourceId === 'symbol:src/helper.ts:1' && symbol.name === 'helper'))
    ok('codebase map emits dependency graph and symbol artifacts')
  }

  {
    const result = await buildSymbolIndex(cwd, { path: 'src', query: 'helper' })
    assert.equal(result.engine, 'typescript-language-service')
    assert.ok(result.filesScanned >= 2)
    assert.ok(result.symbols.some((symbol) => symbol.sourceId === 'symbol:src/helper.ts:1:helper' && symbol.name === 'helper'))
    assert.ok(result.symbols.every((symbol) => symbol.name.toLowerCase().includes('helper') || String(symbol.container ?? '').toLowerCase().includes('helper')))
    assert.match(formatSymbolIndex(result), /Artifact: symbol-index/)
    ok('symbol index uses TypeScript language service source ids')
  }

  {
    const before = await fs.readFile(path.join(cwd, 'src/app.ts'), 'utf8')
    const result = await safeRefactorPreview(cwd, {
      pattern: 'export const $NAME = $VALUE',
      rewrite: 'export let $NAME = $VALUE',
      path: 'src/app.ts',
    })
    assert.equal(result.applied, false)
    assert.equal(result.search.matches.length, 2)
    assert.equal(result.rewrite.files.length, 1)
    assert.ok(result.scan.findings.some((finding) => finding.sourceId.startsWith('semgrep:')))
    assert.equal(await fs.readFile(path.join(cwd, 'src/app.ts'), 'utf8'), before)
    ok('safe refactor workflow searches, previews, scans, and does not mutate')
  }

  await fs.rm(cwd, { recursive: true, force: true })
  console.log(`code intelligence tests: ${passed} passed`)
}

void main()
