import assert from 'node:assert/strict'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import { createRequire } from 'node:module'
import { execFileSync } from 'node:child_process'
const root = process.cwd(), packRoot = path.join(root, 'src/lib/preferences/packs'), sourceRoot = path.join(root, 'src/lib/preferences/sources')
const sourceKeys = readdirSync(sourceRoot).filter(file => file.endsWith('.json')).flatMap(file => JSON.parse(readFileSync(path.join(sourceRoot, file), 'utf8')))
assert.equal(new Set(sourceKeys).size, sourceKeys.length, 'English source fragments cannot hide duplicates')
const sourceSet = new Set(sourceKeys), token = text => [...text.matchAll(/\{([a-zA-Z][a-zA-Z0-9_]*)\}/g)].map(value => value[1]).sort()
const requiredLocales = ['zh-Hans', 'zh-Hant', 'ja', 'ko', 'ru', 'pl', 'la', 'es', 'hi', 'th', 'vi', 'id']
for (const locale of requiredLocales) {
  const entries = new Map(), dir = path.join(packRoot, locale), files = readdirSync(dir).filter(file => file.endsWith('.json'))
  for (const file of files) {
    const raw = readFileSync(path.join(dir, file), 'utf8'), data = JSON.parse(raw)
    assert.ok(raw.split('\n').length < 600, `Small data files: ${locale}/${file}`)
    assert.ok(data && !Array.isArray(data) && typeof data === 'object')
    for (const [key, value] of Object.entries(data)) {
      assert.ok(sourceSet.has(key), `Unregistered UI source in ${locale}: ${key}`)
      assert.ok(!entries.has(key), `Duplicate UI source in ${locale}: ${key}`)
      assert.equal(typeof value, 'string'); assert.ok(value.trim())
      assert.doesNotMatch(value, /[\u0000\u202a-\u202e\u2066-\u2069]/u, 'Pack text cannot smuggle bidi controls; isolate dynamic values at formatting time')
      assert.deepEqual(token(value), token(key), `Placeholder parity in ${locale}: ${key}`)
      entries.set(key, value)
    }
  }
  assert.deepEqual([...entries.keys()].sort(), [...sourceSet].sort(), `Every selectable ${locale} pack needs every registered source, not English fallback coverage`)
}
const require = createRequire(import.meta.url), ts = require('typescript')
const routed = []
function scan(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) { if (entry.name !== 'packs' && entry.name !== 'sources') scan(full); continue }
    if (!/\.tsx?$/.test(entry.name) || /\.test\./.test(entry.name) || /test-fixture/.test(entry.name)) continue
    const sf = ts.createSourceFile(full, readFileSync(full, 'utf8'), ts.ScriptTarget.Latest, true, full.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS)
    const visit = node => {
      if (ts.isCallExpression(node)) {
        const name = node.expression.getText(sf), index = name === 'authoredText' ? 1 : ['tx', 'currentAuthoredText'].includes(name) ? 0 : -1, arg = node.arguments[index]
        if (arg && ts.isStringLiteral(arg)) routed.push({ source: arg.text, file: full.slice(root.length + 1) })
      }
      ts.forEachChild(node, visit)
    }; visit(sf)
  }
}
scan(path.join(root, 'src'))
for (const item of routed) assert.ok(sourceSet.has(item.source), `Register new authored source from ${item.file}: ${item.source}`)
execFileSync(process.execPath, ['scripts/build-language-pack-index.mjs', '--check'], { cwd: root, stdio: 'pipe' })
console.log(`Bundled language coverage passes ${requiredLocales.length} locales, ${sourceSet.size} exact sources, ${routed.length} routed call sites, bounded JSON, placeholders, bidi safety and reproducible offline indices`)
