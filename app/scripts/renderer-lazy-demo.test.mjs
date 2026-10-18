// Optional compiled-output check after a renderer-only build, without running the app.
import assert from 'node:assert/strict'
import { readFileSync, readdirSync, existsSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
const root = path.resolve(new URL('../', import.meta.url).pathname)
const html = path.join(root, 'dist/index.html')
if (!existsSync(html)) {
  console.log('No compiled renderer present; source graph is covered by lean-runtime-contracts.test.mjs')
} else {
  const assets = path.join(root, 'dist/assets')
  const index = readFileSync(html, 'utf8')
  const entry = index.match(/<script[^>]*src="([^"]+)"/)?.[1]
  assert.ok(entry, 'compiled entry script required')
  const appChunk = readdirSync(assets).find(name => /^App-.*\.js$/.test(name))
  assert.ok(appChunk, 'shared app renderer chunk required')
  const queued = [path.join(root, 'dist', entry.replace(/^\//, '')), path.join(assets, appChunk)]
  const seen = new Set()
  const imports = []
  while (queued.length) {
    const file = queued.pop()
    if (seen.has(file)) continue
    seen.add(file)
    const source = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true, ts.ScriptKind.JS)
    for (const statement of source.statements) {
      if ((ts.isImportDeclaration(statement) || ts.isExportDeclaration(statement)) && statement.moduleSpecifier && ts.isStringLiteral(statement.moduleSpecifier)) {
        const specifier = statement.moduleSpecifier.text
        imports.push(specifier)
        if (specifier.startsWith('.')) queued.push(path.resolve(path.dirname(file), specifier))
      }
    }
  }
  assert.ok(readdirSync(assets).some(name => /^mock-.*\.js$/.test(name)), 'demo stays available as its own chunk')
  assert.ok(imports.every(specifier => !/(?:^|\/)mock-/.test(specifier)), 'desktop and paired boot must not synchronously load the demo chunk')
  assert.ok([...seen].every(file => !/(?:^|\/)mock-/.test(file)))
  console.log(`Renderer static bootstrap graph excludes lazy mock chunk across ${seen.size} initial modules`)
}
