import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'
const app = path.resolve(new URL('../', import.meta.url).pathname)
const inventory = JSON.parse(readFileSync(path.join(app, 'experiments/inventory.json'), 'utf8'))
assert.equal(inventory.modules.length, 20)
assert.equal(new Set(inventory.modules).size, 20)
const experiments = path.join(app, 'experiments/agent')
for (const name of inventory.modules) {
  assert.ok(existsSync(path.join(experiments, name + '.ts')))
  assert.ok(existsSync(path.join(experiments, name + '.test.ts')))
  assert.ok(!existsSync(path.join(app, 'electron/agent', name + '.ts')))
}
function walk(dir) { return readdirSync(dir).flatMap(name => { const file = path.join(dir, name); return statSync(file).isDirectory() ? walk(file) : /\.(?:tsx?|mjs|cjs)$/.test(name) ? [file] : [] }) }
for (const dir of ['src', 'electron', 'shared']) {
  for (const file of walk(path.join(app, dir)).filter(file => !file.includes('.test.'))) {
    const source = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true)
    const visit = node => {
      const specifier = ts.isImportDeclaration(node) || ts.isExportDeclaration(node) ? node.moduleSpecifier :
        ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword ? node.arguments[0] : null
      if (specifier && ts.isStringLiteral(specifier) && specifier.text.startsWith('.')) assert.ok(!path.resolve(path.dirname(file), specifier.text).startsWith(experiments + path.sep), `Production source imports unwired experiment: ${path.relative(app, file)}`)
      ts.forEachChild(node, visit)
    }
    visit(source)
  }
}
assert.ok(existsSync(path.join(app, 'electron/agent/acp-server-main.ts')), 'standalone maintained ACP entrypoint must survive')
assert.ok(existsSync(path.join(app, 'electron/agent/acp-server.ts')))
for (const name of ['run-tests.mjs','check-test-agent-coverage.mjs']) assert.match(readFileSync(path.join(app, 'scripts', name), 'utf8'), /'experiments'/)
console.log('Unwired experiment inventory, production exclusion and preserved test discovery passed')
