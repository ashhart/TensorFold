import assert from 'node:assert/strict'
import { readFileSync, readdirSync, statSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

// New run-owning records must opt out or copy an allowlist. Absent is a legacy disk compatibility rule.
const constructors: string[] = [], missing: string[] = []
function inspect(file: string) {
  const ast = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, true)
  const visit = (node: ts.Node) => {
    if (ts.isObjectLiteralExpression(node)) {
      const properties = new Set(node.properties.filter(ts.isPropertyAssignment).map(property => property.name.getText(ast).replace(/["']/g, '')))
      if (properties.has('id') && properties.has('apiMessages') && properties.has('timeline')) {
        constructors.push(file)
        if (!properties.has('documentSourceIds')) missing.push(`${file}:${ast.getLineAndCharacterOfPosition(node.pos).line + 1}`)
      }
    }
    ts.forEachChild(node, visit)
  }
  visit(ast)
}
function walk(folder: string) { for (const name of readdirSync(folder)) { const file = path.join(folder, name); if (statSync(file).isDirectory()) walk(file); else if (file.endsWith('.ts') && !file.includes('.test.') && !file.includes('.integration.')) inspect(file) } }
walk('electron/main/host'); inspect('electron/agent/acp-server.ts')
assert.ok(constructors.length >= 12, 'Check the real production constructor inventory, including independent background runs')
assert.deepEqual(missing, [], 'New ordinary, remote, imported, bot, mission, loop, ticket, handoff and derived thread constructors cannot silently fall back to legacy enabled collections')
console.log(`PASS ${constructors.length} production thread constructors: all explicitly opt out or copy their owning document selection`)
