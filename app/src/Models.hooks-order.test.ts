// Run: node src/Models.hooks-order.test.ts
// React reports #310 ("Rendered more hooks than during the previous render") when a hook sits below a
// conditional return. The Models workspace returns the "connect a provider" page when the build has no
// local engine, so no hook may be declared below that return.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

const file = path.join(__dirname, 'components', 'Models.tsx')
const sourceText = readFileSync(file, 'utf8')
const source = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)

function modelsComponent(node: ts.Node): ts.FunctionDeclaration | null {
  if (ts.isFunctionDeclaration(node) && node.name?.text === 'Models') return node
  let found: ts.FunctionDeclaration | null = null
  ts.forEachChild(node, (child) => { found ??= modelsComponent(child) })
  return found
}

const models = modelsComponent(source)
assert.ok(models?.body, 'Models component found')

const hooks: number[] = []
let earlyReturn: number | null = null
const visit = (node: ts.Node): void => {
  if (ts.isIfStatement(node) && ts.isReturnStatement(node.thenStatement) && node.thenStatement.expression) {
    // The no-local-engine gate: `if (noLocalEngine) return <NoLocalEngineModels />`
    if (node.thenStatement.expression.getText(source).includes('NoLocalEngineModels')) earlyReturn ??= node.getStart(source)
  }
  if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
    const name = node.expression.text
    // Direct hook calls only: useStore(...), useEngine(), useEffect(...), useAuthoredText()
    if (/^use[A-Z]/.test(name)) hooks.push(node.getStart(source))
  }
  ts.forEachChild(node, visit)
}
visit(models.body)

assert.ok(earlyReturn !== null, 'the no-local-engine early return is present')
const lateHooks = hooks.filter((position) => position > earlyReturn!)
assert.deepEqual(
  lateHooks.map((position) => sourceText.slice(position, position + 40).split('(')[0]),
  [],
  'every hook call must precede the early return, or React #310 returns when the engine is missing',
)

console.log(`Models hook order: ${hooks.length} hooks all declared before the early return`)
