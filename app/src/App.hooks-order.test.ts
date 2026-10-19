// Run: node src/App.hooks-order.test.ts
// React reports #310 ("Rendered more hooks than during the previous render") when a hook sits
// below a conditional return. App returns the bootstrap screen while settings load, so every hook
// must be declared above that return.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import ts from 'typescript'

const file = path.join(__dirname, 'App.tsx')
const sourceText = readFileSync(file, 'utf8')
const source = ts.createSourceFile(file, sourceText, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)

function appComponent(node: ts.Node): ts.FunctionDeclaration | null {
  if (ts.isFunctionDeclaration(node) && node.name?.text === 'App') return node
  let found: ts.FunctionDeclaration | null = null
  ts.forEachChild(node, (child) => { found ??= appComponent(child) })
  return found
}

const app = appComponent(source)
assert.ok(app?.body, 'App component found')

const hooks: number[] = []
let earlyReturn: number | null = null
const visit = (node: ts.Node): void => {
  if (ts.isIfStatement(node) && ts.isReturnStatement(node.thenStatement) && node.thenStatement.expression) {
    // The bootstrap gate: `if (bootstrap !== 'ready') return <BootstrapRecovery .../>`
    if (node.thenStatement.expression.getText(source).includes('BootstrapRecovery')) earlyReturn ??= node.getStart(source)
  }
  if (ts.isCallExpression(node) && ts.isIdentifier(node.expression)) {
    const name = node.expression.text
    // Direct hook calls only: useStore(...), useEngine(), useEffect(...), useUiPreferences(...)
    if (/^use[A-Z]/.test(name)) hooks.push(node.getStart(source))
  }
  ts.forEachChild(node, visit)
}
visit(app.body)

assert.ok(earlyReturn !== null, 'the bootstrap early return is present')
const lateHooks = hooks.filter((position) => position > earlyReturn!)
assert.deepEqual(
  lateHooks.map((position) => sourceText.slice(position, position + 40).split('(')[0]),
  [],
  'every hook call must precede the bootstrap early return, or React #310 returns when settings load',
)

console.log(`App hook order: ${hooks.length} hooks all declared before the bootstrap return`)
