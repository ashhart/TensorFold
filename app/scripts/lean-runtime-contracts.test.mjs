import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import path from 'node:path'
const root = new URL('../', import.meta.url)
const read = path => readFileSync(new URL(path, root), 'utf8')
for (const name of ['src/lib/ipc.ts', 'src/lib/remote-transport.ts']) {
  const file = ts.createSourceFile(name, read(name), ts.ScriptTarget.Latest, true)
  for (const statement of file.statements) {
    if (ts.isImportDeclaration(statement) && !statement.importClause?.isTypeOnly) {
      assert.doesNotMatch(statement.moduleSpecifier.text, /(?:^|\/)mock(?:\/|$)/, `${name} eagerly imports browser fixtures`)
    }
  }
}
const contract = ts.createSourceFile('shared/studio-api.ts', read('shared/studio-api.ts'), ts.ScriptTarget.Latest, true)
const apiInterface = contract.statements.find(s => ts.isInterfaceDeclaration(s) && s.name.text === 'TensorFoldStudioApi')
assert.ok(apiInterface)
assert.ok(contract.statements.every(s => !ts.isImportDeclaration(s) || s.importClause?.isTypeOnly), 'API contract must remain type-only')
assert.match(read('src/main.tsx'), /await prepareBrowserDemo\(\)/)
assert.match(read('src/lib/remote-transport.ts'), /STUDIO_API_SCHEMA/)
const schemaSource = ts.createSourceFile('shared/studio-api-schema.ts', read('shared/studio-api-schema.ts'), ts.ScriptTarget.Latest, true)
const schemaDeclaration = schemaSource.statements.find(ts.isVariableStatement).declarationList.declarations[0]
let schemaObject = schemaDeclaration.initializer
while (!ts.isObjectLiteralExpression(schemaObject)) schemaObject = schemaObject.expression
// Resolve the actual local declarations so modular contracts remain exhaustive.
// Unknown imports, spreads and cycles are errors, never silently omitted keys.
const source = name => ts.createSourceFile(name, read(name), ts.ScriptTarget.Latest, true)
function imported(file, name) {
  for (const statement of file.statements.filter(ts.isImportDeclaration)) {
    const binding = statement.importClause?.namedBindings
    if (!binding || !ts.isNamedImports(binding)) continue
    const entry = binding.elements.find(item => item.name.text === name)
    if (!entry) continue
    assert.ok(statement.moduleSpecifier.text.startsWith('.'), 'contract imports must be local')
    return [source(path.posix.join(path.posix.dirname(file.fileName), statement.moduleSpecifier.text) + '.ts'), (entry.propertyName ?? entry.name).text]
  }
  assert.fail(`Unresolved contract declaration ${name} in ${file.fileName}`)
}
function interfaceEntries(file, name, seen = new Set()) {
  const key = `${file.fileName}:${name}`
  assert.ok(!seen.has(key), `Cyclic interface ${key}`)
  const next = new Set([...seen, key])
  const declaration = file.statements.find(item => ts.isInterfaceDeclaration(item) && item.name.text === name)
  if (!declaration) return interfaceEntries(...imported(file, name), next)
  const parents = (declaration.heritageClauses ?? []).flatMap(clause => clause.types.flatMap(type => interfaceEntries(file, type.expression.getText(file), next)))
  return [...parents, ...declaration.members.map(member => {
    assert.ok(member.name, 'contract members must have explicit names')
    return [member.name.getText(file), ts.isMethodSignature(member) ? member.type?.getText(file) === '() => void' ? 'subscription' : 'method' : 'property']
  })]
}
function schemaEntries(file, name, seen = new Set()) {
  const key = `${file.fileName}:${name}`
  assert.ok(!seen.has(key), `Cyclic schema ${key}`)
  const next = new Set([...seen, key])
  const declaration = file.statements.filter(ts.isVariableStatement).flatMap(item => [...item.declarationList.declarations]).find(item => item.name.getText(file) === name)
  if (!declaration) return schemaEntries(...imported(file, name), next)
  let object = declaration.initializer
  while (object && !ts.isObjectLiteralExpression(object)) object = object.expression
  assert.ok(object && ts.isObjectLiteralExpression(object), `Expected literal schema ${key}`)
  return object.properties.flatMap(property => {
    if (ts.isSpreadAssignment(property)) {
      assert.ok(ts.isIdentifier(property.expression), 'schema spreads must reference known constants')
      return schemaEntries(file, property.expression.text, next)
    }
    assert.ok(ts.isPropertyAssignment(property) && ts.isStringLiteral(property.initializer), 'schema entries must have explicit fallback kinds')
    return [[property.name.getText(file), property.initializer.text]]
  })
}
const unique = entries => {
  assert.equal(new Set(entries.map(([key]) => key)).size, entries.length, 'contract keys cannot be overridden')
  return Object.fromEntries(entries)
}
const expected = unique(interfaceEntries(contract, 'TensorFoldStudioApi'))
const actual = unique(schemaEntries(schemaSource, 'STUDIO_API_SCHEMA'))
assert.deepEqual(actual, expected, 'every contract key needs an exact paired fallback kind')
console.log('Renderer contract and lazy demo graph checks passed')
