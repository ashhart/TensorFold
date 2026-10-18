import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { isComposerSendKey } from '../lib/composer-keyboard'
const file = ts.createSourceFile('Composer.tsx', readFileSync('src/components/Composer.tsx', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const owner = file.statements.find((row): row is ts.FunctionDeclaration => ts.isFunctionDeclaration(row) && row.name?.text === 'Composer')!
const statement = owner.body!.statements.find(row => ts.isVariableStatement(row) && row.declarationList.declarations.some(declaration => ts.isIdentifier(declaration.name) && declaration.name.text === 'onKeyDown'))!
const source = ts.createPrinter().printNode(ts.EmitHint.Unspecified, statement, file)
function scenario(sendKey?: 'enter' | 'modified-enter', busy = false, picking = false, mentioning = false) {
  let sends = 0, queued = 0, prevented = 0
  const handler = new Function('isComposerSendKey', 'settings', 'busy', 'thread', 'submit', 'queueMessage', 'pick', 'pickItems', 'mention', 'capabilityItems', 'items', 'selectPick', 'pIdx', 'mIdx', 'selectCapability', 'selectMention', ts.transpileModule(source + '\nreturn onKeyDown;', { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText)(isComposerSendKey, { uiPreferences: { sendKey } }, busy, { id: 'fixture' }, () => sends++, () => queued++, picking ? 'skill' : null, picking ? [{ name: 'fixture' }] : [], mentioning, mentioning ? ['fixture'] : [], [], () => sends++, 0, 0, () => sends++, () => sends++)
  return { key: (extra: any = {}) => handler({ key: 'Enter', nativeEvent: { isComposing: false }, preventDefault() { prevented++ }, ...extra }), value: () => ({ sends, queued, prevented }) }
}
const normal = scenario(); normal.key(); assert.deepEqual(normal.value(), { sends: 1, queued: 0, prevented: 1 })
const modified = scenario('modified-enter'); modified.key(); assert.equal(modified.value().sends, 0); modified.key({ metaKey: true }); assert.equal(modified.value().sends, 1)
const running = scenario('modified-enter', true); running.key(); assert.equal(running.value().queued, 0); running.key({ ctrlKey: true }); assert.equal(running.value().queued, 1)
for (const extra of [{ shiftKey: true, metaKey: true }, { keyCode: 229, metaKey: true }, { nativeEvent: { isComposing: true }, metaKey: true }, { key: 'v', metaKey: true }, { key: 'z', metaKey: true }]) { const current = scenario('modified-enter'); current.key(extra); assert.deepEqual(current.value(), { sends: 0, queued: 0, prevented: 0 }) }
for (const [picking, mentioning] of [[true, false], [false, true]]) {
  const popup = scenario(undefined, false, picking, mentioning)
  popup.key({ shiftKey: true }); assert.deepEqual(popup.value(), { sends: 0, queued: 0, prevented: 0 }, 'Shift+Enter remains a newline even with a mention or skill suggestion open')
  popup.key({ key: 'Tab' }); assert.equal(popup.value().sends, 1, 'Explicit Tab still accepts the selected suggestion')
}
console.log('Production main Composer handler applies the selected send gesture to ordinary and queued sends while preserving editing and IME')
