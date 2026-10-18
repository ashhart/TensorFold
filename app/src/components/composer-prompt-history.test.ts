import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import ts from 'typescript'
import { ConversationDrafts } from '../lib/conversation-drafts'
import { composerDraftKey } from '../../shared/composer-drafts'
import { insertTextareaText } from '../lib/textarea-editing'
const file = ts.createSourceFile('Composer.tsx', readFileSync('src/components/Composer.tsx', 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
let callback: ts.Expression | undefined
const walk = (node: ts.Node) => {
  if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(file) === 'PromptHistory') {
    const attribute = node.attributes.properties.find((item): item is ts.JsxAttribute => ts.isJsxAttribute(item) && item.name.getText(file) === 'onInsert')
    if (attribute?.initializer && ts.isJsxExpression(attribute.initializer)) callback = attribute.initializer.expression
  }
  ts.forEachChild(node, walk)
}
walk(file); assert.ok(callback)
const source = ts.createPrinter().printNode(ts.EmitHint.Expression, callback, file)
const drafts = new ConversationDrafts(null)
const a = { surface: 'chat' as const, conversationId: 'history-one', hostId: null, incognito: false }, b = { ...a, conversationId: 'history-two' }
const previous = { document: globalThis.document, HTMLTextAreaElement: globalThis.HTMLTextAreaElement }
class Area {
  current = ''; selectionStart = 0; selectionEnd = 0
  get value() { return this.current } set value(value: string) { this.current = value }
  focus() {} setSelectionRange(start: number, end: number) { this.selectionStart = start; this.selectionEnd = end } dispatchEvent() { return true }
}
const area = new Area(), undos: string[] = []
Object.defineProperty(globalThis, 'HTMLTextAreaElement', { configurable: true, value: Area })
Object.defineProperty(globalThis, 'document', { configurable: true, value: { execCommand(command: string, _ui: boolean, insert: string) { assert.equal(command, 'insertText'); undos.push(area.value); area.current = area.value.slice(0, area.selectionStart) + insert + area.value.slice(area.selectionEnd); return true } } })
const activeDraftKey = { current: composerDraftKey(a) }, taRef = { current: area }
const onInsert = new Function('activeDraftKey', 'draft', 'taRef', 'setText', 'insertTextareaText', ts.transpileModule('const result = ' + source + '; return result;', { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText)(activeDraftKey, { scopeKey: composerDraftKey(a) }, taRef, (text: string) => drafts.update(a, { text }), insertTextareaText)
try {
  drafts.update(a, { text: 'Unfinished draft', attachments: [{ id: 'pending', name: 'note.txt', kind: 'document', mime: 'text/plain', status: 'needs-reselection' }] })
  drafts.update(b, { text: 'Other conversation draft' })
  // A user can keep typing while the protected main-process recall is being authorized.
  drafts.update(a, { text: 'Unfinished draft with newer typing' }); area.current = drafts.get(a).text
  const attachment = drafts.get(a).attachments[0]
  area.selectionStart = 0; area.selectionEnd = area.value.length
  onInsert('Earlier prompt')
  assert.equal(drafts.get(a).text, 'Unfinished draft with newer typing\n\nEarlier prompt', 'Recall appends to the live editor value rather than replacing selection or a captured stale draft')
  assert.equal(drafts.get(a).attachments[0], attachment)
  assert.equal(drafts.get(b).text, 'Other conversation draft')
  assert.deepEqual(undos, ['Unfinished draft with newer typing'], 'Appending is one native editor transaction and preserves the prior draft for Undo')
  activeDraftKey.current = composerDraftKey(b); const before = drafts.capture(a); onInsert('Late wrong-chat response')
  assert.deepEqual(drafts.capture(a), before); assert.equal(drafts.get(b).text, 'Other conversation draft')
  console.log('Production prompt recall preserves current typed text, selected text, attachments, native undo transaction and a separate conversation draft; stale scope callbacks cannot append')
} finally {
  drafts.dispose()
  Object.defineProperty(globalThis, 'document', { configurable: true, value: previous.document })
  Object.defineProperty(globalThis, 'HTMLTextAreaElement', { configurable: true, value: previous.HTMLTextAreaElement })
}
