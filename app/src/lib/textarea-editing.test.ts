import assert from 'node:assert/strict'
import { insertTextareaText } from './textarea-editing'

// This verifies the native editor contract, not Chromium's own Undo implementation.
// Running-app QA must still prove the real keyboard and context-menu undo stack.
const previous = { document: globalThis.document, HTMLTextAreaElement: globalThis.HTMLTextAreaElement }
class FixtureTextarea {
  private current = 'Hello selected world'
  selectionStart = 6
  selectionEnd = 14
  focused = false
  assignments = 0
  events: Event[] = []
  get value() { return this.current }
  set value(value: string) { this.assignments++; this.current = value }
  nativeEdit(value: string) { this.current = value }
  focus() { this.focused = true }
  setSelectionRange(start: number, end: number) { this.selectionStart = start; this.selectionEnd = end }
  dispatchEvent(event: Event) { this.events.push(event); return true }
}
let area: FixtureTextarea
const commands: any[] = [], undo: string[] = []
Object.defineProperty(globalThis, 'HTMLTextAreaElement', { configurable: true, value: FixtureTextarea })
Object.defineProperty(globalThis, 'document', { configurable: true, value: {
  execCommand(command: string, ui: boolean, insert: string) {
    commands.push([command, ui, insert]); undo.push(area.value)
    area.nativeEdit(area.value.slice(0, area.selectionStart) + insert + area.value.slice(area.selectionEnd))
    area.setSelectionRange(area.selectionStart + insert.length, area.selectionStart + insert.length)
    return true
  },
} })
try {
  area = new FixtureTextarea()
  assert.equal(insertTextareaText(area as any, 'replacement'), 'Hello replacement world')
  assert.equal(area.value, 'Hello replacement world')
  assert.deepEqual(commands[0], ['insertText', false, 'replacement'])
  assert.equal(area.selectionStart, 17); assert.equal(area.selectionEnd, 17)
  assert.equal(area.focused, true)
  assert.equal(area.assignments, 0, 'successful native insertion must not overwrite the value and destroy the native editor transaction')
  assert.equal(area.events.length, 0, 'successful native insertion owns its input event')
  assert.deepEqual(undo, ['Hello selected world'], 'the selected text is replaced in one native edit transaction')
  assert.equal(insertTextareaText(area as any, '😀', 6, 17), 'Hello 😀 world')
  assert.equal(area.selectionStart, 8, 'caret coordinates retain the DOM UTF-16 contract')

  ;(globalThis.document as any).execCommand = () => false
  assert.equal(insertTextareaText(area as any, 'fallback', 6, 8), 'Hello fallback world')
  assert.equal(area.value, 'Hello fallback world')
  assert.equal(area.events.length, 1); assert.equal(area.events[0].type, 'input'); assert.equal(area.events[0].bubbles, true)
  assert.equal(area.selectionStart, 14); assert.equal(area.selectionEnd, 14)
  console.log('Textarea selected-range insertion uses one native edit transaction; explicit ranges, UTF-16 caret, and fallback input contract passed')
} finally {
  Object.defineProperty(globalThis, 'document', { configurable: true, value: previous.document })
  Object.defineProperty(globalThis, 'HTMLTextAreaElement', { configurable: true, value: previous.HTMLTextAreaElement })
}
