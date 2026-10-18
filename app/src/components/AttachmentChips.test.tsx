import { reviewedLanguageImport } from './preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import * as attachmentHelpers from '../lib/attachments'
import type { Attachment } from '../lib/types'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
let states: any[] = [], cursor = 0
const imports: Record<string, unknown> = {
  react: { useState: (initial: any) => { const index = cursor++; if (!(index in states)) states[index] = typeof initial === 'function' ? initial() : initial; return [states[index], (value: any) => { states[index] = typeof value === 'function' ? value(states[index]) : value }] }, useRef: (value: any) => ({ current: value }), useEffect() {} },
  'react/jsx-runtime': jsx, 'react-dom': { createPortal: (element: any) => element },
  '../lib/icons': { Icon: () => null }, '../lib/attachments': attachmentHelpers,
}
const loaded = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'AttachmentChips.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, imports); if (reviewed.matched) return reviewed.value;  assert.ok(id in imports, `Reviewed import ${id}`); return imports[id] }, loaded, loaded.exports)
const items: Attachment[] = [
  { id: 'a', kind: 'document', name: 'first.txt', mime: 'text/plain', status: 'ready', text: 'first' },
  { id: 'b', kind: 'document', name: 'unreadable.txt', mime: 'text/plain', status: 'error', error: 'This file cannot be read.' },
  { id: 'c', kind: 'document', name: 'third.txt', mime: 'text/plain', status: 'ready', text: 'third' },
  { id: 'r', kind: 'image', name: 'remembered.png', mime: 'image/png', status: 'needs-reselection', error: 'Select this file again.' },
]
const html = renderToStaticMarkup(loaded.exports.AttachmentChips({ items, onRemove() {}, onRetry() {}, onReselect() {} }))
for (const name of items.map((item) => item.name)) assert.ok(html.includes(name))
assert.match(html, /This file cannot be read/)
assert.match(html, /Retry unreadable.txt/)
assert.match(html, /Select remembered.png again/)
assert.equal((html.match(/data-attachment-id=/g) ?? []).length, 4)
const unsupported = renderToStaticMarkup(loaded.exports.AttachmentChips({ items: [{ id: 'i', kind: 'image', name: 'note.png', mime: 'image/png', status: 'ready', dataUrl: 'data:image/png;base64,AA' }], onRemove() {}, onExtractText() {}, extractionUnavailableReason: 'Local extraction requires the macOS app.' }))
assert.match(unsupported, /vision-capable model/)
assert.match(unsupported, /Extract text from note.png/)
assert.match(unsupported, /disabled/)
assert.match(unsupported, /Local extraction requires/)

const originalDocument = globalThis.document
Object.defineProperty(globalThis, 'document', { configurable: true, value: { body: {} } })
function all(node: any, predicate: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function button(node: any, label: string) { const found = all(node, (entry) => entry.type === 'button' && entry.props.children === label)[0]; assert.ok(found, label); return found }
const image: Attachment = { id: 'ocr', name: 'handwriting.png', mime: 'image/png', kind: 'image', dataUrl: 'data:image/png;base64,AA', status: 'review', extraction: { method: 'apple-vision', state: 'review', text: 'Fridoy', confidence: 0.9, warnings: ['Check the handwriting.'], pageCount: 2, processedPages: 1, truncated: true } }
let accepted = '', cancelled = 0, shouldReject = false
const props = { attachment: image, onAccept: async (text: string) => { if (shouldReject) throw new Error('Synthetic save failed; review retained.'); accepted = text }, onCancel: () => { cancelled++ } }
function render() { cursor = 0; return loaded.exports.AttachmentTextReview(props) }
async function settle() { for (let i = 0; i < 8; i++) await Promise.resolve() }
try {
  let node = render()
  const markup = renderToStaticMarkup(node)
  assert.match(markup, /Editable transcript/)
  assert.match(markup, /not a measure of accuracy/)
  assert.match(markup, /extraction is incomplete/)
  assert.match(markup, /No vision model or API key is needed/)
  assert.match(markup, /1 of 2 pages processed/)
  let area = all(node, (entry) => entry.type === 'textarea')[0]
  area.props.onChange({ target: { value: '' } }); node = render()
  button(node, 'Accept transcript').props.onClick(); await settle(); node = render()
  assert.equal(accepted, '')
  assert.match(renderToStaticMarkup(node), /No readable text is available/)
  area = all(node, (entry) => entry.type === 'textarea')[0]
  area.props.onChange({ target: { value: 'Friday' } }); node = render()
  shouldReject = true; button(node, 'Accept transcript').props.onClick(); await settle(); node = render()
  assert.match(renderToStaticMarkup(node), /Synthetic save failed/)
  assert.equal(all(node, (entry) => entry.type === 'textarea')[0].props.value, 'Friday')
  shouldReject = false; button(node, 'Accept transcript').props.onClick(); await settle(); assert.equal(accepted, 'Friday')
  button(render(), 'Cancel').props.onClick(); assert.equal(cancelled, 1)
  assert.equal(image.extraction?.text, 'Fridoy', 'review edits do not mutate attachment content before acceptance')
} finally { Object.defineProperty(globalThis, 'document', { configurable: true, value: originalDocument }) }
console.log('attachment rows retain three-file failures, reselect/capability actions, and explicit editable OCR acceptance/cancellation')
