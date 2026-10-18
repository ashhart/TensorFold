import { fixtureText, reviewedLanguageImport } from './preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import * as helpers from '../lib/attachments'
import { ConversationDrafts } from '../lib/conversation-drafts'
import { composerDraftKey } from '../../shared/composer-drafts'
import { composerFileSources, forgetComposerFileSources } from '../lib/composer-file-sources'
import type { Attachment } from '../lib/types'
import { isComposerSendKey } from '../lib/composer-keyboard'

// Compile the production handlers verbatim so the tests exercise Composer's ownership and
// acceptance wiring, while the filesystem/OCR/host transport boundaries remain controlled.
const source = ts.createSourceFile('Composer.tsx', readFileSync(path.join(__dirname, 'Composer.tsx'), 'utf8'), ts.ScriptTarget.ES2022, true, ts.ScriptKind.TSX)
const composer = source.statements.find((node): node is ts.FunctionDeclaration => ts.isFunctionDeclaration(node) && node.name?.text === 'Composer')!
assert.ok(composer?.body)
const printer = ts.createPrinter()
const operationDeclarations = source.statements.filter((node) => ts.isFunctionDeclaration(node) && node.name?.text === 'attachmentOperation' || ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === 'attachmentOperations'))
assert.equal(operationDeclarations.length, 2)
const operationCode = ts.transpileModule(`${operationDeclarations.map((node) => printer.printNode(ts.EmitHint.Unspecified, node, source)).join('\n')}\nreturn attachmentOperation;`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
const attachmentOperation = new Function(operationCode)()
function handlers(names: string[], environment: Record<string, any>): any {
  const declarations = names.map((name) => {
    const statement = composer.body!.statements.find((node) => ts.isVariableStatement(node) && node.declarationList.declarations.some((declaration) => ts.isIdentifier(declaration.name) && declaration.name.text === name))
    assert.ok(statement, `Production handler ${name} exists`)
    return printer.printNode(ts.EmitHint.Unspecified, statement, source)
  }).join('\n')
  const code = ts.transpileModule(`${declarations}\nreturn { ${names.join(',')} };`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText
  return new Function('tx', ...Object.keys(environment), code)(fixtureText, ...Object.values(environment))
}
function deferred<T>() { let resolve!: (value: T) => void, reject!: (reason: Error) => void; const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no }); return { promise, resolve, reject } }
const scope = (id: string) => ({ surface: 'chat' as const, conversationId: id, hostId: null, incognito: false })
const a = scope('composer-a'), b = scope('composer-b')
const drafts = new ConversationDrafts(null)
function owner(target: typeof a) {
  return { ...drafts.get(target), scopeKey: composerDraftKey(target), capture: () => drafts.capture(target), clearAccepted: (token: any) => drafts.clearAccepted(token) }
}
function attachmentSetter(target: typeof a) { return (value: Attachment[] | ((old: Attachment[]) => Attachment[])) => drafts.update(target, { attachments: typeof value === 'function' ? value(drafts.get(target).attachments) : value }) }
const setA = attachmentSetter(a)
const originalGlobals = { document: globalThis.document }
let notices: any[] = []
const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
let modalSlots: any[] = [], modalCursor = 0
const modalImports: Record<string, any> = {
  react: { useState(initial: any) { const slot = modalCursor++; if (!(slot in modalSlots)) modalSlots[slot] = typeof initial === 'function' ? initial() : initial; return [modalSlots[slot], (value: any) => { modalSlots[slot] = typeof value === 'function' ? value(modalSlots[slot]) : value }] }, useRef: (initial: any) => ({ current: initial }), useEffect() {} },
  'react/jsx-runtime': jsx, 'react-dom': { createPortal: (element: any) => element }, '../lib/icons': { Icon: () => null }, '../lib/attachments': helpers,
}
const modalModule = { exports: {} as any }
new Function('require', 'module', 'exports', ts.transpileModule(readFileSync(path.join(__dirname, 'AttachmentChips.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText)((id: string) => { const reviewed = reviewedLanguageImport(id, modalImports); if (reviewed.matched) return reviewed.value;  assert.ok(id in modalImports); return modalImports[id] }, modalModule, modalModule.exports)
function descendants(node: any, match: (node: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(match(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => descendants(child, match))] }
function modalButton(node: any, label: string) { const found = descendants(node, (entry) => entry.type === 'button' && entry.props.children === label)[0]; assert.ok(found, label); return found }
function reviewElement(environment: Record<string, any>) {
  let review: ts.JsxSelfClosingElement | undefined
  const visit = (node: ts.Node) => { if (ts.isJsxSelfClosingElement(node) && node.tagName.getText(source) === 'AttachmentTextReview') review = node; ts.forEachChild(node, visit) }
  visit(composer); assert.ok(review)
  const code = ts.transpileModule(`const result = ${printer.printNode(ts.EmitHint.Expression, review, source)}; return result;`, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX } }).outputText
  return new Function('require', 'exports', ...Object.keys(environment), code)((id: string) => { assert.equal(id, 'react/jsx-runtime'); return jsx }, {}, ...Object.values(environment))
}
function ingestion(target = a, api: any = {}) {
  return handlers(['sourceFiles', 'addFiles', 'extractFile'], {
    draft: owner(target), setAttachments: attachmentSetter(target), attachments: drafts.get(target).attachments,
    composerFileSources, attachmentOperation, ...helpers, textCapability: { available: true, maxBytes: 20 * 1024 * 1024 }, api,
  })
}
async function settle() { for (let i = 0; i < 20; i++) await Promise.resolve() }
try {
  forgetComposerFileSources()
  const first = deferred<string>(), third = deferred<string>()
  function textFile(name: string, read: () => Promise<string>) { return { name, type: 'text/plain', size: 10, lastModified: 1, slice: () => ({ text: read }) } as unknown as File }
  const selected = [textFile('first.txt', () => first.promise), textFile('unreadable.txt', async () => { throw new Error('Synthetic unreadable file') }), textFile('third.txt', () => third.promise)]
  drafts.update(a, { text: 'Unfinished A' }); drafts.update(b, { text: 'Unfinished B' })
  const pending = ingestion().addFiles(selected)
  assert.deepEqual(drafts.get(a).attachments.map((item) => item.name), selected.map((file) => file.name), 'the actual handler publishes every selected filename before any read completes')
  assert.ok(drafts.get(a).attachments.every((item) => item.status === 'processing'))
  drafts.update(b, { text: 'Continue typing in B after picker A' })
  first.resolve('first contents'); third.resolve('third contents'); await pending
  assert.deepEqual(drafts.get(a).attachments.map((item) => item.status), ['ready', 'error', 'ready'])
  assert.match(drafts.get(a).attachments[1].error!, /Synthetic unreadable file/)
  assert.equal(drafts.get(b).attachments.length, 0, 'late completion in A cannot leak files into B')
  assert.equal(drafts.get(b).text, 'Continue typing in B after picker A')

  // A pending OCR response must not overwrite a different file selected into the same row.
  const old: Attachment = { id: 'reselected', name: 'old-note.pdf', kind: 'document', mime: 'application/pdf', status: 'error' }
  setA([old])
  const oldFile = { name: old.name, size: 5, arrayBuffer: async () => new Uint8Array([1, 2, 3]).buffer } as File
  composerFileSources(composerDraftKey(a)).set(old.id, oldFile)
  const response = deferred<any>()
  let requests = 0
  const extracting = ingestion(a, { extractAttachmentText: async () => { requests++; return response.promise } }).extractFile(old.id)
  await settle(); assert.equal(requests, 1)
  await ingestion().addFiles([textFile('replacement.txt', async () => 'New file contents')], old.id)
  assert.equal(drafts.get(a).attachments[0].name, 'replacement.txt')
  response.resolve({ ok: true, text: 'Old OCR must be discarded', confidence: 0.5, warnings: ['Check handwriting.'], pageCount: 1, processedPages: 1 })
  await extracting
  assert.equal(drafts.get(a).attachments[0].name, 'replacement.txt', 'a stale OCR response must not restore the file that the user replaced')
  assert.equal(drafts.get(a).attachments[0].text, 'New file contents')

  // Replacement before file bytes finish reading must not even submit stale bytes to OCR.
  setA([old]); const reading = deferred<ArrayBuffer>()
  composerFileSources(composerDraftKey(a)).set(old.id, { ...oldFile, arrayBuffer: () => reading.promise } as File)
  const beforeRead = ingestion(a, { extractAttachmentText: async () => { throw new Error('Stale bytes reached OCR') } }).extractFile(old.id)
  await ingestion().addFiles([textFile('replacement-before-read.txt', async () => 'Current bytes')], old.id)
  reading.resolve(new Uint8Array([4, 5]).buffer); await beforeRead
  assert.equal(drafts.get(a).attachments[0].name, 'replacement-before-read.txt')

  // Two requests for the same original file settle out of order; the newest request owns it.
  setA([old]); composerFileSources(composerDraftKey(a)).set(old.id, oldFile)
  const firstOcr = deferred<any>(), secondOcr = deferred<any>(); let extractionCount = 0
  const duplicate = ingestion(a, { extractAttachmentText: async () => (++extractionCount === 1 ? firstOcr : secondOcr).promise })
  const oldRequest = duplicate.extractFile(old.id); await settle()
  const newRequest = duplicate.extractFile(old.id); await settle()
  secondOcr.resolve({ ok: true, text: 'Newest extraction', warnings: [] }); await newRequest
  firstOcr.resolve({ ok: true, text: 'Outdated extraction', warnings: [] }); await oldRequest
  assert.equal(drafts.get(a).attachments[0].extraction?.text, 'Newest extraction')

  // Render the real review dialog with Composer's real JSX accept/cancel callbacks.
  Object.defineProperty(globalThis, 'document', { configurable: true, value: { body: {} } })
  let reviewId: string | null = old.id
  const beforeReview = structuredClone(drafts.get(a).attachments)
  const element = reviewElement({ reviewingAttachment: drafts.get(a).attachments[0], draft: owner(a), AttachmentTextReview: modalModule.exports.AttachmentTextReview, setReviewId: (id: string | null) => { reviewId = id }, setAttachments: setA, ...helpers })
  const renderReview = () => { modalCursor = 0; return element.type(element.props) }
  let modal = renderReview()
  const markup = renderToStaticMarkup(modal)
  assert.match(markup, /Review extracted text/); assert.match(markup, /No vision model or API key is needed/)
  assert.match(markup, /recognition can omit or misread text/)
  descendants(modal, (entry) => entry.type === 'textarea')[0].props.onChange({ target: { value: 'Cancelled correction' } })
  modal = renderReview(); modalButton(modal, 'Cancel').props.onClick()
  assert.equal(reviewId, null); assert.deepEqual(drafts.get(a).attachments, beforeReview, 'Composer Cancel changes neither extraction nor model-ready content')
  modalSlots = []; reviewId = old.id; modal = renderReview()
  descendants(modal, (entry) => entry.type === 'textarea')[0].props.onChange({ target: { value: 'Manually corrected handwriting' } })
  modal = renderReview(); modalButton(modal, 'Accept transcript').props.onClick(); await settle()
  assert.equal(reviewId, null)
  assert.equal(drafts.get(a).attachments[0].status, 'ready')
  assert.equal(drafts.get(a).attachments[0].extraction?.text, 'Manually corrected handwriting')
  assert.match(drafts.get(a).attachments[0].text!, /Recognition may contain errors/)
  assert.equal(drafts.get(b).attachments.length, 0)

  // Actual dispatch preserves blocked files and clears only accepted, unchanged owner input.
  const calls: any[] = []
  const admitted = deferred<boolean>()
  const dispatch = (sendImpl = (...args: any[]) => { calls.push(args); return admitted.promise }) => handlers(['dispatch'], {
    submittingRef: { current: false }, quoteEditingRef: { current: false }, setSubmitting() {}, draft: owner(a), attachments: drafts.get(a).attachments,
    isAttachmentReady: helpers.isAttachmentReady, useStore: { setState: (value: any) => notices.push(value) },
    provider: { id: 'local', models: ['fixture'] }, mode: 'thread', thread: { id: 'composer-a' },
    send: sendImpl,
  }).dispatch
  setA([{ ...old, status: 'review', extraction: { method: 'apple-vision', state: 'review', text: 'Unreviewed', confidence: null, warnings: [], pageCount: 1, processedPages: 1, truncated: false } }])
  await dispatch()('Unfinished A')
  assert.equal(calls.length, 0)
  assert.match(notices.at(-1).notice.message, /draft is preserved/)
  const accepted = helpers.acceptAttachmentExtraction(drafts.get(a).attachments[0], 'Reviewed and corrected')
  setA([accepted]); drafts.update(a, { text: 'Accepted A' })
  const sending = dispatch()('Accepted A')
  assert.equal(calls[0][2], 'composer-a', 'Composer passes the captured conversation ID instead of relying on whichever view is current later')
  assert.match(calls[0][1][0].text, /Reviewed and corrected/)
  assert.doesNotMatch(calls[0][1][0].text, /Unreviewed/)
  drafts.update(a, { text: 'Newer A while sending' }); drafts.update(b, { text: 'B while sending A' })
  admitted.resolve(true); await sending
  assert.equal(drafts.get(a).text, 'Newer A while sending')
  assert.equal(drafts.get(b).text, 'B while sending A')
  const untouched = drafts.capture(a)
  await dispatch(async () => false)('Newer A while sending')
  assert.deepEqual(drafts.capture(a), untouched, 'rejected host admission leaves the exact text and files untouched')
  await dispatch(async () => true)('Newer A while sending')
  assert.equal(drafts.get(a).text, ''); assert.equal(drafts.get(a).attachments.length, 0)
  assert.equal(drafts.get(b).text, 'B while sending A')

  const keyboardCalls: string[] = []
  const keyDown = handlers(['onKeyDown'], {
    isComposerSendKey, settings: {}, pick: null, pickItems: [], mention: null, capabilityItems: [], items: [], busy: false, thread: { id: 'composer-a' },
    submit: () => keyboardCalls.push('submit'), queueMessage: () => keyboardCalls.push('queue'),
  }).onKeyDown
  for (const key of ['a', 'c', 'x', 'v', 'z', 'Z']) {
    let prevented = false
    keyDown({ key, metaKey: true, ctrlKey: true, shiftKey: key === 'Z', nativeEvent: {}, preventDefault: () => { prevented = true } })
    assert.equal(prevented, false, `native editing shortcut ${key} is not intercepted by Composer`)
  }
  keyDown({ key: 'Enter', nativeEvent: { isComposing: true }, preventDefault() { throw new Error('IME confirmation intercepted') } })
  keyDown({ key: 'Enter', keyCode: 229, nativeEvent: {}, preventDefault() { throw new Error('IME confirmation intercepted') } })
  keyDown({ key: 'Enter', repeat: true, nativeEvent: {}, preventDefault() { throw new Error('Repeated submit intercepted') } })
  keyDown({ key: 'Enter', shiftKey: true, nativeEvent: {}, preventDefault() { throw new Error('Newline intercepted') } })
  assert.equal(keyboardCalls.length, 0)
  keyDown({ key: 'Enter', nativeEvent: {}, preventDefault() {} })
  assert.deepEqual(keyboardCalls, ['submit'])
  console.log('Actual Composer handlers retain every file/error, scope async work, reject stale OCR, block unreviewed input, and clear only accepted owner drafts')
} finally {
  Object.defineProperty(globalThis, 'document', { configurable: true, value: originalGlobals.document })
  forgetComposerFileSources(); drafts.dispose()
}
