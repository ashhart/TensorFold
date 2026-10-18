import assert from 'node:assert/strict'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { find, harness, jsx, load, nodes, settle, text } from '../reading/reading-test-harness'

const { DocumentSourceViewer } = load('src/components/documents/DocumentSourceViewer.tsx', { react: { useEffect() {}, useRef: () => ({ current: null }) }, 'react/jsx-runtime': jsx, '../ui': { Button: 'button' }, './DocumentEvidence': { documentLocationLabel: () => 'Verified lines 1–2' }, '../OriginalDiagnostic': { OriginalDiagnostic: ({ children }: any) => React.createElement('span', {}, children) }, '../../lib/preferences/current-locale': { currentIntlLocale: () => 'en' } })
const hostile = '<script>neverExecute()</script><img src="https://example.invalid/track">'
const result: any = { status: 'ready', passage: { sourceLabel: 'Synthetic source', relativePath: 'source.html', revision: 'a'.repeat(64), location: {}, text: hostile }, verifiedAt: 1, original: { kind: 'text', text: hostile, truncated: true }, extractedText: hostile, extractedTruncated: true, parser: 'plain-text' }
const rendered = renderToStaticMarkup(React.createElement(DocumentSourceViewer, { result, busy: false, onClose() {} }))
assert.ok(rendered.includes('&lt;script&gt;neverExecute()&lt;/script&gt;'))
assert.doesNotMatch(rendered, /<script>|<img|<iframe|<object/)
assert.ok(rendered.includes('Original preview is truncated') && rendered.includes('Extracted preview is truncated'))
assert.ok(rendered.includes('Current SHA-256 revision:'))
const changed = renderToStaticMarkup(React.createElement(DocumentSourceViewer, { result: { status: 'changed', message: 'Original changed' }, busy: false, onClose() {} }))
assert.ok(changed.includes('Original changed')); assert.ok(!changed.includes('neverExecute'))

const hooks = harness(), selected: string[][] = []
let release!: () => void
const client: any = { documentLibraryList: async () => ({ sources: [{ id: 'one', label: 'First collection', enabled: true }, { id: 'two', label: 'Second collection', enabled: true }, { id: 'disabled', label: 'Unavailable collection', enabled: false }] }) }
const { CollectionPicker } = load('src/components/documents/CollectionPicker.tsx', { react: hooks.hooks, 'react-dom': { createPortal: (node: any) => node }, '../navigation/useAccessibleDialog': { useAccessibleDialog() {} }, 'react/jsx-runtime': jsx, '../../lib/ipc': { api: client }, '../OriginalDiagnostic': { OriginalDiagnostic: 'diagnostic' }, '../ui': { Button: 'button' } })
let threadId = 'first', sourceIds: string[] | undefined = []
const props = () => ({ threadId, sourceIds, client, onChange: async (ids: string[]) => { selected.push(ids); await new Promise<void>(done => { release = done }) } })
const priorDocument = (globalThis as any).document
;(globalThis as any).document = { body: {} }
const render = () => hooks.render(() => CollectionPicker(props()))
let tree = render(); hooks.flush(); await find(tree, node => node.type === 'button').props.onClick(); await settle(); tree = render(); hooks.flush()
assert.ok(text(tree).includes('No selection means this chat cannot retrieve document collections'))
const checkboxes = nodes(tree).filter(node => node.type === 'input' && node.props.type === 'checkbox')
assert.ok(checkboxes.every(node => node.props.checked === false)); assert.equal(checkboxes[2].props.disabled, true)
checkboxes[1].props.onChange({ target: { checked: true } }); tree = render()
const save = find(tree, node => node.type === 'button' && text(node) === 'Save collections')
const firstSave = save.props.onClick(); const duplicate = save.props.onClick(); await settle()
assert.deepEqual(selected, [['two']], 'Rapid repeated Save invokes one policy mutation and retains [] versus explicit selection semantics')
threadId = 'second'; sourceIds = ['one']; tree = render(); hooks.flush(); release(); await firstSave; await duplicate; await settle(); tree = render()
assert.equal(find(tree, node => node.type === 'section'), undefined, 'An old chat callback cannot reopen or update the new chat picker')
assert.ok(text(tree).includes('Knowledge: 1 collections'))
hooks.cleanup(); (globalThis as any).document = priorDocument
console.log('PASS document UI: escaped original/extracted content, visible truncation and revision, changed-source refusal, empty collection policy, individual selection, disabled collections, duplicate save and stale-chat completion')
