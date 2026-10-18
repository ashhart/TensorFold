import assert from 'node:assert/strict'
import { find, harness, jsx, load, settle, text } from './reading-test-harness'

const hooks = harness()
const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document')
Object.defineProperty(globalThis, 'document', { configurable: true, value: { body: {} } })
const requests: any[] = [], saves: any[] = []
let resolvePreview!: (value: any) => void, resolveSave!: (value: any) => void
const api = {
  readableExportPreview(request: any) { requests.push(request); return new Promise((resolve) => { resolvePreview = resolve }) },
  readableExportSave(request: any) { saves.push(request); return new Promise((resolve) => { resolveSave = resolve }) },
}
const { ReadableExportDialog } = load('src/components/reading/ReadableExportDialog.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node }, '../navigation/useAccessibleDialog': { useAccessibleDialog() {} } })
const props: any = { threadId: 'synthetic-thread', messages: [{ id: 'u1', label: 'User message 1' }, { id: 'a1', label: 'Assistant message 2' }], api, onClose() {} }
const render = () => hooks.render(() => ReadableExportDialog(props))
const button = (label: string) => find(render(), (node) => node.type === 'button' && text(node) === label)
const preview: any = { approvalId: 'synthetic-one-use', format: 'markdown', scope: 'conversation', bytes: 120, messageCount: 2, branchLabel: 'Active conversation branch', omissions: [{ kind: 'attachments', label: 'Attachment files are excluded.', count: 1 }], redactions: [], messages: [{ ordinal: 1, role: 'user', text: 'Synthetic redacted question.' }] }
render(); hooks.flush()
assert.equal(button('Save reviewed export').props.disabled, true); assert.equal(requests.length, 0, 'Opening the dialog never saves or automatically exports')
assert.ok(find(render(), (node) => node.props.role === 'dialog'))
button('Preview export').props.onClick(); button('Building preview…').props.onClick()
assert.equal(requests.length, 1, 'Duplicate previews are suppressed synchronously')
assert.deepEqual(requests[0], { threadId: 'synthetic-thread', format: 'markdown', scope: 'conversation', branch: 'active' })
resolvePreview(preview); await settle()
assert.equal(button('Save reviewed export').props.disabled, false); assert.match(text(render()), /Attachment files are excluded/)
const format = find(render(), (node) => node.type === 'select' && node.props.value === 'markdown')
format.props.onChange({ target: { value: 'html' } })
assert.equal(button('Save reviewed export').props.disabled, true, 'Changing format invalidates the reviewed approval')
button('Preview export').props.onClick(); resolvePreview({ ...preview, format: 'html' }); await settle()
button('Save reviewed export').props.onClick(); button('Saving…').props.onClick()
assert.equal(saves.length, 1); assert.deepEqual(saves[0], { approvalId: 'synthetic-one-use' })
resolveSave({ saved: false, canceled: true }); await settle()
assert.match(text(render()), /Save canceled/); assert.equal(button('Save reviewed export').props.disabled, true)
button('Preview export').props.onClick()
props.threadId = 'different-thread'; render(); hooks.flush()
resolvePreview(preview); await settle()
assert.equal(button('Save reviewed export').props.disabled, true, 'Late preview from another thread cannot be approved')
assert.doesNotMatch(text(render()), /Synthetic redacted question/)
hooks.cleanup()
if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument); else Reflect.deleteProperty(globalThis, 'document')
console.log('Readable export dialog requires explicit format/scope preview, displays omissions, blocks duplicate requests, consumes canceled approvals and discards stale thread results.')
