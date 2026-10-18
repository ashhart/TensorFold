import assert from 'node:assert/strict'
import * as fit from '../../../shared/model-fit'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'

const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document')
Object.defineProperty(globalThis, 'document', { configurable: true, value: { body: {} } })
const hooks = harness(), renames: any[] = [], helperRequests: any[] = [], clipboard: string[] = []
const route = { providerId: 'local', accountId: null, model: 'synthetic-helper' }
const state: any = { settings: { modelHelperDefaults: { naming: route, document_extraction: route } }, renameThread: async (id: string, title: string) => { renames.push({ id, title }) } }
const api: any = { modelHelperRun: async (input: any) => { helperRequests.push(input); return { route, text: 'Reviewed synthetic title', complete: true, sources: [] } }, modelHelperCancel: async () => {}, writeClipboard: async (value: string) => { clipboard.push(value); return { ok: false } } }
const { ModelHelperDialog } = load('src/components/models/ModelHelperDialog.tsx', {
  react: hooks.hooks, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (node: any) => node }, '../../lib/ipc': { api }, '../../lib/store': { useStore: Object.assign((select: any) => select(state), { getState: () => state }) }, '../navigation/useAccessibleDialog': { useAccessibleDialog() {} }, '../documents/DocumentEvidenceButtons': { DocumentEvidenceButtons: 'sources-fixture' },
})
const props: any = { thread: { id: 'synthetic-thread', title: 'Original title' }, onClose() {} }
const render = () => hooks.render(() => ModelHelperDialog(props))
const button = (label: string) => find(render(), (node) => node.type === 'button' && text(node) === label)
render(); hooks.flush()
assert.equal(helperRequests.length, 0, 'Opening helpers never invokes a model')
find(render(), (node) => node.type === 'select').props.onChange({ target: { value: 'naming' } })
find(render(), (node) => node.type === 'textarea').props.onChange({ target: { value: 'Only the explicitly supplied title source.' } })
button('Run local helper').props.onClick(); await settle()
assert.equal(helperRequests.length, 1); assert.equal(helperRequests[0].text, 'Only the explicitly supplied title source.')
assert.equal(renames.length, 0, 'A completed naming job never changes a title automatically')
button('Use as conversation title').props.onClick(); await settle()
assert.deepEqual(renames, [{ id: props.thread.id, title: 'Reviewed synthetic title' }])
button('Copy result').props.onClick(); await settle()
assert.equal(clipboard[0], 'Reviewed synthetic title')
assert.doesNotMatch(text(render()), /Helper result copied/, 'A refused clipboard write cannot announce successful copying')
assert.match(text(render()), /Copy failed/)
hooks.cleanup()
if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument); else Reflect.deleteProperty(globalThis, 'document')
console.log('Actual helper dialog invokes only on click, applies naming only on a separate click and announces clipboard refusal truthfully.')

const { ModelHelperDefaultsPanel } = load('src/components/models/ModelHelperDefaultsPanel.tsx', {
  'react/jsx-runtime': jsx,
  './qualification/useModelQualifications': { useModelQualifications: () => ({ reports: [
    { ...route, key: JSON.stringify([route.providerId, null, route.model]), probeRevision: 1, revision: 1, checkedAt: 1, checks: { text: { state: 'observed' } } },
    { ...route, key: JSON.stringify([route.providerId, null, route.model]), probeRevision: 1, revision: 2, checkedAt: 2, checks: { text: { state: 'failed' } } },
  ], error: '' }) },
  '../../../shared/provider-trust': { isExternalProvider: () => false }, '../../../shared/model-fit': fit, '../settings/shared': { Section: 'section' },
})
const defaults = ModelHelperDefaultsPanel({ draft: { providers: [{ id: route.providerId, name: 'Synthetic local' }], modelHelperDefaults: { naming: route } }, patch() {} })
const naming = find(defaults, (node) => node.type === 'select' && node.props['aria-label'] === 'Conversation naming helper model')
assert.match(text(naming), /Configured helper unavailable/, 'A newer failed text qualification must revoke an older observed helper choice')
console.log('Actual helper defaults picker uses current qualification evidence instead of a superseded successful check.')
