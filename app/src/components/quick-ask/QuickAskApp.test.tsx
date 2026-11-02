import assert from 'node:assert/strict'
import path from 'node:path'
import { harness, jsx, load, nodes, settle, text } from '../reading/reading-test-harness'
import { PRESETS } from '../../../electron/agent/theme-presets'
import { quickAskDefaults, type QuickAskDraft, type QuickAskSnapshot } from '../../../shared/quick-ask'

const h = harness(), hooks = { ...h.hooks, useLayoutEffect: h.hooks.useEffect, useCallback: (fn: unknown) => h.hooks.useMemo(() => fn) }
const listeners = new Set<(event: any) => void>(), calls: any[] = []
let draft: QuickAskDraft = { revision: 3, text: 'Synthetic question', routeId: 'local', presetId: 'ask', attachments: [], web: false }
const local = { id: 'local', providerId: 'fixture', accountId: null, model: 'Fixture', label: 'Fixture', destination: 'device' as const, location: 'This computer', state: 'ready' as const, vision: 'observed' as const, binding: 'test' }
const snapshot = { preferences: quickAskDefaults(), routes: [local, { ...local, id: 'cloud', destination: 'cloud', location: 'Fixture cloud' }],
  appearance: { dark: true, palette: PRESETS.tensorfold.dark }, session: { id: 'test', draft, route: null, prompt: null, turns: [] }, routeErrors: [], shortcutError: null,
  platform: 'darwin', voice: { input: 'unavailable', output: 'unavailable', destination: '' } } as QuickAskSnapshot
let ready = true
const api = {
  onEvent(callback: (event: any) => void) { listeners.add(callback); return () => listeners.delete(callback) },
  pickWindow: async () => ({ kind: 'attached', attachments: [{ id: 'window-11111111-1111-1111-1111-111111111111-image', kind: 'image', name: 'Selected window', mime: 'image/png', dataUrl: 'data:image/png;base64,example' }, { id: 'window-11111111-1111-1111-1111-111111111111-text', kind: 'document', name: 'Visible text', mime: 'text/plain', text: 'Synthetic visible text' }] }),
  send: async (value: unknown) => { calls.push(value) }, hide: async () => { calls.push('hide') },
}
const savedDocument = globalThis.document, savedWindow = globalThis.window
globalThis.document = { body: { classList: { add() {} } }, documentElement: { classList: { toggle() {} } }, querySelector: () => null } as any
globalThis.window = { addEventListener() {}, removeEventListener() {} } as any
const { default: QuickAskApp } = load(path.join(__dirname, 'QuickAskApp.tsx'), {
  react: hooks, 'react/jsx-runtime': jsx,
  '../BrandMark': { BrandMark: 'brand' }, '../../lib/icons': { Icon: 'icon' },
  '../../lib/preferences/language-packs': { authoredText: (_locale: unknown, source: string) => source },
  '../../lib/preferences/model': { normalizeUiPreferences: (value: unknown) => value }, '../../lib/preferences/apply': { applyUiPreferences() {} }, '../../lib/theme': { applyTheme() {} },
  '../../lib/attachments': { filesToAttachments: async () => [] }, './AskPreferences': { AskPreferences: 'preferences' }, './AskDialog': { AskDialog: 'dialog' }, './AskAnswer': { AskAnswer: 'answer' },
  './useQuickAsk': { askError: String, useQuickAsk: () => ({ snapshot: ready ? snapshot : null, draft: ready ? draft : null, busy: false, saving: false, error: '', setError() {}, revision: { current: draft.revision },
    edit: (patch: Partial<QuickAskDraft>) => { draft = { ...draft, ...patch } }, action: async (work: () => Promise<unknown>) => { await work(); return true } }) },
  './useAskVoice': { useAskVoice: () => ({ listening: false, stopSpeech() {} }) }, './useAskPanelSize': { useAskPanelSize() {} },
  './AskOptions': { AskOptions: 'options' }, './AskContext': { WindowTargetIcon: 'target', AskContextChips: 'context', AskContextPreview: 'preview', AskWindowChoices: 'choices', askContextGroups: (items: unknown[]) => items.length ? [{ id: 'context', ids: items.map((item: any) => item.id) }] : [] },
  './quick-ask.css': {},
})
const render = () => h.render(() => QuickAskApp({ api }))
const button = (name: string) => nodes(render()).find(node => node.type === 'button' && (node.props['aria-label'] === name || text(node) === name))!
try {
  let tree = render(); h.flush()
  assert.match(tree.props.className, /ask-compact/)
  assert.equal(nodes(tree).find(node => node.type === 'textarea')?.props.rows, 1)
  assert.equal(nodes(tree).some(node => node.type === 'options' || node.type === 'answer'), false, 'opening starts with one compact prompt')
  await button('Choose a window').props.onClick(); await settle()
  assert.equal(draft.attachments.length, 2); assert.equal(calls.length, 0, 'choosing context never sends it')
  const context = nodes(render()).find(node => node.type === 'context')!
  context.props.remove(draft.attachments.map(item => item.id)); assert.equal(draft.attachments.length, 0)
  button('More options').props.onClick(); assert.ok(nodes(render()).some(node => node.type === 'options'))
  h.flush(); listeners.forEach(callback => callback({ kind: 'focus' }))
  assert.equal(nodes(render()).some(node => node.type === 'options'), false)
  assert.equal(draft.text, 'Synthetic question', 'reopening preserves the unsent question')
  draft = { ...draft, routeId: 'cloud' }
  button('Ask').props.onClick(); await settle()
  assert.equal(calls.length, 0, 'an off-device route requires review before any send')
  assert.ok(nodes(render()).some(node => node.type === 'dialog' && node.props.title === 'Review before sending'))
  button('Cancel').props.onClick(); button('Ask').props.onClick(); await settle()
  button('Send to this destination').props.onClick(); await settle()
  assert.deepEqual(calls, [{ revision: 3, approvedRouteId: 'cloud' }])
  ready = false; assert.equal(nodes(render()).find(node => node.type === 'textarea')?.props.disabled, true)
} finally { h.cleanup(); globalThis.document = savedDocument; globalThis.window = savedWindow }
console.log('Compact Ask starts collapsed, preserves drafts, reviews selected context, removes paired attachments and requires explicit off-device consent.')
