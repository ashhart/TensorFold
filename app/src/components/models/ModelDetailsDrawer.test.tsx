import { reviewedLanguageImport } from '../preferences/language-test-fixture'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'
import { cachedBenchmarkModels } from './benchmark/benchmark-display'
import * as profiles from '../../../shared/model-profiles'
import { defaultSettings } from '../../../electron/store/defaults'
import type { EngineInstalledModel, EngineStatus } from '../../lib/types'

const jsx = { jsx: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), jsxs: (type: any, props: any, key: any) => React.createElement(type, { ...props, ...(key == null ? {} : { key }) }), Fragment: React.Fragment }
const Button = ({ children, ...props }: any) => React.createElement('button', props, children)
const Input = (props: any) => React.createElement('input', props)
class Hooks {
  slots: any[] = []; cursor = 0; pending: Array<() => void> = []; beforeEffects?: (tree: any) => void
  react = {
    useState: (initial: any) => { const index = this.cursor++; if (!(index in this.slots)) this.slots[index] = typeof initial === 'function' ? initial() : initial; return [this.slots[index], (next: any) => { this.slots[index] = typeof next === 'function' ? next(this.slots[index]) : next }] },
    useRef: (initial: any) => { const index = this.cursor++; return this.slots[index] ??= { current: initial } },
    useCallback: (fn: any, deps: any[]) => { const index = this.cursor++; const old = this.slots[index]; if (!old || deps.some((item, n) => item !== old.deps[n])) this.slots[index] = { fn, deps }; return this.slots[index].fn },
    useEffect: (fn: any, deps?: any[]) => { const index = this.cursor++; const old = this.slots[index]; if (!old || !deps || deps.some((item, n) => item !== old.deps?.[n])) this.pending.push(() => { old?.cleanup?.(); this.slots[index] = { deps, cleanup: fn() } }) },
  }
  render(fn: any, props: any) { this.cursor = 0; const tree = fn(props); this.beforeEffects?.(tree); const effects = this.pending.splice(0); for (const effect of effects) effect(); return tree }
  dispose() { for (const slot of this.slots) slot?.cleanup?.() }
}
function load(name: string, hooks: Hooks, imports: Record<string, any>) {
  const module = { exports: {} as any }
  const approved = { '../navigation/useAccessibleDialog': { hasAccessibleDialogAbove: () => false }, react: hooks.react, 'react/jsx-runtime': jsx, 'react-dom': { createPortal: (element: any) => element }, '../ui': { Button, Input }, ...imports }
  const compiled = ts.transpileModule(readFileSync(path.join(__dirname, name + '.tsx'), 'utf8'), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, esModuleInterop: true } }).outputText
  new Function('require', 'module', 'exports', compiled)((id: string) => { const reviewed = reviewedLanguageImport(id, approved); if (reviewed.matched) return reviewed.value;  assert.ok(id in approved, `Reviewed import ${id}`); return approved[id as keyof typeof approved] }, module, module.exports)
  return module.exports[name]
}
function all(node: any, predicate: (entry: any) => boolean): any[] { if (!node || typeof node !== 'object') return []; return [...(predicate(node) ? [node] : []), ...React.Children.toArray(node.props?.children).flatMap((child) => all(child, predicate))] }
function find(node: any, predicate: (entry: any) => boolean) { const found = all(node, predicate)[0]; assert.ok(found, 'Actual component node exists'); return found }
function text(node: any): string { return React.Children.toArray(node).map((child: any) => typeof child === 'object' ? text(child.props?.children) : String(child)).join('') }
function button(node: any, label: string) { return find(node, (entry) => entry.type === Button && text(entry.props.children) === label) }
function field(node: any, id: string) { return find(node, (entry) => entry.props?.id === id) }
async function settle() { for (let i = 0; i < 8; i++) await Promise.resolve() }
const model: EngineInstalledModel = { repo: 'example/test-model', title: 'Test model', family: 'qwen', modelType: 'qwen3_5', format: 'mlx', contextTokens: 32768, weightsBytes: 1024, memoryFloorBytes: 2048, fitsMemory: true, drafter: null, drafterInstalled: null }
const status: EngineStatus = { available: true, version: 'test', binaryPath: null, state: 'ready', model: model.repo, drafter: null, port: 0, startedAt: 1, readyAt: 1, message: null, logTail: [], totalMemoryBytes: 1e9, memoryLimitBytes: 1e8, contextTokens: 8192 }
const originalDocument = globalThis.document, originalWindow = globalThis.window
const docListeners = new Map<string, any>(), winListeners = new Map<string, any>()
const trigger = { isConnected: true, focused: 0, focus() { this.focused++; documentState.activeElement = this } }
const documentState: any = { body: { style: { overflow: 'scroll' } }, activeElement: trigger, addEventListener: (name: string, fn: any) => docListeners.set(name, fn), removeEventListener: (name: string, fn: any) => { if (docListeners.get(name) === fn) docListeners.delete(name) } }
Object.defineProperty(globalThis, 'document', { configurable: true, value: documentState })
Object.defineProperty(globalThis, 'window', { configurable: true, value: { addEventListener: (name: string, fn: any) => winListeners.set(name, fn), removeEventListener: (name: string, fn: any) => { if (winListeners.get(name) === fn) winListeners.delete(name) } } })
try {
  // Exercise the reusable frame's actual document/window handlers with a synthetic focus tree.
  const frameHooks = new Hooks(); const ModelsDrawer = load('ModelsDrawer', frameHooks, {})
  let closes = 0
  const props = { title: 'Test model', labelId: 'test-title', onClose: () => { closes++ }, description: 'Synthetic details.', children: React.createElement('p', null, 'Details') }
  let frame = frameHooks.render(ModelsDrawer, props)
  const first = { getClientRects: () => [{}], closest: () => null, focus() { documentState.activeElement = first } }
  const last = { ...first, focus() { documentState.activeElement = last } }
  const panel = { focused: 0, querySelectorAll: () => [first, last], contains: (target: any) => target === first || target === last || target === panel, focus() { this.focused++; documentState.activeElement = panel } }
  find(frame, (entry) => entry.props?.role === 'dialog').ref.current = panel
  assert.equal(documentState.body.style.overflow, 'hidden')
  assert.match(renderToStaticMarkup(frame), /aria-modal="true"/)
  assert.match(renderToStaticMarkup(frame), /aria-describedby="test-title-description"/)
  function key(extra: any) { const event = { key: '', metaKey: false, ctrlKey: false, altKey: false, shiftKey: false, prevented: 0, stopped: 0, preventDefault() { this.prevented++ }, stopPropagation() { this.stopped++ }, ...extra }; winListeners.get('keydown')(event); return event }
  documentState.activeElement = last; assert.equal(key({ key: 'Tab' }).prevented, 1); assert.equal(documentState.activeElement, first)
  documentState.activeElement = first; assert.equal(key({ key: 'Tab', shiftKey: true }).prevented, 1); assert.equal(documentState.activeElement, last)
  docListeners.get('focusin')({ target: {} }); assert.equal(documentState.activeElement, panel)
  key({ key: 'Tab', shiftKey: true }); assert.equal(documentState.activeElement, last, 'Initial dialog focus wraps backwards to its last control')
  panel.focus(); key({ key: 'Tab' }); assert.equal(documentState.activeElement, first, 'Initial dialog focus moves forwards to its first control')
  assert.equal(key({ key: 'n', metaKey: true }).prevented, 1, 'Global navigation cannot silently unmount dirty editor')
  for (const editKey of ['a', 'c', 'x', 'v', 'z', 'ArrowLeft', 'Backspace']) assert.equal(key({ key: editKey, metaKey: true, target: { closest: () => ({}) } }).prevented, 0, `Native text editing remains available: ${editKey}`)
  assert.equal(key({ key: 'ArrowLeft', metaKey: true }).prevented, 1, 'History navigation is blocked outside an editable field')
  key({ key: 'Escape' }); assert.equal(closes, 1)
  frame = frameHooks.render(ModelsDrawer, { ...props, closeDisabled: true })
  key({ key: 'Escape' }); assert.equal(closes, 1, 'Saving blocks Escape')
  frame.props.onMouseDown({ target: frame, currentTarget: frame }); assert.equal(closes, 1, 'Saving blocks backdrop dismissal')
  frame = frameHooks.render(ModelsDrawer, props)
  frame.props.onMouseDown({ target: frame, currentTarget: frame }); assert.equal(closes, 2)
  frameHooks.dispose(); assert.equal(documentState.body.style.overflow, 'scroll'); assert.equal(trigger.focused, 1); assert.equal(winListeners.size, 0); assert.equal(docListeners.size, 0)

  // Real drawer navigation and its dirty/busy exit policy; no modal code is duplicated.
  const DetailsFrame = (next: any) => React.createElement('div', null, next.children), Editor = () => null, Preview = () => null
  const detailsHooks = new Hooks(); const ModelDetailsDrawer = load('ModelDetailsDrawer', detailsHooks, { '../../lib/engine': { formatBytes: (n: number) => `${n} bytes` }, '../../lib/turn-stats-line': { repoDisplayName: (repo: string) => repo.split('/').at(-1) }, './ModelsDrawer': { ModelsDrawer: DetailsFrame }, './ModelProfileEditor': { ModelProfileEditor: Editor }, './ModelPreviewPanel': { ModelPreviewPanel: Preview }, './benchmark/benchmark-display': { cachedBenchmarkModels }, './qualification/ModelQualificationPanel': { ModelQualificationPanel: () => null } })
  let benchmarks = 0
  let detailsClosed = 0, dirty = false, busy = false, discarded = 0, saveOK = false, saves = 0
  const detailsProps = { model, status, blocked: false, onClose: () => { detailsClosed++ }, onBenchmark: () => { benchmarks++ } }
  const renderDetails = (extra = {}) => detailsHooks.render(ModelDetailsDrawer, { ...detailsProps, ...extra })
  let promptScrolled = 0, promptFocused = 0, settingsScrolled = 0, settingsFocused = 0
  const settingField = { scrollIntoView() { settingsScrolled++ }, focus() { settingsFocused++; documentState.activeElement = settingField } }
  const promptButton = { focus() { promptFocused++; documentState.activeElement = promptButton } }
  detailsHooks.beforeEffects = (tree) => {
    const prompt = all(tree, (entry) => entry.props?.role === 'alert')[0]
    if (prompt?.ref) prompt.ref.current = { scrollIntoView() { promptScrolled++ }, querySelector: () => promptButton }
    field(tree, 'model-panel-settings').ref.current = { querySelector: () => settingField }
  }
  let details = renderDetails()
  const edit = find(details, (entry) => entry.type === Editor)
  edit.props.controllerRef.current = { hasChanges: () => dirty, isBusy: () => busy, save: async () => { saves++; return saveOK }, discard: () => { discarded++; dirty = false } }
  const preview = find(details, (entry) => entry.type === Preview)
  assert.equal(preview.props.fixedModel, model.repo); assert.deepEqual(preview.props.models, [model])
  assert.equal(field(details, 'model-panel-overview').props.hidden, false)
  assert.equal(field(details, 'model-panel-settings').props.hidden, true)
  field(details, 'model-tab-settings').props.onClick(); details = renderDetails()
  assert.equal(field(details, 'model-panel-settings').props.hidden, false)
  let focusedTab = ''
  find(details, (entry) => entry.props?.role === 'tablist').ref.current = { querySelector: (selector: string) => ({ focus: () => { focusedTab = selector } }) }
  field(details, 'model-tab-settings').props.onKeyDown({ key: 'End', preventDefault() {} }); details = renderDetails(); assert.equal(field(details, 'model-tab-performance').props['aria-selected'], true); assert.match(focusedTab, /performance/)
  field(details, 'model-tab-performance').props.onKeyDown({ key: 'Home', preventDefault() {} }); details = renderDetails(); assert.equal(field(details, 'model-tab-overview').props['aria-selected'], true)
  const editorIdentity = find(details, (entry) => entry.type === Editor).type
  field(details, 'model-tab-performance').props.onClick(); details = renderDetails()
  assert.equal(find(details, (entry) => entry.type === Editor).type, editorIdentity, 'Form remains mounted while changing tabs')
  assert.equal(field(details, 'model-panel-performance').props.hidden, false)
  button(details, 'Open repeatable benchmarks').props.onClick(); assert.equal(benchmarks, 1, 'performance opens the separate repeatable benchmark workflow')
  dirty = true; edit.props.onStateChange({ dirty, busy }); details = renderDetails(); assert.equal(button(details, 'Open repeatable benchmarks').props.disabled, true); button(details, 'Open repeatable benchmarks').props.onClick(); assert.equal(benchmarks, 1, 'dirty profile changes cannot be discarded through the benchmark link'); details.props.onClose(); details = renderDetails()
  assert.equal(detailsClosed, 0); assert.ok(button(details, 'Keep editing')); assert.equal(promptScrolled, 1); assert.equal(promptFocused, 1); assert.equal(documentState.activeElement, promptButton, 'Dirty close scrolls and focuses its visible prompt')
  button(details, 'Keep editing').props.onClick(); details = renderDetails(); assert.equal(all(details, (entry) => entry.props?.role === 'alert').length, 0); assert.equal(field(details, 'model-panel-settings').props.hidden, false); assert.equal(settingsScrolled, 1); assert.equal(settingsFocused, 1); assert.equal(documentState.activeElement, settingField, 'Keep editing restores a visible Settings field instead of body focus')
  details.props.onClose(); details = renderDetails(); button(details, 'Save & close').props.onClick(); await settle(); details = renderDetails()
  assert.equal(detailsClosed, 0); assert.equal(saves, 1); assert.equal(field(details, 'model-panel-settings').props.hidden, false, 'Failed save/error stays on visible Settings')
  button(details, 'Discard changes').props.onClick(); assert.equal(discarded, 1); assert.equal(detailsClosed, 1)
  button(details, 'Keep editing').props.onClick(); details = renderDetails() // Host normally unmounts after close; reset the synthetic exit prompt.
  dirty = true; details = renderDetails({ blocked: true }); details.props.onClose(); details = renderDetails({ blocked: true }); assert.equal(button(details, 'Save & close').props.disabled, true)
  busy = true; edit.props.onStateChange({ dirty, busy }); details = renderDetails(); assert.equal(details.props.closeDisabled, true)
  details.props.onClose(); assert.equal(detailsClosed, 1)
  const previousTab = field(details, 'model-tab-settings').props['aria-selected']; field(details, 'model-tab-overview').props.onClick(); details = renderDetails(); assert.equal(field(details, 'model-tab-settings').props['aria-selected'], previousTab)
  busy = false; saveOK = true; edit.props.onStateChange({ dirty, busy }); details = renderDetails(); button(details, 'Save & close').props.onClick(); await settle(); assert.equal(detailsClosed, 2)
  const markup = renderToStaticMarkup(field(renderDetails(), 'model-panel-overview'))
  assert.match(markup, /32,768 tokens/); assert.match(markup, /8,192 tokens/); assert.match(markup, /Running context window/)
  assert.doesNotMatch(renderToStaticMarkup(field(renderDetails({ status: { ...status, model: 'example/other-model' } }), 'model-panel-overview')), /8,192 tokens/, 'Another model running does not claim this context')
  detailsHooks.dispose()

  // Embedded editor owns the actual validated/durable-save draft; controller never closes on a rejected save.
  const profileHooks = new Hooks(); let settings = defaultSettings(), saveMode = 'reject', hostSaves = 0, refreshed = 0, editorClosed = 0
  const store = Object.assign((selector: any) => selector({ settings }), { getState: () => ({ settings }), setState: (next: any) => { settings = next.settings } })
  const ModelProfileEditor = load('ModelProfileEditor', profileHooks, { '../../../shared/model-profiles': profiles, '../../lib/store': { useStore: store }, '../../lib/engine': { refreshEngine: async () => { refreshed++ } }, '../../lib/ipc': { api: { engineSaveModelProfiles: async (input: any) => { hostSaves++; if (saveMode === 'reject') throw new Error('Synthetic save rejected'); return profiles.updateEngineModelProfiles(settings, input) } } } })
  const control: any = { current: null }; let editorState: any
  const editorProps = { model, embedded: true, controllerRef: control, onStateChange: (next: any) => { editorState = next }, onClose: () => { editorClosed++ } }
  const renderEditor = (extra = {}) => profileHooks.render(ModelProfileEditor, { ...editorProps, ...extra })
  let editor = renderEditor()
  assert.equal(editor.props.role, undefined); assert.equal(control.current.hasChanges(), false)
  field(editor, 'model-profile-contextTokens').props.onChange({ currentTarget: { value: '' } }); editor = renderEditor()
  assert.equal(await control.current.save(), false); editor = renderEditor(); assert.match(renderToStaticMarkup(editor), /Context window must/); assert.equal(hostSaves, 0)
  field(editor, 'model-profile-contextTokens').props.onChange({ currentTarget: { value: '8192' } }); editor = renderEditor(); assert.doesNotMatch(renderToStaticMarkup(editor), /Context window must/, 'Correcting a number clears its stale validation immediately')
  assert.equal(field(editor, 'model-profile-contextTokens').props.value, 8192); assert.equal(hostSaves, 0, 'Clearing stale validation does not save or apply the draft')
  field(editor, 'model-profile-contextTokens').props.onChange({ currentTarget: { value: '4096' } }); editor = renderEditor()
  assert.equal(control.current.hasChanges(), true); assert.equal(editorState.dirty, true)
  assert.equal(await control.current.save(), false); editor = renderEditor(); assert.match(renderToStaticMarkup(editor), /Synthetic save rejected/); assert.equal(field(editor, 'model-profile-contextTokens').props.value, 4096)
  button(editor, 'Cancel changes').props.onClick(); editor = renderEditor(); assert.equal(field(editor, 'model-profile-contextTokens').props.value, 32768); assert.equal(editorClosed, 0); assert.equal(control.current.hasChanges(), false)
  field(editor, 'model-profile-temperature').props.onChange({ currentTarget: { value: '0.7' } }); editor = renderEditor(); saveMode = 'success'; assert.equal(await control.current.save(), true); editor = renderEditor()
  assert.equal(settings.engineModelProfiles?.[0].profiles[0].temperature, 0.7); assert.equal(control.current.hasChanges(), false); assert.equal(refreshed, 1)
  field(editor, 'model-profile-temperature').props.onChange({ currentTarget: { value: '0.9' } }); editor = renderEditor(); button(editor, 'Cancel changes').props.onClick(); editor = renderEditor(); assert.equal(field(editor, 'model-profile-temperature').props.value, 0.7, 'Cancel restores last host-accepted save')
  const saveCount = hostSaves; editor = renderEditor({ blocked: true }); assert.equal(await control.current.save(), false); assert.equal(hostSaves, saveCount)
  profileHooks.dispose(); assert.equal(control.current, null)
  profileHooks.slots = []
  let standalonePromptScroll = 0, standaloneFieldFocus = 0
  profileHooks.beforeEffects = (tree) => {
    const prompt = all(tree, (entry) => entry.props?.role === 'alert' && entry.ref)[0]
    if (prompt) prompt.ref.current = { scrollIntoView() { standalonePromptScroll++ }, querySelector: () => promptButton }
    const fields = all(tree, (entry) => entry.ref && typeof entry.props?.className === 'string' && entry.props.className.startsWith('scroll-thin min-h-0'))[0]
    if (fields) fields.ref.current = { querySelector: () => ({ scrollIntoView() {}, focus() { standaloneFieldFocus++ } }) }
  }
  editor = renderEditor({ embedded: false })
  field(editor, 'model-profile-temperature').props.onChange({ currentTarget: { value: '1.0' } }); editor = renderEditor({ embedded: false })
  button(editor, 'Close').props.onClick(); editor = renderEditor({ embedded: false }); assert.equal(editorClosed, 0); assert.ok(button(editor, 'Keep editing'), 'Standalone close also protects unsaved profiles'); assert.equal(standalonePromptScroll, 1)
  button(editor, 'Keep editing').props.onClick(); editor = renderEditor({ embedded: false }); assert.equal(all(editor, (entry) => entry.props?.role === 'alert').length, 0); assert.equal(standaloneFieldFocus, 1)
  button(editor, 'Close').props.onClick(); editor = renderEditor({ embedded: false }); button(editor, 'Discard changes').props.onClick(); assert.equal(editorClosed, 1)
  profileHooks.dispose()
  console.log('Actual model drawer handlers: tabs, exact preview identity, dirty save/discard/keep, rejected-save draft, blocked/busy policy, focus trap/restore, backdrop/Escape and navigation isolation passed')
} finally {
  Object.defineProperty(globalThis, 'document', { configurable: true, value: originalDocument })
  Object.defineProperty(globalThis, 'window', { configurable: true, value: originalWindow })
}
