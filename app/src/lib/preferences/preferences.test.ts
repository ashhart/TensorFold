import assert from 'node:assert/strict'
import { applyUiPreferences } from './apply'
import { english, spanish, translate } from './catalogue'
import { DEFAULT_UI_PREFERENCES, normalizeUiPreferences } from './model'

assert.deepEqual(Object.keys(spanish).sort(), Object.keys(english).sort(), 'Every supported catalogue key has a Spanish entry')
for (const key of Object.keys(english)) {
  const en = translate('en', key, { percent: 130 }), es = translate('es', key, { percent: 130 })
  assert.ok(en.trim()); assert.ok(es.trim()); assert.doesNotMatch(es, /^preferences\.|^common\.|\{percent\}/)
}
assert.equal(translate('unknown', 'common.save'), 'Save changes')
assert.equal(translate('es', 'unknown.internal.key'), 'Text unavailable', 'Unknown keys fall back to readable English instead of exposing implementation keys')
assert.equal(translate('es', 'common.save'), 'Guardar cambios')
assert.match(translate('es', 'preferences.scale', { percent: 130 }), /130/)
assert.doesNotMatch(translate('en', 'preferences.scale', { percent: '\u202E<script>' }), /\u202e/)
assert.doesNotMatch(translate('en', 'preferences.scale'), /\{percent\}/)
assert.match(translate('en', 'preferences.coverage'), /other screens use English/)
assert.deepEqual(normalizeUiPreferences(null), DEFAULT_UI_PREFERENCES)
for (const unsafe of [undefined, false, [], 'rtl', { locale: 'ar', direction: 'javascript:', chatScale: '2; color:red', codeScale: Infinity, density: '__proto__' }, { chatScale: -1, codeScale: 999 }, { chatScale: '1.5', codeScale: NaN }]) assert.deepEqual(normalizeUiPreferences(unsafe), DEFAULT_UI_PREFERENCES)
assert.deepEqual(normalizeUiPreferences({ locale: 'es', direction: 'rtl', chatScale: 1.5, codeScale: 1.3, density: 'compact', ignored: 'x' }), { locale: 'es', direction: 'rtl', chatScale: 1.5, codeScale: 1.3, density: 'compact' })

class ElementFixture {
  attrs = new Map<string, string>()
  isConnected = true
  nodeType = 1
  children: ElementFixture[] = []
  constructor(readonly kind: 'root' | 'input' | 'message' | 'code' | 'other') {}
  getAttribute(name: string) { return this.attrs.get(name) ?? null }
  setAttribute(name: string, value: string) { this.attrs.set(name, value) }
  removeAttribute(name: string) { this.attrs.delete(name) }
  matches(selector: string) { return selector.includes('.prose-tf') ? this.kind === 'message' || this.kind === 'input' : this.kind === 'code' }
  querySelectorAll(selector: string): ElementFixture[] { return this.children.flatMap((child) => [...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector)]) }
}
const properties = new Map<string, { value: string; priority: string }>()
const style = { getPropertyValue: (name: string) => properties.get(name)?.value ?? '', getPropertyPriority: (name: string) => properties.get(name)?.priority ?? '', setProperty: (name: string, value: string, priority = '') => { properties.set(name, { value, priority }) }, removeProperty: (name: string) => { properties.delete(name) } }
const root = Object.assign(new ElementFixture('root'), { style })
root.setAttribute('lang', 'fr'); root.setAttribute('dir', 'ltr'); style.setProperty('--theme-private-token', 'keep'); style.setProperty('--tf-chat-scale', '0.9', 'important')
const message = new ElementFixture('message'), input = new ElementFixture('input'), code = new ElementFixture('code')
input.setAttribute('dir', 'rtl'); root.children.push(message, input, code)
let callback!: (records: any[]) => void, disconnected = 0
class ObserverFixture { constructor(receive: any) { callback = receive } observe() {} disconnect() { disconnected++ } }
const doc: any = { documentElement: root, querySelectorAll: (selector: string) => root.querySelectorAll(selector), defaultView: { MutationObserver: ObserverFixture } }
const cleanup = applyUiPreferences({ locale: 'es', direction: 'rtl', chatScale: 1.5, codeScale: 1.15, density: 'compact' }, doc)
assert.equal(root.getAttribute('lang'), 'es'); assert.equal(root.getAttribute('dir'), 'rtl'); assert.equal(root.getAttribute('data-ui-density'), 'compact')
assert.equal(style.getPropertyValue('--tf-chat-scale'), '1.5'); assert.equal(style.getPropertyValue('--tf-code-scale'), '1.15')
assert.equal(style.getPropertyValue('--tf-row-height'), '2rem'); assert.equal(style.getPropertyValue('--theme-private-token'), 'keep')
assert.equal(message.getAttribute('dir'), 'auto'); assert.equal(input.getAttribute('dir'), 'auto'); assert.equal(code.getAttribute('dir'), 'ltr')
const lateInput = new ElementFixture('input'); root.children.push(lateInput); callback([{ addedNodes: [lateInput] }]); assert.equal(lateInput.getAttribute('dir'), 'auto')
cleanup()
assert.equal(disconnected, 1); assert.equal(root.getAttribute('lang'), 'fr'); assert.equal(root.getAttribute('dir'), 'ltr'); assert.equal(root.getAttribute('data-ui-density'), null)
assert.equal(style.getPropertyValue('--tf-chat-scale'), '0.9'); assert.equal(style.getPropertyPriority('--tf-chat-scale'), 'important'); assert.equal(style.getPropertyValue('--tf-code-scale'), '')
assert.equal(input.getAttribute('dir'), 'rtl'); assert.equal(code.getAttribute('dir'), null); assert.equal(lateInput.getAttribute('dir'), null)
const reset = applyUiPreferences(undefined, doc)
assert.equal(root.getAttribute('lang'), 'en'); assert.equal(root.getAttribute('dir'), 'ltr'); assert.equal(style.getPropertyValue('--tf-chat-scale'), '1'); assert.equal(style.getPropertyValue('--tf-row-height'), '2.5rem')
reset()
console.log('Preferences pass complete English/Spanish key coverage, readable fallback/interpolation, hostile-value normalization, independent scales/density, mixed-direction inputs/code, dynamic DOM projection and exact cleanup/profile-default reset.')
