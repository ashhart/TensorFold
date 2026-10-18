import * as languages from '../../../shared/languages'
import { authoredText, interfaceLanguages } from '../../lib/preferences/language-packs'
import assert from 'node:assert/strict'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'
import { normalizeUiPreferences, DEFAULT_UI_PREFERENCES, UI_SCALES } from '../../lib/preferences/model'
import { translate } from '../../lib/preferences/catalogue'

const hooks = harness(), calls: any[] = []
let finish!: () => void, fail = false
const state: any = { settings: { uiPreferences: { ...DEFAULT_UI_PREFERENCES }, unrelated: 'preserve' }, async saveSettings(settings: any, options: any) { calls.push({ settings, options }); await new Promise<void>((resolve) => { finish = resolve }); if (fail) throw new Error('Synthetic host refusal.'); state.settings = settings } }
const useStore = Object.assign((select: any) => select(state), { getState: () => state })
const { PreferencesPanel } = load('src/components/preferences/PreferencesPanel.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, '../../lib/store': { useStore }, '../../lib/preferences/model': { normalizeUiPreferences, DEFAULT_UI_PREFERENCES, UI_SCALES }, '../../lib/preferences/catalogue': { translate }, '../settings/shared': { Section: 'section' }, '../../../shared/languages': languages, '../../lib/preferences/language-packs': { authoredText, interfaceLanguages } })
let blocked = false
const render = () => hooks.render(() => PreferencesPanel({ blocked }))
const button = (label: string) => find(render(), (node) => node.type === 'button' && text(node) === label)
render(); hooks.flush()
assert.equal(calls.length, 0, 'Opening preferences never persists anything')
const coverage = 'Language packs are bundled for offline use. New replies follow your selection when the model supports that language; existing messages, code and original diagnostics stay unchanged.'
assert.ok(text(render()).includes(authoredText('en', coverage)))
assert.ok(languages.LANGUAGES.length > 10, 'Bundled major languages are selectable')
const locale = find(render(), (node) => node.type === 'select' && node.props.value === 'en')
locale.props.onChange({ target: { value: 'es' } })
assert.match(text(render()), /Idioma de la interfaz/); assert.ok(text(render()).includes(authoredText('es', coverage)))
assert.equal(state.settings.uiPreferences.locale, 'en', 'Draft preview does not cross the settings boundary before explicit save')
button('Guardar cambios').props.onClick(); button('Guardando…').props.onClick()
assert.equal(calls.length, 1); assert.equal(calls[0].settings.uiPreferences.locale, 'es'); assert.equal(calls[0].settings.unrelated, 'preserve'); assert.equal(calls[0].options.silent, true)
finish(); await settle()
assert.match(text(render()), /guardadas para este perfil/)
button('Restablecer valores').props.onClick(); assert.equal(calls.length, 1, 'Reset remains a draft until saved')
assert.match(text(render()), /Interface language/)
fail = true; button('Save changes').props.onClick(); finish(); await settle()
assert.match(text(render()), /could not be saved/); assert.equal(state.settings.uiPreferences.locale, 'es')
const stale = button('Save changes'); state.settings = { uiPreferences: { ...DEFAULT_UI_PREFERENCES }, unrelated: 'new-profile' }
stale.props.onClick(); assert.equal(calls.length, 2, 'A stale callback cannot copy an old profile draft to new settings')
blocked = true; assert.ok(find(render(), (node) => node.type === 'fieldset' && node.props.disabled))
hooks.cleanup()
console.log('Preferences panel passes English/Spanish live draft labels, explicit host-settings persistence, unrelated-setting preservation, duplicate/refusal/reset handling and stale-profile callback refusal without localStorage.')
