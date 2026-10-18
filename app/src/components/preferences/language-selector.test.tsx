import assert from 'node:assert/strict'
import * as languages from '../../../shared/languages'
import { authoredText, interfaceLanguages } from '../../lib/preferences/language-packs'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'
const hooks = harness(), calls: any[] = []
let fail = false, finish!: () => void
const providers = [{ id: 'synthetic-provider', models: ['synthetic-model'], apiKey: '[set]' }]
const initial = { providers, activeProviderId: 'synthetic-provider', activeModel: 'synthetic-model', uiPreferences: { locale: 'en', direction: 'auto', chatScale: 1.3, codeScale: 1.15, density: 'compact' }, unrelated: 'keep' }
const state: any = { settings: initial, async saveSettings(next: any, options: any) { calls.push({ next, options }); await new Promise<void>(resolve => { finish = resolve }); if (fail) throw new Error('Synthetic persistence refusal'); state.settings = next } }
const useStore = Object.assign((select: any) => select(state), { getState: () => state })
const { useAuthoredText } = load('src/lib/preferences/useAuthoredText.ts', { react: { useCallback: (fn: any) => fn }, '../store': { useStore }, './language-packs': { authoredText } })
const { LanguageQuickSelect } = load('src/components/preferences/LanguageQuickSelect.tsx', { react: hooks.hooks, 'react/jsx-runtime': jsx, '../../../shared/languages': languages, '../../lib/store': { useStore }, '../../lib/preferences/useAuthoredText': { useAuthoredText }, '../../lib/preferences/language-packs': { interfaceLanguages } })
let blocked = false
const render = () => hooks.render(() => LanguageQuickSelect({ blocked }))
render(); assert.equal(calls.length, 0)
assert.deepEqual(interfaceLanguages().map(language => language.id), languages.LANGUAGE_IDS, 'Every advertised language has a complete bundled pack')
const select = find(render(), node => node.type === 'select')
assert.equal(select.props.value, 'en')
assert.ok(find(render(), node => node.type === 'option' && node.props.value === 'zh-Hans'))
select.props.onChange({ target: { value: 'zh-Hans' } }); select.props.onChange({ target: { value: 'ja' } })
assert.equal(calls.length, 1, 'Pending saves cannot overlap even through a stale DOM handler')
assert.equal(calls[0].next.providers, providers); assert.equal(calls[0].next.activeModel, initial.activeModel)
assert.deepEqual(calls[0].next.uiPreferences, { ...initial.uiPreferences, locale: 'zh-Hans' }); assert.equal(calls[0].options.silent, true)
finish(); await settle()
assert.equal(find(render(), node => node.type === 'select').props.value, 'zh-Hans')
assert.ok(text(render()).includes(authoredText('zh-Hans', 'Language')))
fail = true
find(render(), node => node.type === 'select').props.onChange({ target: { value: 'ru' } }); finish(); await settle()
assert.equal(state.settings.uiPreferences.locale, 'zh-Hans', 'Failed saves retain the last authoritative profile language')
assert.ok(find(render(), node => node.props?.role === 'alert'))
const stale = find(render(), node => node.type === 'select')
state.settings = { ...initial, unrelated: 'second-profile' }
stale.props.onChange({ target: { value: 'pl' } }); assert.equal(calls.length, 2, 'An old profile callback cannot write a new profile')
blocked = true; assert.equal(find(render(), node => node.type === 'select').props.disabled, true)
hooks.cleanup()
console.log('Onboarding language selection preserves providers/model/scales, never starts model work, isolates profile changes and handles duplicate, blocked and refused settings saves')
