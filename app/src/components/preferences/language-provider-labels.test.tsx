import assert from 'node:assert/strict'
import clsx from 'clsx'
import * as fit from '../../../shared/model-fit'
import * as cli from '../../lib/cli-tool-display'
import { authoredText } from '../../lib/preferences/language-packs'
import { configureUiLanguageReader } from '../../lib/preferences/current-locale'
import { find, harness, jsx, load, settle, text } from '../reading/reading-test-harness'
let locale = 'zh-Hans'
const tx = (source: string, params = {}) => authoredText(locale, source, params)
const state = { settings: { uiPreferences: { locale } } }
const useStore = Object.assign((select: any) => select(state), { getState: () => state })
const restore = configureUiLanguageReader(() => locale)
const hooks = harness()
try {
  const { ModelHelperDefaultsPanel } = load('src/components/models/ModelHelperDefaultsPanel.tsx', {
    'react/jsx-runtime': jsx, '../../lib/preferences/useAuthoredText': { useAuthoredText: () => { const lang = locale; return (source: string, params = {}) => authoredText(lang, source, params) } }, '../../lib/store': { useStore },
    './qualification/useModelQualifications': { useModelQualifications: () => ({ reports: [], error: '' }) },
    '../../../shared/provider-trust': { isExternalProvider: () => false }, '../../../shared/model-fit': fit, '../settings/shared': { Section: 'section' },
  })
  const defaults = ModelHelperDefaultsPanel({ draft: { providers: [] }, patch() {} })
  assert.ok(text(defaults).includes(tx('Conversation naming')))
  assert.doesNotMatch(text(defaults), /Conversation naming|Document facts|Source answers/)
  const { ProviderToolsPanel } = load('src/components/settings/ProviderToolsPanel.tsx', {
    clsx,
    '../../lib/icons': { Icon: () => null }, '../ui': { Button: 'button', Field: 'label', Input: 'input' },
    react: hooks.hooks, 'react/jsx-runtime': jsx, '../../lib/preferences/useAuthoredText': { useAuthoredText: () => { const lang = locale; return (source: string, params = {}) => authoredText(lang, source, params) } }, '../../lib/cli-tool-display': cli,
    '../../lib/ipc': { api: { listCliTools: async () => [
      { id: 'codex', name: 'Settings', command: 'codex', installed: true, state: 'installed', version: '1.2.3' },
      { id: 'cursor', name: 'Original tool', command: 'cursor', installed: false, state: 'missing' },
      { id: 'synthetic', name: 'Original error', command: 'synthetic', installed: true, state: 'broken', problem: 'Settings / Cancel / diagnostic data' },
    ] } }, './shared': { Spinner: () => null },
  })
  const props: any = { ctx: { draft: { providers: [] }, commit() {} } }
  const render = () => hooks.render(() => ProviderToolsPanel(props))
  render(); hooks.flush(); await settle()
  const rendered = text(render())
  assert.doesNotMatch(rendered, /Runs Codex CLI sessions| installed| missing| broken/)
  assert.ok(rendered.includes(tx('Missing')))
  assert.equal(cli.cliToolStatusText({ command: null, installed: true }).text, tx('Built-in tools'))
  assert.ok(rendered.includes('Settings / Cancel / diagnostic data'), 'Original probe errors remain data')
  assert.ok(find(render(), node => node.type === 'span' && node.props?.title === 'Settings'), 'A tool named after a UI action retains its name')
  locale = 'pl'; state.settings.uiPreferences.locale = locale
  assert.ok(text(render()).includes(tx('Missing')), 'Summary and status labels update after a profile-language change')
  locale = 'en'
  assert.equal(cli.cliToolStatusText({ command: 'tool', installed: false }).text, 'Missing')
} finally { hooks.cleanup(); restore() }
console.log('Actual helper titles and provider-tool rows translate fixed authored metadata and states while retaining model/tool IDs, names, versions and original diagnostics')
