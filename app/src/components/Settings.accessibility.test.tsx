import assert from 'node:assert/strict'
import React from 'react'
import { find, harness, jsx, load, nodes, text } from './reading/reading-test-harness'

const hooks = harness()
const settings = {}
const state = {
  settings,
  saveSettings: async () => settings,
  purgeLocalMemory: async () => ({}),
  setView() {},
  updateStatus: null,
  checkForUpdates: async () => {},
  openUpdateDownload() {},
}
const useStore = Object.assign((selector: (value: typeof state) => unknown) => selector(state), { getState: () => state })
const groups = [
  { id: 'ai', label: 'AI & Models', sections: [
    { id: 'models', label: 'Models & Accounts', desc: 'Sources for chats and agents', icon: 'cpu' },
    { id: 'library', label: 'Model folders', desc: 'Find existing models', icon: 'folder' },
  ] },
]
const clsx = (...values: unknown[]) => values.filter(Boolean).join(' ')
const placeholder = () => null
const persistence = { edited() {}, discard() {}, persist: async () => settings }
const imports: Record<string, unknown> = {
  react: hooks.hooks,
  'react/jsx-runtime': jsx,
  clsx,
  '../lib/store': { SETTINGS_SECTION_REQUEST_EVENT: 'settings-section', useStore, consumeSettingsSection: () => null },
  '../lib/update-safety': { registerUpdatePreparationBlocker: () => () => {} },
  '../lib/settings-draft': { createSettingsDraftPersistence: () => persistence },
  '../lib/icons': { Icon: placeholder },
  './ui': { Button: (props: React.ComponentProps<'button'>) => React.createElement('button', props, props.children) },
  './settings/shared': {
    ALL_SETTINGS_SECTIONS: groups.flatMap((group) => group.sections),
    SETTINGS_GROUPS: groups,
    groupOfSection: (section: string) => groups.find((group) => group.sections.some((entry) => entry.id === section)),
  './ui': { Button: (props: React.ComponentProps<'button'>) => React.createElement('button', props, props.children) },
    Section: placeholder,
  },
}
for (const name of ['WorkspaceBackupPanel', 'ThemePanel', 'SkillsCommands', 'RuntimeSettings', 'OrchestratorSettings', 'VoiceSettings', 'ModelsSettings', 'SecuritySettings', 'ConnectorsSettings', 'PersonaSettings', 'ReadinessSettings', 'AppSettings', 'ModelFoldersSettings', 'EngineAdvancedSettings', 'PhoneSettings']) {
  const directory = name === 'WorkspaceBackupPanel' ? 'workspace-backup/' : ['ThemePanel', 'SkillsCommands'].includes(name) ? '' : 'settings/'
  imports[`./${directory}${name}`] = { [name]: placeholder }
}
for (const name of ['useProviderSettings', 'useConnectorSources', 'useBotConnectors', 'useRemoteSecurity']) imports[`./settings/${name}`] = { [name]: () => ({}) }

const originalLocalStorage = globalThis.localStorage
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: () => null, setItem() {} } })
try {
  const { Settings } = load('src/components/Settings.tsx', imports)
  const render = () => hooks.render(() => Settings({ initialSection: 'models' }))
  let tree = render()
  assert.ok(find(tree, (node) => node.type === 'main'), 'Settings content is the main landmark')
  const category = find(tree, (node) => node.type === 'button' && text(node) === 'AI & Models')
  assert.equal(category.props['aria-expanded'], true)
  const models = find(tree, (node) => node.type === 'button' && text(node) === 'Models & Accounts')
  const folders = find(tree, (node) => node.type === 'button' && text(node) === 'Model folders')
  assert.equal(models.props['aria-current'], 'page')
  assert.equal(folders.props['aria-current'], undefined)

  folders.props.onClick()
  tree = render()
  assert.equal(find(tree, (node) => node.type === 'button' && text(node) === 'Models & Accounts').props['aria-current'], undefined)
  assert.equal(find(tree, (node) => node.type === 'button' && text(node) === 'Model folders').props['aria-current'], 'page')

  category.props.onClick()
  tree = render()
  assert.equal(find(tree, (node) => node.type === 'button' && text(node) === 'AI & Models').props['aria-expanded'], false)
  assert.equal(nodes(tree).filter((node) => node.type === 'button' && text(node) === 'Model folders').length, 0)
} finally {
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: originalLocalStorage })
}
console.log('Settings exposes its main landmark, disclosure state and current section')
