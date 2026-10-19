import assert from 'node:assert/strict'
import clsx from 'clsx'
import { harness, jsx, load } from './reading/reading-test-harness'

// Render the actual headers with inert adapters: chrome must not depend on new IPC.
const api = { platform: 'linux' }
const state = { view: { name: 'home' }, threads: [], projects: [], sidebarCollapsed: true, settings: null, status: {} }
const useStore = Object.assign((select: any) => select(state), { getState: () => state })
const fixture = () => null
const hooks = harness()
const common = { react: hooks.hooks, 'react/jsx-runtime': jsx, clsx }
const { TitleBar } = load('src/components/TitleBar.tsx', {
  ...common, '../lib/store': { useStore }, '../lib/ipc': { api }, '../lib/icons': { Icon: fixture },
  '../lib/bot-labels': { localExecutionSource: (platform: string) => platform === 'darwin' ? 'This Mac' : 'Local' },
  './ThreadTrustBadge': { ThreadTrustBadge: fixture }, './CortheonBadge': { CortheonBadge: fixture },
  './full-workspace/ChatWorkspaceHeader': { ChatWorkspaceHeader: fixture }, './full-workspace/FullLayoutSwitch': { FullLayoutSwitch: fixture },
})
const { ChatWorkspaceHeader } = load('src/components/full-workspace/ChatWorkspaceHeader.tsx', {
  ...common, '../../lib/store': { useStore }, '../../lib/ipc': { api }, '../../lib/icons': { Icon: fixture },
  '../../lib/preferences/hooks': { useUiText: () => (key: string) => key },
  '../../lib/full-workspace-activity': { fullWorkspaceActivity: () => [] }, '../search/search-events': { requestConversationSearch: fixture },
  '../ui': { Button: fixture, Menu: fixture, MenuItem: fixture }, '../ThreadTrustBadge': { ThreadTrustBadge: fixture },
  '../CortheonBadge': { CortheonBadge: fixture }, './FullLayoutSwitch': { FullLayoutSwitch: fixture },
})
const minimalHooks = harness()
const { MinimalWorkspace } = load('src/components/MinimalWorkspace.tsx', {
  ...common, react: minimalHooks.hooks, '../lib/store': { useStore }, '../lib/ipc': { api }, './OriginalDiagnostic': { OriginalDiagnostic: fixture },
  '../lib/desktop-mode': { changeDesktopMode: fixture }, '../lib/engine': { useEngine: () => ({ snapshot: null }) },
  '../lib/learning': { useLearning: () => ({ snapshot: null }) }, './BrandMark': { BrandMark: fixture },
  './ui': { Button: fixture }, './Models': { Models: fixture }, './models/EngineRuntimePanel': { EngineRuntimePanel: fixture },
  './models/EngineApiPanel': { EngineApiPanel: fixture },
})
function hasClass(node: any, value: string) { return node.props.className.split(/\s+/).includes(value) }
for (const platform of ['linux', 'darwin', 'win32', 'browser', 'web-remote']) {
  api.platform = platform
  state.sidebarCollapsed = true
  const title = hooks.render(() => TitleBar({}))
  const chat = hooks.render(() => ChatWorkspaceHeader({ title: 'Fixture', thread: null })).props.children[0]
  const minimal = minimalHooks.render(() => MinimalWorkspace({ target: 'models', onTarget: fixture })).props.children[0]
  for (const node of [title, chat, minimal]) {
    assert.equal(hasClass(node, 'pl-[96px]'), platform !== 'linux', `${platform}: reserve traffic-light space only on unchanged platforms`)
  }
  assert.equal(hasClass(title, 'pl-3'), platform === 'linux')
  assert.equal(hasClass(chat, 'pl-3'), platform === 'linux')
  assert.equal(hasClass(minimal, 'pl-4'), platform === 'linux')
  state.sidebarCollapsed = false
  assert.equal(hasClass(hooks.render(() => TitleBar({})), 'pl-3'), true, 'expanded-sidebar header keeps normal padding')
}
console.log('window chrome renderer spacing tests passed')
