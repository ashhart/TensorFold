import assert from 'node:assert/strict'
import { FULL_WORKSPACE_LAYOUT_KEY, firstFullOnboardingLayout, fullLayoutPreferenceReadable, fullLayoutStartupChat, readFullWorkspaceLayout, savedFullWorkspaceLayout, saveFullWorkspaceLayout, simpleFullLayoutActive } from './full-workspace-layout'
import { uiSlice } from './store/slices/uiSlice'

const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
const values = new Map<string, string>([['vontra.workPaneWidth', '620'], ['vontra.sidebarWidth', '280']])
const storage = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => { values.set(key, value) } }
Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true })
try {
  assert.equal(readFullWorkspaceLayout(), 'workspace');assert.equal(savedFullWorkspaceLayout(), null, 'legacy absent preference keeps the complete workspace')
  assert.equal(values.size, 2, 'reading a legacy preference does not reset any saved layout')
  for (const bad of ['unknown', '{broken', '"chat"', 'minimal']) { values.set(FULL_WORKSPACE_LAYOUT_KEY, bad);assert.equal(readFullWorkspaceLayout(), 'workspace');assert.equal(values.get(FULL_WORKSPACE_LAYOUT_KEY), bad) }
  values.delete(FULL_WORKSPACE_LAYOUT_KEY)
  assert.equal(saveFullWorkspaceLayout('chat'), true);assert.equal(readFullWorkspaceLayout(), 'chat')
  assert.equal(saveFullWorkspaceLayout('workspace'), true);assert.equal(readFullWorkspaceLayout(), 'workspace')
  assert.equal(firstFullOnboardingLayout(false, null), 'chat')
  assert.equal(firstFullOnboardingLayout(true, null), null, 'replayed setup does not choose a new layout')
  assert.equal(firstFullOnboardingLayout(false, 'workspace'), null, 'a user-selected layout is preserved even on first setup')
  assert.equal(firstFullOnboardingLayout(false, null, false), null, 'unreadable storage is not evidence of a fresh layout')
  const denied = { getItem: () => { throw new Error('Synthetic storage denied') }, setItem: () => { throw new Error('Synthetic storage denied') } }
  assert.equal(fullLayoutPreferenceReadable(denied), false);assert.equal(savedFullWorkspaceLayout(denied), null);assert.equal(saveFullWorkspaceLayout('chat', denied), false)
  assert.throws(() => saveFullWorkspaceLayout('minimal' as any))
  for (const platform of ['darwin', 'web']) assert.equal(simpleFullLayoutActive({ layout: 'chat', desktopMode: 'minimal', platform, onboarding: false }), false, 'Minimal stays the tray/server manager')
  assert.equal(simpleFullLayoutActive({ layout: 'chat', desktopMode: 'full', platform: 'darwin', onboarding: false }), true)
  assert.equal(simpleFullLayoutActive({ layout: 'chat', desktopMode: 'full', platform: 'darwin', onboarding: true }), false)
  const chats: any[] = [{ id: 'old', kind: 'chat', updatedAt: 2 }, { id: 'new', kind: 'chat', updatedAt: 3 }, { id: 'archived', kind: 'chat', archivedAt: 4, updatedAt: 4 }, { id: 'task', kind: 'task', updatedAt: 5 }]
  assert.equal(fullLayoutStartupChat('chat', 'home', chats)?.id, 'new');assert.equal(fullLayoutStartupChat('workspace', 'home', chats), null);assert.equal(fullLayoutStartupChat('chat', 'thread', chats), null)
  const settings = { desktopMode: 'full', approvalMode: 'auto', activeProviderId: 'synthetic-local', activeModel: 'synthetic-model', approvalRules: [{ id: 'existing-grant' }] }
  const threads = [{ id: 'active', kind: 'task', cwd: '/workspace', status: 'running', updatedAt: 1, timeline: [] }]
  const jobs = [{ id: 'active-job', status: 'running' }];const approvals = { active: [{ callId: 'pending', itemId: 'tool' }] }
  let state: any = { settings, threads, jobs, approvals, projects: [], view: { name: 'thread', id: 'active' }, status: { active: 'awaiting' } }
  const set = (patch: any) => { state = { ...state, ...(typeof patch === 'function' ? patch(state) : patch) } };const get = () => state
  state = { ...state, ...uiSlice(set, get), sidebarCollapsed: true, rightPanelHidden: false, chatPaneOpenId: 'prior-chat', workPaneTabs: { active: 'preview' }, workspaceDockOpen: true }
  const layout = () => ({ sidebarCollapsed: state.sidebarCollapsed, rightPanelHidden: state.rightPanelHidden, chatPaneOpenId: state.chatPaneOpenId, workPaneTabs: state.workPaneTabs, workspaceDockOpen: state.workspaceDockOpen })
  const prior = layout();state.setFullWorkspaceLayout('chat');state.setFullWorkspaceLayout('workspace')
  assert.deepEqual(layout(), prior);assert.equal(state.settings, settings);assert.equal(state.threads, threads);assert.equal(state.jobs, jobs);assert.equal(state.approvals, approvals)
  assert.equal(values.get('vontra.workPaneWidth'), '620');assert.equal(values.get('vontra.sidebarWidth'), '280')
  state.setFullWorkspaceLayout('chat');state.toggleSidebar();assert.equal(state.fullWorkspaceLayout, 'workspace');assert.deepEqual(layout(), prior, 'keyboard escape to the workspace restores rather than overwrites prior panels')
  state.setFullWorkspaceLayout('chat');state.toggleRightPanel();assert.deepEqual(layout(), prior)
  state.setFullWorkspaceLayout('chat');state.openWorkPane('terminal');assert.equal(state.fullWorkspaceLayout, 'workspace');assert.equal(state.workPaneTabs.active, 'terminal');assert.equal(state.settings, settings)
  Object.defineProperty(globalThis, 'localStorage', { value: denied, configurable: true });state.setFullWorkspaceLayout('chat')
  assert.equal(state.fullWorkspaceLayout, 'chat');assert.match(state.notice.message, /session.*could not be saved/);assert.equal(state.settings, settings, 'an unavailable preference store cannot lock the user out or mutate model/permission state')
  console.log('Full chat layout preference/legacy/minimal guards, startup choice, storage failures and active-work-preserving UI actions passed')
} finally { if (descriptor) Object.defineProperty(globalThis, 'localStorage', descriptor);else delete (globalThis as any).localStorage }
