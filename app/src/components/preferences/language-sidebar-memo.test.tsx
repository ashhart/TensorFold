import assert from 'node:assert/strict'
import React from 'react'
import clsx from 'clsx'
import { readFileSync, writeFileSync, mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { authoredText } from '../../lib/preferences/language-packs'
import { configureUiLanguageReader } from '../../lib/preferences/current-locale'
import * as threadRows from '../../lib/thread-row'
import { jsx, load, text } from '../reading/reading-test-harness'

let locale = 'en'
const restoreLanguage = configureUiLanguageReader(() => locale)
const thread = { id: 'synthetic-sidebar-thread', title: 'Cancel 你好', kind: 'task', cwd: '/synthetic/project', timeline: [], updatedAt: Date.now(), status: 'done' }
const draft = { text: '未送信 draft', attachments: [{ id: 'synthetic-attachment', name: 'Cancel.png' }] }
const state: any = { threads: [thread], projects: [], rooms: [], tickets: [], automations: [], roomStatus: {}, view: { name: 'thread', id: thread.id }, status: {}, approvals: {}, jobs: [], settings: { theme: 'system' }, terminals: {}, activeTerminalId: {}, pinnedThreadIds: [], projectTagFilter: 'all', assistantSource: null, draft }
const store = Object.assign((select: any) => select(state), { getState: () => state })
const translators = { en: (source: string) => authoredText('en', source), 'zh-Hans': (source: string) => authoredText('zh-Hans', source) }
const originalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
Object.defineProperty(globalThis, 'localStorage', { configurable: true, value: { getItem: () => null, setItem: () => { throw new Error('A language-only render must not write drafts or storage') } } })

function mount(file: string) {
  const values: any[] = [], dependencies: any[][] = []
  let cursor = 0
  const hooks = {
    ...React,
    useEffect: () => { cursor++ },
    useState: (initial: any) => { const index = cursor++; if (!(index in values)) values[index] = typeof initial === 'function' ? initial() : initial; return [values[index], () => {}] },
    useRef: (initial: any) => { const index = cursor++; return values[index] ??= { current: initial } },
    useMemo: (build: () => any, next: any[]) => {
      const index = cursor++
      if (!dependencies[index] || next.some((value, position) => value !== dependencies[index][position])) { values[index] = build(); dependencies[index] = next }
      return values[index]
    },
  }
  const { Sidebar } = load(file, {
    'react/jsx-runtime': jsx, react: hooks, clsx,
    '../lib/preferences/useAuthoredText': { useAuthoredText: () => translators[locale as keyof typeof translators] },
    '../lib/preferences/hooks': { useUiText: () => (key: string) => key },
    './search/search-events': { requestConversationSearch: () => { throw new Error('Locale change cannot search') } },
    '../lib/store': { useStore: store }, '../lib/ipc': { api: {} }, '../lib/engine': { useEngine: () => ({ snapshot: null }) }, '../lib/engine-visibility': { engineManagementVisible: () => true }, '../lib/icons': { Icon: 'icon' }, './ui': { IconButton: 'icon-button' },
    '../lib/format': { basename: (value: string) => value.split('/').at(-1) },
    '../lib/project-tags': { collectTags: () => [], hasUntaggedProjects: () => false, projectMatchesTag: () => true, UNTAGGED_FILTER: 'untagged' },
    '../lib/thread-row': threadRows,
    '../lib/thread-workspace': { effectiveThreadCwd: (value: typeof thread) => value.cwd, effectiveThreadWorkspace: () => null },
    '../lib/terminal-sessions': { terminalSessionKey: () => 'synthetic-terminal' },
    './sidebar/constants': { NEEDS_ATTENTION: new Set() }, './sidebar/primitives': { NavRow: 'nav-row', SectionTitle: 'section-title' },
    './sidebar/rows': { ThreadRow: 'thread-row' }, './sidebar/ProjectsSection': { ProjectsSection: 'projects' },
    './sidebar/MoreSection': { MoreSection: 'more' }, './sidebar/EngineChip': { EngineChip: 'engine' },
    './sidebar/UpdatePrompt': { UpdatePrompt: 'update' }, './sidebar/useBranches': { useBranches: () => ({}) },
    './work-activity/WorkActivityMenu': { WorkActivityMenu: 'work-activity' },
  })
  return () => { cursor = 0; return Sidebar() }
}

const directory = mkdtempSync(path.join(tmpdir(), 'tensorfold-language-sidebar-'))
try {
  const source = readFileSync('src/components/Sidebar.tsx', 'utf8')
  const legacyFile = path.join(directory, 'legacy-sidebar.tsx')
  assert.ok(source.includes('effectiveFilter, pinned, tx]'))
  writeFileSync(legacyFile, source.replace('effectiveFilter, pinned, tx]', 'effectiveFilter, pinned]'))
  const legacy = mount(legacyFile)
  locale = 'en'; assert.ok(text(legacy()).includes('Today'))
  locale = 'zh-Hans'; assert.ok(text(legacy()).includes('Today'), 'Negative control reproduces a memo retaining the previous language')
  const render = mount('src/components/Sidebar.tsx')
  const savedDraft = JSON.stringify(draft)
  locale = 'en'; assert.ok(text(render()).includes('Today'))
  locale = 'zh-Hans'
  const updated = text(render())
  assert.ok(updated.includes(authoredText(locale, 'Today')), updated)
  assert.ok(!updated.includes('Today'), 'The existing sidebar updates day labels without a changed thread array')
  assert.equal(state.draft, draft)
  assert.equal(JSON.stringify(draft), savedDraft)
  assert.equal(state.threads[0], thread)
  assert.equal(thread.title, 'Cancel 你好')
  console.log('Actual memoized Sidebar updates translated day groups on language-only renders and preserves drafts, attachments and source records; removing its locale dependency reproduces stale English.')
} finally {
  rmSync(directory, { recursive: true, force: true })
  restoreLanguage()
  if (originalStorage) Object.defineProperty(globalThis, 'localStorage', originalStorage)
  else delete (globalThis as any).localStorage
}
