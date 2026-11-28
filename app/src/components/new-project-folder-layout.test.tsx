import assert from 'node:assert/strict'
import clsx from 'clsx'
import { find, harness, jsx, load, text } from './reading/reading-test-harness'

const hooks = harness()
const state = {
  newProjectOpen: true,
  newProjectIntent: null,
  newProject: null,
  projects: [],
  settings: { remoteOrchestrators: [], showGitInitPrompt: true },
  closeNewProject() {},
  saveSettings() {},
}
const useStore = Object.assign(<T,>(select: (current: typeof state) => T) => select(state), { getState: () => state })
const controls = load('src/components/ui.tsx', {
  react: hooks.hooks,
  'react/jsx-runtime': jsx,
  'react-dom': { createPortal: (node: unknown) => node },
  clsx,
  '../lib/icons': { Icon: 'icon-fixture' },
})
const { NewProjectModal } = load('src/components/NewProjectModal.tsx', {
  react: hooks.hooks,
  'react/jsx-runtime': jsx,
  clsx,
  './OriginalDiagnostic': { OriginalDiagnostic: 'original-diagnostic' },
  '../lib/preferences/useAuthoredText': { useAuthoredText: () => (value: string) => value },
  '../lib/store': { useStore },
  '../lib/execution-target': { resolveExecutionTarget: () => ({ hostId: null }) },
  '../lib/ipc': { api: {} },
  '../lib/format': { shortPath: (value: string) => value },
  '../lib/icons': { Icon: 'icon-fixture' },
  './ui': controls,
  '../lib/project-tags': {
    TAG_COLORS: [], TAG_PRESETS: [], addTag: () => [], collectTags: () => [],
    normalizeTag: (value: string) => value, tagTone: () => '', toggleTag: () => [],
  },
})

let tree = hooks.render(() => NewProjectModal())
hooks.flush()
tree = hooks.render(() => NewProjectModal())
const browse = find(tree, (node) => node.type === controls.Button && text(node) === 'Browse')
const input = find(tree, (node) => node.type === controls.Input && node.props.readOnly)
assert.ok(browse && input, 'The real New Project modal renders its folder controls')
const browseClasses = browse.type(browse.props).props.className as string
const inputClasses = input.type(input.props).props.className as string
assert.match(browseClasses, /(?:^|\s)whitespace-nowrap(?:\s|$)/, 'UI preference overflow wrapping cannot split the Browse label')
assert.match(browseClasses, /(?:^|\s)shrink-0(?:\s|$)/, 'Browse retains its single-line intrinsic width')
assert.match(inputClasses, /(?:^|\s)min-w-0(?:\s|$)/, 'The folder input, rather than Browse, absorbs available shrinking')

hooks.cleanup()
console.log('New Project folder controls preserve a one-line Browse button and allow the folder input to shrink.')
