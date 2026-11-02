import assert from 'node:assert/strict'
import * as capabilities from '../../lib/capabilities'
import * as textarea from '../../lib/textarea-editing'
import * as terminal from '../../lib/terminal-context'
import { COMPOSER_INSERT_EVENT } from '../../lib/browser-preview'
import { ConversationDrafts } from '../../lib/conversation-drafts'
import { composerDraftKey } from '../../../shared/composer-drafts'
import { harness, load, settle } from '../reading/reading-test-harness'

function deferred<T>() { let resolve!: (value: T) => void; const promise = new Promise<T>((yes) => { resolve = yes }); return { promise, resolve } }
const hooks = harness()
hooks.hooks.useLayoutEffect = hooks.hooks.useEffect
const drafts = new ConversationDrafts(null)
const scopeA = { surface: 'chat' as const, conversationId: 'input-a', hostId: null, incognito: false }
const scopeB = { ...scopeA, conversationId: 'input-b' }
const draft = (scope: typeof scopeA) => ({ ...drafts.get(scope), scopeKey: composerDraftKey(scope), setText: (value: string | ((text: string) => string)) => drafts.update(scope, { text: typeof value === 'function' ? value(drafts.get(scope).text) : value }) })
let focused: Area | null = null
class Area extends EventTarget {
  current = ''; selectionStart = 0; selectionEnd = 0; scrollHeight = 40; style = { height: '' }
  get value() { return this.current } set value(value: string) { this.current = value }
  focus() { focused = this }
  setSelectionRange(start: number, end: number) { this.selectionStart = start; this.selectionEnd = end }
}
const globalNames = ['window', 'document', 'HTMLTextAreaElement', 'requestAnimationFrame'] as const
const descriptors = new Map(globalNames.map((name) => [name, Object.getOwnPropertyDescriptor(globalThis, name)]))
const windowEvents = new EventTarget()
const replacements = {
  window: windowEvents, HTMLTextAreaElement: Area,
  document: { execCommand(_command: string, _ui: boolean, text: string) {
    assert.ok(focused)
    focused.current = focused.value.slice(0, focused.selectionStart) + text + focused.value.slice(focused.selectionEnd)
    return true
  } },
  requestAnimationFrame: (callback: () => void) => { callback(); return 1 },
}
for (const name of globalNames) Object.defineProperty(globalThis, name, { configurable: true, value: replacements[name] })
const oldFiles = deferred<string[]>(), newFiles = deferred<string[]>(), backlog = deferred<string>()
const calls: string[] = []
let voice: { onText: (value: string) => void; onEnd: () => void } | undefined
let stops = 0
let mayCapture = false
const api = {
  getSkills: async () => [], getCommands: async () => [],
  listFiles: (cwd: string) => { calls.push(`files:${cwd}`); return cwd === '/synthetic/a' ? oldFiles.promise : newFiles.promise },
  terminalBacklog: () => { calls.push('terminal'); return backlog.promise },
  getControllerStatus: async () => ({ canCapture: mayCapture }),
  enableController: async () => ({ canCapture: false, message: 'Synthetic permission is required.' }),
  captureController: async () => { calls.push('capture'); throw new Error("Error invoking remote method 'controller': Error: Synthetic capture unavailable") },
}
const state = { pendingMention: null, composerSeed: null }
const { useComposerInput } = load('src/components/composer/useComposerInput.ts', {
  react: hooks.hooks, '../../lib/store': { useStore: Object.assign((select: any) => select(state), { setState: (patch: any) => Object.assign(state, patch) }) },
  '../../lib/ipc': { api }, '../../lib/browser-preview': { COMPOSER_INSERT_EVENT },
  '../../lib/capabilities': capabilities, '../../lib/textarea-editing': textarea, '../../lib/terminal-context': terminal,
  '../../lib/voice': { dictationSupported: () => true, startDictation: (_settings: unknown, callbacks: typeof voice) => { voice = callbacks; return () => { stops++; callbacks!.onEnd() } } },
})
const taRef = { current: new Area() }
let owner = scopeA
let work: any = { workCwd: '/synthetic/a', workspace: null, activeTerminalSession: null }
const render = () => {
  const value = hooks.render(() => useComposerInput({ mode: 'thread', thread: { id: owner.conversationId, kind: 'task' }, settings: { voice: {} }, draft: draft(owner), work, taRef, setDragOver() {}, addFiles: async () => {} }))
  hooks.flush()
  return value
}

try {
  let input = render()
  await settle()
  input.onAttach()
  input = render()
  assert.equal(calls[0], 'files:/synthetic/a')
  work = { ...work, workCwd: '/synthetic/b' }
  input = render()
  input.onAttach()
  newFiles.resolve(['current.ts'])
  await settle()
  oldFiles.resolve(['stale.ts'])
  await settle()
  input = render()
  assert.deepEqual(input.items, ['current.ts'], 'late file lists cannot replace the current workspace suggestions')

  input.toggleDictation()
  voice!.onText('Dictated in A')
  assert.match(drafts.get(scopeA).text, /Dictated in A/)
  const previousVoice = voice!
  owner = scopeB
  input = render()
  assert.equal(stops, 1, 'changing draft ownership stops the old speech session')
  previousVoice.onText('Late A fragment')
  assert.equal(drafts.get(scopeB).text, '', 'even a late speech fragment retains its original draft setter')
  assert.equal(render().dictating, false)

  work = { ...work, activeTerminalSession: { id: 'remote', title: 'Remote terminal', hostLocation: 'orchestrator' } }
  input = render()
  await input.referenceTerminalOutput()
  assert.equal(calls.includes('terminal'), false, 'a remote terminal reference cannot read a same-id host PTY')
  work = { ...work, activeTerminalSession: { id: 'local', title: 'Local terminal', hostLocation: 'local' } }
  input = render()
  const pendingReference = input.referenceTerminalOutput()
  taRef.current = new Area()
  const before = drafts.get(scopeB).text
  backlog.resolve('late output')
  await pendingReference
  assert.equal(drafts.get(scopeB).text, before, 'terminal completion cannot insert into a replaced editor')

  await input.onController()
  input = render()
  assert.equal(input.controllerModal, true)
  assert.equal(calls.includes('capture'), false, 'opening a disabled capture surface does not enable or capture automatically')
  await input.enableControllerAndCapture()
  assert.equal(render().controllerError, 'Synthetic permission is required.')
  mayCapture = true
  await input.onController()
  assert.equal(render().controllerError, 'Synthetic capture unavailable', 'provider diagnostics remain intact after bridge prefixes are removed')

  input = render()
  input.toggleDictation()
  hooks.cleanup()
  assert.equal(stops, 2, 'unmount also stops speech once')
  const previousText = drafts.get(scopeB).text
  windowEvents.dispatchEvent(Object.assign(new Event(COMPOSER_INSERT_EVENT), { detail: { text: 'Late insertion' } }))
  assert.equal(drafts.get(scopeB).text, previousText, 'unmount removes the external insertion listener')
  console.log('Composer input relocation preserves workspace fencing, scoped dictation cleanup, terminal boundaries and capture consent')
} finally {
  hooks.cleanup()
  drafts.dispose()
  for (const name of globalNames) {
    const descriptor = descriptors.get(name)
    if (descriptor) Object.defineProperty(globalThis, name, descriptor)
    else Reflect.deleteProperty(globalThis, name)
  }
}
