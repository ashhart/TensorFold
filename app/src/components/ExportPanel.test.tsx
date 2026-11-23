import assert from 'node:assert/strict'
import { find, harness, jsx, load, settle, text } from './reading/reading-test-harness'

type Request = { threadId: string; options: { mode: string; redact: boolean } }
const hook = harness()
hook.hooks.useCallback = (callback: unknown) => hook.hooks.useRef(callback).current
const builds: (Request & { finish: (label: string) => void; fail: () => void })[] = []
const saves: (Request & { finish: () => void })[] = []
const copies: { json: string; finish: () => void }[] = []
const windowFixture = new EventTarget()
Object.defineProperty(globalThis, 'window', { configurable: true, value: windowFixture })
Object.defineProperty(globalThis, 'navigator', { configurable: true, value: {
  clipboard: { writeText: (json: string) => new Promise<void>((finish) => copies.push({ json, finish })) },
} })

const api = {
  buildExport: (threadId: string, options: Request['options']) => new Promise((resolve, reject) => {
    builds.push({
      threadId,
      options,
      finish: (label) => resolve({ json: label, bundle: {
        thread: { title: label },
        stats: { messages: 0, toolCalls: 0, diffs: 0, denied: 0, errors: 0 },
        redactionReport: [],
      } }),
      fail: () => reject(new Error('synthetic build failure')),
    })
  }),
  saveExport: (threadId: string, options: Request['options']) => new Promise((resolve) => {
    saves.push({ threadId, options, finish: () => resolve({ saved: true }) })
  }),
}
const { ExportPanel } = load('src/components/ExportPanel.tsx', {
  react: hook.hooks, 'react/jsx-runtime': jsx,
  clsx: (...classes: string[]) => classes.filter(Boolean).join(' '),
  '../lib/ipc': { api }, '../lib/icons': { Icon: () => null },
})
const render = () => hook.render(() => ExportPanel())
const open = () => windowFixture.dispatchEvent(new CustomEvent('vontra:export', { detail: { threadId: 'test-run' } }))
const button = (tree: ReturnType<typeof render>, label: string) => find(tree, (node) => node.type === 'button' && text(node) === label)
const setRedact = (checked: boolean) => find(render(), (node) => node.type === 'input').props.onChange({ target: { checked } })
const preview = (tree: ReturnType<typeof render>) => text(find(tree, (node) => node.type === 'pre'))
const assertUnavailable = (tree: ReturnType<typeof render>) => {
  assert.equal(button(tree, 'Copy JSON').props.disabled, true)
  assert.equal(button(tree, 'Save…').props.disabled, true)
}

render()
hook.flush()
open()
render()
hook.flush()
builds[0].finish('first preview')
await settle()
const ready = render()
assert.equal(preview(ready), 'first preview')
assert.equal(button(ready, 'Copy JSON').props.disabled, false)
assert.equal(button(ready, 'Save…').props.disabled, false)
const copy = button(ready, 'Copy JSON').props.onClick
const save = button(ready, 'Save…').props.onClick
const copying = copy()
const saving = save()
assert.equal(copies[0].json, 'first preview')
assert.deepEqual(saves[0].options, { mode: 'full', redact: true })

setRedact(false)
const staleCopy = copy()
const staleSave = save()
const pending = render()
assert.equal(preview(pending), 'Building…')
assert.equal(text(pending).includes('first preview'), false)
assertUnavailable(pending)
await Promise.all([staleCopy, staleSave])
assert.equal(copies.length, 1)
assert.equal(saves.length, 1)
hook.flush()
assert.deepEqual(builds[1].options, { mode: 'full', redact: false })
setRedact(true)
render()
hook.flush()
assert.deepEqual(builds[2].options, { mode: 'full', redact: true })
copies[0].finish()
saves[0].finish()
await Promise.all([copying, saving])
assert.equal(text(render()).includes('Saved'), false)
assertUnavailable(render())
builds[2].finish('replacement preview')
await settle()
builds[1].fail()
await settle()
assert.equal(preview(render()), 'replacement preview')

button(render(), 'Share packet').props.onClick()
render()
hook.flush()
assert.deepEqual(builds[3].options, { mode: 'share', redact: true })
builds[3].fail()
await settle()
assert.equal(preview(render()), 'Nothing to export.')
assertUnavailable(render())
button(render(), 'Full run').props.onClick()
render()
hook.flush()
button(render(), 'esc').props.onClick()
assert.equal(render(), null)
hook.flush()
open()
assertUnavailable(render())
hook.flush()
assert.equal(builds[5].threadId, 'test-run')
builds[4].finish('closed preview')
await settle()
assert.equal(preview(render()), 'Building…')
assertUnavailable(render())
builds[5].finish('reopened preview')
await settle()
const reopened = render()
assert.equal(preview(reopened), 'reopened preview')
const reopenedCopy = button(reopened, 'Copy JSON').props.onClick()
const reopenedSave = button(reopened, 'Save…').props.onClick()
assert.equal(copies[1].json, 'reopened preview')
assert.equal(saves[1].threadId, 'test-run')
assert.deepEqual(saves[1].options, { mode: 'full', redact: true })
copies[1].finish()
saves[1].finish()
await Promise.all([reopenedCopy, reopenedSave])
assert.ok(button(render(), 'Copied'))
assert.equal(text(render()).includes('Saved'), true)
hook.cleanup()
