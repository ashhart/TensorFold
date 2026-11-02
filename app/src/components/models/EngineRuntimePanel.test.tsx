import assert from 'node:assert/strict'
import path from 'node:path'
import clsx from 'clsx'
import type { EngineSnapshot, PullProgress } from '../../lib/types'
import * as format from '../../lib/engine-format'
import { engineReadiness } from '../../lib/engine-readiness'
import { engineBaseUrl, formatTokenRate } from '../../lib/engine-workspace'
import { repoDisplayName } from '../../lib/turn-stats-line'
import { flowTextFor } from '../../lib/preferences/flow-catalogue'
import * as downloads from './download-display'
import { harness, jsx, load, nodes, settle, text } from '../reading/reading-test-harness'

const fixture: EngineSnapshot = {
  status: {
    available: true, version: 'fixture', binaryPath: null, state: 'ready', model: 'fixture/model', drafter: null,
    port: 8000, startedAt: 1, readyAt: 2, message: null, logTail: [], totalMemoryBytes: 16 * 1024 ** 3,
    memoryLimitBytes: 12 * 1024 ** 3, contextTokens: 32768,
  },
  installed: [{ repo: 'fixture/model', family: 'Fixture', modelType: 'fixture', format: '4-bit', contextTokens: 32768,
    weightsBytes: 1024 ** 3, memoryFloorBytes: 2 * 1024 ** 3, fitsMemory: true, drafter: null, drafterInstalled: null }],
  suggestions: [{ repo: 'fixture/model', title: 'Friendly fixture model', revision: null, note: '' }],
  performance: { model: 'fixture/model', measuredAt: 3, decodeTokensPerSecond: 55, prefillTokensPerSecond: 100,
    firstTokenMs: 10, completionTokens: 20 },
  pulls: [], hubCache: '/synthetic/cache',
}
let snapshot: EngineSnapshot | null = structuredClone(fixture)
let error: string | null = null
let learningLocked = false
const calls: string[] = []
const engine = { ...format, useEngine: () => ({ snapshot, error }) }
const shared = {
  './EngineUpdates': { EngineUpdates: 'engine-updates' },
  'react/jsx-runtime': jsx, clsx, '../../lib/engine': engine, '../../lib/engine-readiness': { engineReadiness },
  '../../lib/engine-workspace': { engineBaseUrl, formatTokenRate },
  '../../lib/turn-stats-line': { repoDisplayName }, '../ui': { Button: 'button' },
  '../BrandMark': { BrandMark: 'brand-mark' }, '../../lib/icons': { Icon: 'icon' },
  '../../lib/preferences/useFlowText': { useFlowText: () => flowTextFor('en') },
}
const { RuntimeSummary } = load(path.join(__dirname, 'RuntimeSummary.tsx'), shared)
const { EngineRuntimePanel } = load(path.join(__dirname, 'EngineRuntimePanel.tsx'), {
  ...shared, '../../lib/learning': { useLearning: () => ({ snapshot: {} }), learningBusy: () => learningLocked },
  './CortheonToggle': { CortheonToggle: 'cortheon' }, './RuntimeSummary': { RuntimeSummary },
})
const { EngineDot, EngineStatusRow, engineDotTone } = load(path.join(__dirname, 'EngineStatus.tsx'), shared)
const { EngineChoice } = load(path.join(__dirname, 'EngineChoice.tsx'), { ...shared, './EngineStatus': { engineDotTone } })
const props = { activeRuns: 0, compact: true, onApi: () => calls.push('api'), onWorkers: () => calls.push('workers'),
  onUnload: () => calls.push('cancel'), onRetry: () => calls.push('retry'), onChat: () => calls.push('chat'),
  onDetails: () => calls.push('details'), stopping: false }
const render = (overrides = {}) => {
  const node = EngineRuntimePanel({ ...props, ...overrides })
  return node.type === RuntimeSummary ? RuntimeSummary(node.props) : node
}
const button = (tree: any, label: string) => nodes(tree).find(node => node.type === 'button' && text(node) === label)

let tree = render()
assert.match(text(tree), /Ready locally/)
assert.match(text(tree), /Friendly fixture model/)
assert.match(text(tree), /Model memory floor/)
assert.match(text(tree), /32,768 tokens/)
assert.match(text(tree), /55 tok\/s/)
button(tree, 'Chat with this model').props.onClick()
assert.deepEqual(calls, ['chat'])
assert.equal(button(tree, 'Cancel loading'), undefined)
learningLocked = true
assert.equal(button(render(), 'Chat with this model').props.disabled, true)
learningLocked = false
button(render({ minimal: true }), 'Use local API').props.onClick()
assert.equal(calls.at(-1), 'api')

snapshot!.status.state = 'starting'
tree = render()
assert.match(text(tree), /Loading model/)
assert.match(text(tree), /Reported after loading/)
assert.match(text(tree), /Measured after a request/)
assert.doesNotMatch(text(tree), /55 tok\/s|32,768 tokens|Ready locally/)
button(tree, 'Cancel loading').props.onClick()
assert.equal(calls.at(-1), 'cancel')
assert.equal(button(render({ activeRuns: 1 }), 'Cancel loading').props.disabled, true)
assert.equal(button(render({ stopping: true }), 'Cancelling…').props.disabled, true)

snapshot!.status.state = 'failed'
snapshot!.status.message = 'Synthetic load failure'
tree = render()
assert.match(text(tree), /Needs attention.*Synthetic load failure/)
button(tree, 'Retry loading').props.onClick()
assert.equal(calls.at(-1), 'retry')
assert.equal(button(render({ activeRuns: 1 }), 'Retry loading').props.disabled, true)
assert.equal(button(tree, 'Chat with this model'), undefined)

snapshot!.status.available = false
snapshot!.status.state = 'ready'
snapshot!.status.message = 'Synthetic engine unavailable'
tree = render()
assert.match(text(tree), /Engine unavailable/)
assert.match(text(tree), /Synthetic engine unavailable/)
assert.doesNotMatch(text(tree), /55 tok\/s|32,768 tokens|Ready locally/)
assert.equal(button(tree, 'Chat with this model'), undefined)
assert.equal(button(tree, 'Retry loading'), undefined)
assert.ok(nodes(tree).some(node => node.props.className?.includes('bg-danger')))

snapshot = structuredClone(fixture)
error = 'Synthetic snapshot failure'
tree = render()
assert.match(text(tree), /Engine status unavailable/)
assert.doesNotMatch(text(tree), /55 tok\/s|32,768 tokens|Ready locally/)
assert.equal(button(tree, 'Chat with this model'), undefined)
error = null
snapshot!.performance!.model = 'fixture/old-model'
assert.doesNotMatch(text(render()), /55 tok\/s/)
snapshot = null
assert.doesNotThrow(() => render({ compact: false }))
assert.match(text(render()), /Checking engine/)

snapshot = structuredClone(fixture)
snapshot.status.state = 'stopped'
snapshot.status.model = null
const choiceProps = { onOpenModels: () => calls.push('models'), onUse: (repo: string) => calls.push(repo) }
tree = EngineChoice(choiceProps)
assert.match(text(tree), /Friendly fixture model/)
assert.match(text(tree), /Downloaded/)
assert.match(text(tree), /loads when you send/)
assert.doesNotMatch(text(tree), /Running on this Mac/)
button(tree, 'Choose this model').props.onClick()
assert.equal(calls.at(-1), 'fixture/model')
snapshot.status.available = false
assert.equal(button(EngineChoice(choiceProps), 'Choose this model').props.disabled, true)
for (const value of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
  snapshot.installed[0].memoryFloorBytes = value
  assert.doesNotMatch(text(EngineChoice(choiceProps)), /memory floor/)
}
snapshot.status.available = true
snapshot.status.state = 'starting'
assert.equal(button(EngineChoice(choiceProps), 'Choose this model').props.disabled, true)

const { DownloadActivity } = load(path.join(__dirname, 'DownloadActivity.tsx'), {
  ...shared, './ModelRows': { PullRow: 'pull-row' }, './download-display': downloads,
})
const pull: PullProgress = { repo: 'fixture/model', revision: 'fixed-revision', state: 'failed', phase: 'Failed',
  receivedBytes: 100, totalBytes: 200, lastLine: null, message: 'Synthetic download error', startedAt: 1, finishedAt: 2 }
const activityProps = { pulls: [pull], installed: fixture.installed, showCompleted: true, locked: false, minimal: false,
  onAll: () => calls.push('all'), onCancel: async (repo: string) => calls.push(`cancel:${repo}`),
  onRetry: async (repo: string, revision: string | null) => calls.push(`retry:${repo}@${revision}`),
  onLoad: (repo: string) => calls.push(`load:${repo}`) }
tree = DownloadActivity(activityProps)
assert.match(text(tree), /Download needs attention/)
await nodes(tree).find(node => node.type === 'pull-row')!.props.onRetry()
assert.equal(calls.at(-1), 'retry:fixture/model@fixed-revision')
pull.state = 'cancelled'
assert.match(text(DownloadActivity(activityProps)), /Cached files are kept/)
pull.state = 'done'
tree = DownloadActivity(activityProps)
assert.match(text(tree), /Download complete/)
assert.equal(button(tree, 'Use model').props.disabled, false)
button(tree, 'Use model').props.onClick()
assert.equal(calls.at(-1), 'load:fixture/model')
assert.equal(button(DownloadActivity({ ...activityProps, locked: true }), 'Use model').props.disabled, true)
assert.equal(button(DownloadActivity({ ...activityProps, installed: [] }), 'Use model').props.disabled, true)
assert.equal(DownloadActivity({ ...activityProps, showCompleted: false }), null)

const chipHooks = harness()
let ejections = 0
const { EngineChip } = load(path.join(__dirname, '../sidebar/EngineChip.tsx'), {
  ...shared, react: chipHooks.hooks, '../../lib/ipc': { api: { engineUnload: async () => { ejections++ } } },
})
const chip = () => chipHooks.render(() => EngineChip({ active: false, onOpen: () => calls.push('chip-models') }))
const pickerStatus = () => EngineStatusRow({ onOpenModels: () => calls.push('picker-models') })
const hasReadyTone = (node: any) => nodes(node).some(child => child.props.className?.includes('bg-positive'))
const assertUnavailable = () => {
  const compact = chip()
  assert.match(text(compact), /Engine status unavailable/)
  assert.doesNotMatch(text(compact), /Loaded|fixture\/model/)
  assert.equal(button(compact, 'Eject'), undefined)
  assert.equal(hasReadyTone(compact), false)
  assert.equal(hasReadyTone(EngineDot()), false)
  assert.equal(hasReadyTone(pickerStatus()), false)
}

snapshot = structuredClone(fixture)
error = 'Synthetic stale ready snapshot'
assertUnavailable()
assert.equal(EngineDot().props.title, 'Engine status unavailable')
assert.equal(EngineDot().props.role, 'img')
assert.equal(EngineDot().props['aria-label'], 'Engine status unavailable')
assert.match(text(pickerStatus()), /Engine status unavailable/)
error = ''
assertUnavailable()
error = null
snapshot.status.available = false
assertUnavailable()
assert.match(EngineDot().props.title, /Engine missing/)
assert.match(text(pickerStatus()), /Engine missing/)

snapshot = structuredClone(fixture)
for (const model of [null, '', '   ']) {
  snapshot.status.model = model
  assert.match(text(chip()), /No model loaded/)
  assert.equal(button(chip(), 'Eject'), undefined)
  assert.equal(hasReadyTone(chip()), false)
  assert.equal(EngineDot().props.title, 'No model loaded')
  assert.match(text(pickerStatus()), /No model loaded/)
  assert.equal(hasReadyTone(EngineDot()), false)
}

snapshot = null
error = 'Synthetic missing snapshot'
assertUnavailable()
error = null
assert.match(text(chip()), /Checking engine/)
assert.equal(button(chip(), 'Eject'), undefined)

snapshot = structuredClone(fixture)
tree = chip()
assert.match(text(tree), /Loaded/)
assert.equal(hasReadyTone(tree), true)
assert.equal(hasReadyTone(EngineDot()), true)
assert.equal(hasReadyTone(pickerStatus()), true)
assert.match(EngineDot().props.title, /Running fixture\/model/)
assert.match(EngineDot().props['aria-label'], /Running fixture\/model/)
const chipButtons = nodes(tree).filter(node => node.type === 'button')
assert.equal(chipButtons.length, 2, 'Navigation and Eject are native sibling buttons')
assert.ok(chipButtons.every(node => node.props.type === 'button' && nodes(node).filter(child => child.type === 'button').length === 1))
assert.ok(!nodes(tree).some(node => node.props.role === 'button'), 'No synthetic button contains another button')
assert.match(button(tree, 'Eject').props.className, /group-focus-within:opacity-100/)
assert.match(chipButtons[0].props.className, /focus-visible:ring/)
chipButtons[0].props.onClick()
assert.equal(calls.at(-1), 'chip-models')
button(tree, 'Eject').props.onClick()
await settle()
assert.equal(ejections, 1)
assert.equal(calls.at(-1), 'chip-models', 'Eject does not bubble into model navigation')
button(pickerStatus(), 'Models').props.onClick()
assert.equal(calls.at(-1), 'picker-models')
error = 'Synthetic second read failure'
assertUnavailable()
error = null
assert.match(text(chip()), /Loaded/, 'A fresh successful read restores readiness and Eject')
assert.ok(button(chip(), 'Eject'))
assert.equal(hasReadyTone(EngineDot()), true)
console.log('Actual model setup rendering: honest readiness/metrics, cancel/retry guards, explicit selection and download recovery.')
