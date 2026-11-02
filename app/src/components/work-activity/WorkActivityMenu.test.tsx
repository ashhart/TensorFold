import assert from 'node:assert/strict'
import clsx from 'clsx'
import * as activityModule from '../../lib/full-workspace-activity'
import { harness, jsx, load, nodes, text } from '../reading/reading-test-harness'

const hooks = harness()
const state: any = { threads: [], status: {}, approvals: {}, jobs: [], rooms: [], roomStatus: {}, automations: [], projects: [], setView: (view: unknown) => opened.push(view) }
const opened: unknown[] = []
const Menu = () => null
const { WorkActivityMenu, WorkActivityContent } = load('src/components/work-activity/WorkActivityMenu.tsx', {
  react: hooks.hooks, 'react/jsx-runtime': jsx, clsx,
  '../../lib/full-workspace-activity': activityModule, '../../lib/icons': { Icon: 'icon' },
  '../../lib/store': { useStore: (select: any) => select(state) }, '../ui': { Menu },
})
const menu = (variant?: string) => nodes(hooks.render(() => WorkActivityMenu({ variant }))).find((node) => node.type === Menu)!
const renderContent = (activity: unknown[]) => hooks.render(() => WorkActivityContent({ id: 'work', activity, navigate: (view: unknown) => opened.push(view) }))

let panel = renderContent([])
assert.match(text(panel), /No active work recorded/)
assert.match(text(panel), /Models and API servers can still be loaded/, 'empty work does not imply an idle model runtime')
const openFull = nodes(panel).find((node) => node.type === 'button')!
openFull.props.onClick()
assert.deepEqual(opened.pop(), { name: 'commandCentre' })

state.threads = [
  { id: 'ask', title: 'Question task', kind: 'task', projectId: 'compiler', status: 'idle', updatedAt: 5, timeline: [{ question: { state: 'pending', resolvedAt: null } }] },
  { id: 'work', title: 'Running task', kind: 'task', status: 'running', updatedAt: 6, timeline: [] },
]
state.projects = [{ id: 'compiler', name: 'Compiler' }]
state.jobs = [{ id: 'job', kind: 'thread', currentThreadId: 'work', status: 'running', updatedAt: 7, pendingApprovals: [] }]
const activity = activityModule.fullWorkspaceActivity(state)
panel = renderContent(activity)
const sections = nodes(panel).filter((node) => node.type === 'section')
assert.equal(sections.length, 2)
assert.match(text(sections[0]), /Needs you 1/)
assert.match(text(sections[0]), /Question task Task · Compiler Question to answer/)
assert.doesNotMatch(text(sections[0]), /Running task/)
assert.match(text(sections[1]), /In progress 1/)
assert.match(text(sections[1]), /Running task Task Running/)
nodes(panel).find((node) => node.props['data-activity-key'] === 'thread:ask')!.props.onClick()
assert.deepEqual(opened.pop(), { name: 'thread', id: 'ask' })

for (const variant of ['toolbar', 'sidebar']) {
  const current = menu(variant)
  const trigger = current.props.trigger(false)
  assert.equal(trigger.props['aria-label'], 'Work activity: 2 items, 1 need attention')
  assert.equal(trigger.props['aria-expanded'], false)
  assert.equal(current.props.trigger(true).props['aria-controls'].length > 0, true)
  if (variant === 'sidebar') assert.match(trigger.props.className, /w-full/)
  let closed = false
  const content = current.props.children(() => { closed = true })
  content.props.navigate({ name: 'thread', id: 'ask' })
  assert.deepEqual(opened.pop(), { name: 'thread', id: 'ask' })
  assert.equal(closed, true, 'opening work closes the popover in either workspace layout')
}

const bounded = Array.from({ length: 17 }, (_, i) => ({ ...activity[0], key: `thread:${i}`, title: `Pending ${i}` }))
panel = renderContent(bounded)
assert.equal(nodes(panel).filter((node) => node.props['data-activity-key']).length, 12)
assert.match(text(panel), /Show all 17 items/)
nodes(panel).find((node) => node.type === 'button' && text(node) === 'Show all 17 items')!.props.onClick()
panel = renderContent(bounded)
assert.equal(nodes(panel).filter((node) => node.props['data-activity-key']).length, 17, 'overflow questions and outcome reviews remain accessible without relying on a different dashboard')
assert.match(text(panel), /Show fewer/)
assert.match(text(panel), /Open full activity/)
console.log('Shared work activity accessible counts, attention/progress grouping, provenance and navigation passed')
