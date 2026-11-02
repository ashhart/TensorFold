import assert from 'node:assert/strict'
import clsx from 'clsx'
import { jsx, load, nodes, text } from '../reading/reading-test-harness'

const { TaskContextBar } = load('src/components/thread-panel/TaskContextBar.tsx', {
  'react/jsx-runtime': jsx, clsx,
  '../../lib/icons': { Icon: 'icon' },
  '../../lib/format': { basename: (value: string) => value.split('/').pop() },
})
const actions: string[] = []
const props = {
  cwd: '/synthetic/worktree', projectName: 'Compiler', remote: false, branch: 'app-improvements', gitAvailable: true,
  changes: { files: 3, additions: 21, deletions: 4 },
  onWorkspace: () => actions.push('details'), onReview: () => actions.push('changes'), onTerminal: () => actions.push('terminal'),
}
let rendered = TaskContextBar(props)
assert.match(text(rendered), /Compiler Local workspace · worktree app-improvements/)
assert.match(text(rendered), /Review changes 3 \+\s*21 −\s*4/)
const buttons = nodes(rendered).filter(node => node.type === 'button')
buttons[0].props.onClick()
buttons[1].props.onClick()
buttons[2].props.onClick()
assert.deepEqual(actions, ['details', 'terminal', 'changes'], 'actions open existing panes and never mutate Git')
assert.equal(buttons[0].props.title, '/synthetic/worktree')
assert.equal(buttons[1].props['aria-label'], 'Open workspace terminal')
rendered = TaskContextBar({ ...props, remote: true, branch: null, gitAvailable: false, changes: { files: 0, additions: 0, deletions: 0 } })
assert.match(text(rendered), /Remote workspace/)
assert.match(text(rendered), /Git status unavailable/)
assert.doesNotMatch(text(rendered), /clean|verified|Detached HEAD/i, 'unknown Git state never looks clean or verified')
assert.equal(nodes(rendered).filter(node => node.type === 'button').at(-1)!.props.disabled, undefined, 'review remains available for an empty or unavailable snapshot')
rendered = TaskContextBar({ ...props, branch: null, gitAvailable: true })
assert.match(text(rendered), /No branch reported/)
assert.doesNotMatch(text(rendered), /Detached HEAD/, 'a missing branch is also possible outside a Git repository')
console.log('Task context, change counts and pane navigation preserve unknown and remote workspace states')
