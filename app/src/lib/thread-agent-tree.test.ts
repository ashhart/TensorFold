// Run: npx tsx src/lib/thread-agent-tree.test.ts
import assert from 'node:assert/strict'
import { buildThreadAgentTree, countThreadAgentNodes } from './thread-agent-tree'
import type { Thread } from './types'

function thread(events: Thread['timeline']): Thread {
  return {
    id: 'thread-1',
    title: 'Solo implementation',
    kind: 'task',
    projectId: 'project-1',
    cwd: '/tmp/project',
    providerId: 'openai',
    model: 'gpt-4o',
    createdAt: 1,
    updatedAt: 5,
    status: 'idle',
    apiMessages: [],
    timeline: events,
  }
}

{
  const tree = buildThreadAgentTree(thread([]))
  assert.equal(tree.status, 'idle')
  assert.equal(countThreadAgentNodes(tree), 0)
}

{
  const tree = buildThreadAgentTree(thread([
    { id: 'user-1', kind: 'user', text: 'Ship the panel.', createdAt: 1 },
    {
      id: 'launch-1',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Audit scout',
      args: { task: 'Inspect the timeline event shape', name: 'Audit scout' },
      state: 'ok',
      output: 'Sub-agent result: Timeline stores title, args, output, and meta.',
      meta: 'sub-agent succeeded',
      createdAt: 2,
    },
  ]))

  assert.equal(tree.name, 'Main agent')
  assert.equal(tree.detail, 'Solo implementation')
  assert.equal(countThreadAgentNodes(tree), 1)
  assert.equal(tree.children[0].name, 'Audit scout')
  assert.equal(tree.children[0].detail, 'Inspect the timeline event shape')
  assert.equal(tree.children[0].status, 'succeeded')
  assert.match(tree.children[0].result, /Timeline stores/)
}

{
  const tree = buildThreadAgentTree(thread([
    {
      id: 'launch-missing-metadata',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Launch sub-agent',
      args: {},
      state: 'running',
      createdAt: 10,
    },
  ]))

  assert.equal(tree.status, 'running')
  assert.equal(tree.children[0].id, 'subagent:launch-missing-metadata')
  assert.equal(tree.children[0].name, 'Main agent sub-agent 1')
  assert.equal(tree.children[0].detail, 'Sub-agent launch')
  assert.equal(tree.children[0].status, 'running')
}

{
  const rawId = '8f14e45f-ea4c-4a2a-8f05-33f4fd2b7a11'
  const tree = buildThreadAgentTree({
    ...thread([
      {
        id: 'launch-raw-name',
        kind: 'tool',
        tool: 'launch_subagent',
        title: 'Launch sub-agent',
        subAgent: { id: rawId, name: `${rawId} sub-agent 1`, task: 'Use the parent owner label', status: 'running' },
        state: 'running',
        createdAt: 11,
      },
    ] as any),
    title: rawId,
  })

  assert.equal(tree.children[0].name, 'Main agent sub-agent 1')
  assert.equal(tree.children[0].detail, 'Use the parent owner label')
}

{
  const tree = buildThreadAgentTree(thread([
    {
      id: 'launch-parent',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Parent mapper',
      args: { agentId: 'agent-parent', task: 'Map the work', status: 'succeeded' },
      state: 'ok',
      createdAt: 20,
    },
    {
      id: 'launch-child',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Child verifier',
      args: { agentId: 'agent-child', parentAgentId: 'agent-parent', task: 'Check the map', status: 'failed' },
      output: 'Sub-agent failed: Provider limit reached.',
      state: 'error',
      createdAt: 21,
    },
  ]))

  assert.equal(tree.children.length, 1)
  assert.equal(tree.children[0].name, 'Parent mapper')
  assert.equal(tree.children[0].children.length, 1)
  assert.equal(tree.children[0].children[0].name, 'Child verifier')
  assert.equal(tree.children[0].children[0].status, 'failed')
  assert.equal(countThreadAgentNodes(tree), 2)
}

{
  const longName = '  Very long research specialist name that should remain available for the UI to truncate gracefully instead of crashing layout  '
  const tree = buildThreadAgentTree(thread([
    {
      id: 'launch-long',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Launch sub-agent',
      args: { name: longName, task: '  Gather    context  ' },
      state: 'pending',
      createdAt: Number.NaN,
    },
  ]))

  assert.equal(tree.children[0].name.startsWith('Very long research specialist name'), true)
  assert.equal(tree.children[0].detail, 'Gather context')
  assert.equal(tree.children[0].lastEventAt, null)
}

{
  const tree = buildThreadAgentTree(thread([
    {
      id: 'launch-child-first',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Child implementer',
      args: {
        agentId: 'agent-child-first',
        parentAgentId: 'agent-parent-later',
        task: 'Implement after the parent plan exists',
        status: 'running',
      },
      state: 'running',
      createdAt: 31,
    },
    {
      id: 'launch-parent-later',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Parent planner',
      args: { agentId: 'agent-parent-later', task: 'Plan the sub-agent wave', status: 'succeeded' },
      state: 'ok',
      createdAt: 30,
    },
    {
      id: 'launch-sibling',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Sibling reviewer',
      args: { agentId: 'agent-sibling', task: 'Review the wave', status: 'queued' },
      state: 'pending',
      createdAt: 32,
    },
  ]))

  assert.equal(tree.children.length, 2)
  assert.equal(tree.children[0].name, 'Parent planner')
  assert.equal(tree.children[0].children.length, 1)
  assert.equal(tree.children[0].children[0].name, 'Child implementer')
  assert.equal(tree.children[0].children[0].status, 'running')
  assert.equal(tree.children[1].name, 'Sibling reviewer')
  assert.equal(countThreadAgentNodes(tree), 3)
}

{
  const tree = buildThreadAgentTree(thread([
    {
      id: 'launch-object-parent',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Launch sub-agent',
      subAgent: { id: 'agent-object-parent', name: 'Object parent', task: 'Map object lineage', status: 'succeeded' },
      state: 'ok',
      createdAt: 40,
    },
    {
      id: 'launch-object-child',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Launch sub-agent',
      subAgent: {
        id: 'agent-object-child',
        name: 'Object child',
        task: 'Use parent ref',
        status: 'succeeded',
        parent: { kind: 'agent', id: 'agent-object-parent', label: 'Object parent' },
      },
      state: 'ok',
      createdAt: 41,
    },
  ] as any))

  assert.equal(tree.children.length, 1)
  assert.equal(tree.children[0].name, 'Object parent')
  assert.equal(tree.children[0].children.length, 1)
  assert.equal(tree.children[0].children[0].name, 'Object child')
  assert.equal(tree.children[0].children[0].detail, 'Use parent ref')
}

{
  const tree = buildThreadAgentTree(thread([
    {
      id: 'launch-unnamed-child-first',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Launch sub-agent',
      args: { agentId: 'unnamed-child', parentAgentId: 'unnamed-parent', task: 'Child should wait for parent naming', status: 'running' },
      state: 'running',
      createdAt: 51,
    },
    {
      id: 'launch-unnamed-parent-later',
      kind: 'tool',
      tool: 'launch_subagent',
      title: 'Launch sub-agent',
      args: { agentId: 'unnamed-parent', task: 'Parent should get a main-agent label', status: 'succeeded' },
      state: 'ok',
      createdAt: 50,
    },
  ]))

  assert.equal(tree.children.length, 1)
  assert.equal(tree.children[0].name, 'Main agent sub-agent 1')
  assert.equal(tree.children[0].children[0].name, 'Main agent sub-agent 1 sub-agent 1')
  assert.equal(tree.children[0].children[0].detail, 'Child should wait for parent naming')
}

console.log('thread-agent-tree tests passed')
