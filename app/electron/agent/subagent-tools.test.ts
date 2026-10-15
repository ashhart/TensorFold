import assert from 'node:assert/strict'
import { executeTool, READ_ONLY_TOOLS } from './tools'
import type { SubAgent, Thread } from './types'

const thread: Thread = {
  id: 'parent', title: 'Parent', kind: 'task', cwd: '/repo', providerId: 'provider', model: 'model',
  createdAt: 1, updatedAt: 1, status: 'running', apiMessages: [], timeline: [],
}

let active = 0
let maxActive = 0
const finished: string[] = []
const batch = await executeTool('launch_subagents', {
  tasks: [{ task: 'slow' }, { task: 'fast' }, { task: 'medium' }],
}, {
  cwd: '/repo',
  currentThread: thread,
  launchSubAgent: async (input) => {
    active++
    maxActive = Math.max(maxActive, active)
    const delay = input.task === 'slow' ? 20 : input.task === 'fast' ? 2 : 10
    await new Promise((resolve) => setTimeout(resolve, delay))
    active--
    finished.push(input.task)
    return { agent: agent(input.task, 'succeeded'), output: `${input.task} done` }
  },
})

assert.equal(batch.ok, true)
assert.equal(maxActive, 3, 'all three child launches overlap')
assert.deepEqual(finished, ['fast', 'medium', 'slow'], 'fixture proves completion order differs')
assert.match(batch.output, /^1\. slow: slow done\n2\. fast: fast done\n3\. medium: medium done$/, 'reported results stay in request order')

const agents = [agent('background one', 'running'), agent('background two', 'succeeded')]
const status = await executeTool('subagent_status', {}, { cwd: '/repo', currentThread: thread, listSubAgents: () => agents })
assert.equal(status.ok, true)
assert.match(status.output, /background one \[running\]/)
assert.match(status.output, /background two \[succeeded\]/)
assert.ok(READ_ONLY_TOOLS.has('subagent_status'))

let sentText = ''
const sent = await executeTool('subagent_send', { agent_id: agents[0].id, kind: 'question', text: 'Need a decision.' }, {
  cwd: '/repo',
  currentThread: thread,
  sendSubAgentMessage: (input) => {
    sentText = input.text
    return {
      id: 'message-1', sender: 'parent', audience: 'child', kind: input.kind ?? 'message', text: input.text,
      replyToId: null, createdAt: 4, readByParentAt: 4, readByChildAt: null,
    }
  },
})
assert.equal(sent.ok, true)
assert.equal(sentText, 'Need a decision.')

let markedRead = false
const inbox = await executeTool('subagent_inbox', { agent_id: agents[0].id }, {
  cwd: '/repo',
  currentThread: thread,
  readSubAgentMessages: (input) => {
    markedRead = input.markRead !== false
    return [{
      id: 'message-2', sender: 'child', audience: 'parent', kind: 'question', text: 'Which branch?',
      replyToId: null, createdAt: 5, readByParentAt: null, readByChildAt: 5,
    }]
  },
})
assert.equal(inbox.ok, true)
assert.equal(markedRead, true)
assert.match(inbox.output, /Which branch\?/)
assert.ok(READ_ONLY_TOOLS.has('subagent_send'))
assert.ok(READ_ONLY_TOOLS.has('subagent_inbox'))

// Agent-typed presets (specialist delegation): the batch-level agent default
// applies to every item, an item-level agent overrides it, and unknown values pass as
// null rather than erroring mid-flight.
{
  const received: Array<string | null> = []
  const presetBatch = await executeTool('launch_subagents', {
    agent: 'scout',
    tasks: [{ task: 'research deps' }, { task: 'audit call sites', agent: 'reviewer' }, { task: 'plain worker' }],
  }, {
    cwd: '/repo',
    currentThread: thread,
    launchSubAgent: async (input) => {
      received.push(input.agent ?? null)
      return { agent: agent(input.task, 'succeeded'), output: `${input.task} done` }
    },
  })
  assert.equal(presetBatch.ok, true)
  assert.deepEqual(received, ['scout', 'reviewer', 'scout'], 'batch default applies unless an item overrides it')

  const single = await executeTool('launch_subagent', { task: 'check auth', agent: 'security' }, {
    cwd: '/repo',
    currentThread: thread,
    launchSubAgent: async (input) => {
      assert.equal(input.agent, 'security')
      return { agent: agent(input.task, 'succeeded'), output: 'no findings' }
    },
  })
  assert.equal(single.ok, true)
  console.log('sub-agent tool concurrency, status, and preset tests passed')
}

function agent(task: string, status: SubAgent['status']): SubAgent {
  return {
    id: `agent-${task.replace(/\s+/g, '-')}`,
    parent: { kind: 'thread', id: thread.id, label: thread.title, threadId: thread.id },
    name: task,
    task,
    status,
    projectId: null,
    roomId: null,
    threadId: thread.id,
    seatId: null,
    providerId: 'provider',
    model: 'model',
    runId: null,
    result: status === 'succeeded' ? `${task} result` : null,
    error: null,
    events: [],
    messages: [],
    createdAt: 1,
    updatedAt: status === 'succeeded' ? 3 : 2,
    startedAt: 1,
    finishedAt: status === 'succeeded' ? 3 : null,
  }
}
