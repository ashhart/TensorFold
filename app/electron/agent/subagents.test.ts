// Run: npx tsx electron/agent/subagents.test.ts
import assert from 'node:assert/strict'
import {
  appendSubAgentMessage,
  appendSubAgentEvent,
  cancelSubAgent,
  completeSubAgent,
  createSubAgent,
  createSubAgentFromLaunch,
  failSubAgent,
  groupSubAgentsByParent,
  normalizeSubAgents,
  parentForRoomSeat,
  parentForSubAgent,
  parentForThread,
  parentKey,
  prepareSubAgentLaunch,
  readSubAgentMessages,
  startSubAgent,
  subAgentCategoryLabel,
  subAgentUnreadCount,
  subAgentsForParent,
  finalSubAgentText,
  summarizeSubAgentTimeline,
} from './subagents'
import type { Room, Seat, Thread } from './types'

let seq = 0
const id = () => `id-${++seq}`

function seat(name: string): Seat {
  return { id: `seat-${name.toLowerCase()}`, name, role: `${name} role`, providerId: 'openai', model: 'gpt-4o' }
}

function room(): Room {
  return {
    id: 'room-1',
    projectId: 'project-1',
    name: 'Backend lane',
    seats: [seat('Bob'), seat('Vera')],
    problem: 'Build sub-agent hierarchy',
    state: 'running',
    events: [],
    finalText: '',
    createdAt: 100,
    updatedAt: 100,
  }
}

function thread(): Thread {
  return {
    id: 'thread-1',
    title: 'Solo implementation',
    kind: 'task',
    projectId: 'project-1',
    cwd: '/tmp/project',
    providerId: 'openai',
    model: 'gpt-4o',
    createdAt: 100,
    updatedAt: 100,
    status: 'idle',
    apiMessages: [],
    timeline: [],
  }
}

{
  const r = room()
  const bob = parentForRoomSeat(r, r.seats[0])
  const vera = parentForRoomSeat(r, r.seats[1])

  assert.equal(parentKey(bob), 'seat:room:room-1:seat-bob')
  assert.equal(subAgentCategoryLabel(bob), "Bob's spawned agents")
  assert.equal(subAgentCategoryLabel({ label: 'Jess' }), "Jess' spawned agents")

  let agents = createSubAgent([], { parent: bob, task: 'Map the persistence path' }, 1_000, id).agents
  agents = createSubAgent(agents, { parent: vera, task: 'Check renderer affordances' }, 1_010, id).agents
  const created = createSubAgent(agents, { parent: bob, task: 'Find IPC integration point' }, 1_020, id)
  agents = created.agents

  assert.equal(created.agent.name, 'Bob sub-agent 2')
  assert.deepEqual(subAgentsForParent(agents, bob).map((agent) => agent.task), ['Find IPC integration point', 'Map the persistence path'])

  const groups = groupSubAgentsByParent(agents)
  assert.equal(groups.length, 2)
  assert.equal(groups[0].label, "Bob's spawned agents")
  assert.equal(groups[0].agents.length, 2)
  assert.equal(groups[1].label, "Vera's spawned agents")
  assert.equal(groups[1].agents.length, 1)
}

{
  const r = room()
  const bob = parentForRoomSeat(r, r.seats[0])
  let { agents, agent } = createSubAgent([], { parent: bob, task: 'Run focused research', providerId: 'openai', model: 'gpt-4o' }, 2_000, id)

  let started = startSubAgent(agents, agent.id, 'run-1', 2_100)
  agents = started.agents
  agent = started.agent!
  assert.equal(agent.status, 'running')
  assert.equal(agent.startedAt, 2_100)
  assert.equal(agent.runId, 'run-1')
  assert.equal(agent.events.at(-1)?.kind, 'started')

  const progressed = appendSubAgentEvent(agents, agent.id, 'progress', 'Read room orchestration.', 2_150, id)
  agents = progressed.agents
  assert.equal(progressed.agent?.events.at(-1)?.text, 'Read room orchestration.')

  const completed = completeSubAgent(agents, agent.id, 'Use subagents.json plus room events.', 2_200)
  agent = completed.agent!
  assert.equal(agent.status, 'succeeded')
  assert.equal(agent.finishedAt, 2_200)
  assert.equal(agent.result, 'Use subagents.json plus room events.')
  assert.equal(agent.error, null)
  assert.equal(agent.events.at(-1)?.kind, 'result')
  assert.equal(subAgentUnreadCount(agent, 'parent'), 1)
  assert.equal(agent.messages.at(-1)?.kind, 'result')
}

{
  const parent = parentForThread(thread())
  let { agents, agent } = createSubAgent([], { parent, task: 'Mailbox handshake' }, 2_300, id)
  const question = appendSubAgentMessage(agents, agent.id, {
    sender: 'child',
    kind: 'question',
    text: 'Should I inspect the encrypted store?',
  }, 2_310, id)
  agents = question.agents
  agent = question.agent!
  assert.equal(subAgentUnreadCount(agent, 'parent'), 1)
  assert.equal(subAgentUnreadCount(agent, 'child'), 0)

  const parentRead = readSubAgentMessages(agents, agent.id, { audience: 'parent', unreadOnly: true }, 2_320)
  agents = parentRead.agents
  assert.equal(parentRead.messages[0].id, question.message?.id)
  assert.equal(subAgentUnreadCount(parentRead.agent!, 'parent'), 0)

  const reply = appendSubAgentMessage(agents, agent.id, {
    sender: 'parent',
    kind: 'reply',
    text: 'Yes. Treat it as canonical and do not modify it.',
    replyToId: question.message?.id,
  }, 2_330, id)
  assert.equal(subAgentUnreadCount(reply.agent!, 'child'), 1)
  const childRead = readSubAgentMessages(reply.agents, agent.id, { audience: 'child', unreadOnly: true }, 2_340)
  assert.equal(childRead.messages[0].replyToId, question.message?.id)
  assert.equal(subAgentUnreadCount(childRead.agent!, 'child'), 0)
}

{
  const r = room()
  const bob = parentForRoomSeat(r, r.seats[0])
  let { agents, agent } = createSubAgent([], { parent: bob, task: 'Nested verification' }, 3_000, id)
  const childParent = parentForSubAgent(agent)
  assert.equal(childParent.kind, 'agent')
  assert.equal(childParent.roomId, 'room-1')

  const nested = createSubAgent(agents, { parent: childParent, task: 'Verify parent context' }, 3_010, id)
  agents = nested.agents
  assert.equal(groupSubAgentsByParent(agents).find((group) => group.parent.kind === 'agent')?.label, `${agent.name}'s spawned agents`)

  const failed = failSubAgent(agents, nested.agent.id, 'Provider limit reached.', 3_100)
  assert.equal(failed.agent?.status, 'failed')
  assert.equal(failed.agent?.finishedAt, 3_100)
  assert.equal(failed.agent?.error, 'Provider limit reached.')

  const cancelled = cancelSubAgent(agents, nested.agent.id, 'Parent run cancelled.', 3_200)
  assert.equal(cancelled.agent?.status, 'cancelled')
  assert.equal(cancelled.agent?.finishedAt, 3_200)
  assert.equal(cancelled.agent?.events.at(-1)?.kind, 'cancelled')
}

{
  const parent = parentForThread(thread())
  assert.equal(parentKey(parent), 'thread:thread:thread-1:thread-1')
  const { agent } = createSubAgent([], { parent, task: 'Summarize thread blockers', name: 'Thread scout' }, 4_000, id)
  assert.equal(agent.name, 'Thread scout')
  assert.equal(agent.threadId, 'thread-1')
  assert.equal(agent.projectId, 'project-1')
}

{
  const r = room()
  const bob = parentForRoomSeat(r, r.seats[0])
  const launch = prepareSubAgentLaunch({
    source: 'room_delegate',
    parent: bob,
    task: '  Inspect launch semantics  ',
    name: '  Launch scout  ',
    providerId: 'openai',
    providerAccountId: 'acct-1',
    model: 'gpt-4o',
    runId: 'run-sub-1',
    cwd: '/tmp/project',
  })
  assert.equal(launch.task, 'Inspect launch semantics')
  assert.equal(launch.name, 'Launch scout')
  assert.equal(launch.parent.roomId, 'room-1')
  assert.equal(launch.providerAccountId, 'acct-1')

  const created = createSubAgentFromLaunch([], { ...launch, source: 'thread_tool' }, 4_500, id)
  assert.equal(created.agent.name, 'Launch scout')
  assert.equal(created.agent.task, 'Inspect launch semantics')
  assert.equal(created.agent.runId, 'run-sub-1')
  assert.equal(created.agent.parent.kind, 'seat')
}

{
  const summaries = summarizeSubAgentTimeline([
    { id: 'a', kind: 'assistant', text: 'Thinking through it.', createdAt: 1 },
    { id: 't1', kind: 'tool', tool: 'read_file', path: 'src/App.tsx', state: 'ok', output: 'read contents', createdAt: 2 },
    { id: 't2', kind: 'tool', tool: 'run_command', state: 'denied', output: 'Denied by policy.', createdAt: 3 },
    { id: 'e1', kind: 'error', text: 'Provider stopped.', createdAt: 4 },
  ])
  assert.equal(summaries.length, 3)
  assert.equal(summaries[0].kind, 'progress')
  assert.match(summaries[0].text, /read_file src\/App.tsx/)
  assert.equal(summaries[1].kind, 'error')
  assert.equal(summaries[2].kind, 'error')
  assert.equal(finalSubAgentText([
    { id: 't1', kind: 'tool', tool: 'search', state: 'ok', output: 'search evidence', createdAt: 1 },
    { id: 'a1', kind: 'assistant', text: 'Final child answer.', createdAt: 2 },
  ]), 'Final child answer.')
}

{
  const normalized = normalizeSubAgents([
    null,
    { id: 'bad' },
    {
      id: 'sub-1',
      parent: { kind: 'seat', id: 'seat-bob', label: ' Bob ', projectId: 'project-1', roomId: 'room-1' },
      name: '',
      task: '  Inspect persisted data  ',
      status: 'nonsense',
      events: [{ id: 'event-1', kind: 'weird', text: '  restored  ', createdAt: Number.NaN }],
      messages: [{ id: 'message-1', sender: 'child', audience: 'parent', kind: 'question', text: '  restored question  ', createdAt: 40 }],
      updatedAt: 50,
    },
  ], 5_000)

  assert.equal(normalized.length, 1)
  assert.equal(normalized[0].name, 'Bob sub-agent')
  assert.equal(normalized[0].task, 'Inspect persisted data')
  assert.equal(normalized[0].status, 'queued')
  assert.equal(normalized[0].events[0].kind, 'progress')
  assert.equal(normalized[0].events[0].createdAt, 5_000)
  assert.equal(normalized[0].messages[0].text, 'restored question')
  assert.equal(subAgentUnreadCount(normalized[0], 'parent'), 1)
  assert.equal(normalized[0].createdAt, 5_000)
}

console.log('subagents tests passed')
