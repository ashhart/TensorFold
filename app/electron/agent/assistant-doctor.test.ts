import assert from 'node:assert/strict'
import { buildToolSchemas, executeTool, READ_ONLY_TOOLS, TOOL_SCHEMAS } from './tools.ts'
import { buildAssistantDoctorReport, formatAssistantDoctorReport } from './assistant-doctor'
import type { McpToolDefinition, Thread } from './types'

const currentThread: Thread = {
  id: 'thread-current',
  title: 'Hermes parity',
  kind: 'task',
  projectId: 'p1',
  cwd: '/repo',
  providerId: 'provider',
  model: 'model',
  createdAt: 1,
  updatedAt: 2,
  status: 'idle',
  apiMessages: [{ role: 'system', content: 'base' }],
  systemPromptSnapshot: {
    sections: { base: 'base', persona: 'persona', goal: 'goal', capabilities: 'capability registry' },
    fingerprint: 'abc123',
    createdAt: 10,
    updatedAt: 11,
  },
  timeline: [],
}

const mcpTools: McpToolDefinition[] = Array.from({ length: 26 }, (_, index) => ({
  serverId: 'srv',
  serverName: 'demo',
  approvalMode: 'ask',
  name: `tool_${index}`,
  callName: `mcp__demo__tool_${index}`,
  inputSchema: { type: 'object', properties: {} },
}))

const report = buildAssistantDoctorReport({
  currentThread,
  threads: [currentThread, { ...currentThread, id: 'thread-prior', title: 'Prior local work' }],
  mcpTools,
  skills: [{ id: 's1', name: 'Skill', description: 'Reusable flow', body: 'body', enabled: true, createdAt: 1, updatedAt: 1 } as any],
  commands: [{ id: 'c1', name: 'triage', prompt: 'prompt', enabled: true, createdAt: 1, updatedAt: 1 } as any],
  disabledCapabilityIds: ['browser'],
})

assert.equal(report.promptSnapshot.present, true)
assert.equal(report.promptSnapshot.fingerprint, 'abc123')
assert.equal(report.toolSurface.mcpTools, 26)
assert.equal(report.toolSurface.deferredMcpTools, true)
assert.equal(report.memoryAndRecall.sessionThreads, 2)
assert.equal(report.memoryAndRecall.sessionRecallReady, true)
assert.match(formatAssistantDoctorReport(report), /Prompt snapshot: present \(abc123\)/)
assert.match(formatAssistantDoctorReport(report), /MCP tools: 26 \(deferred behind bridge\)/)
console.log('ok assistant doctor reports prompt, tool disclosure, and session recall readiness')

assert.ok(READ_ONLY_TOOLS.has('assistant_doctor'))
assert.ok(TOOL_SCHEMAS.some((tool: any) => tool.function?.name === 'assistant_doctor'))
assert.ok(buildToolSchemas().some((tool: any) => tool.function?.name === 'assistant_doctor'))

const toolResult = await executeTool('assistant_doctor', {}, {
  cwd: '/repo',
  currentThread,
  threads: [currentThread],
  mcpTools,
  disabledCapabilityIds: ['browser'],
})
assert.equal(toolResult.ok, true)
assert.equal(toolResult.title, 'Assistant doctor')
assert.match(toolResult.output, /Prompt snapshot: present/)
assert.match(toolResult.output, /Disabled capabilities: browser/)
console.log('ok assistant_doctor is exposed as a read-only tool')
