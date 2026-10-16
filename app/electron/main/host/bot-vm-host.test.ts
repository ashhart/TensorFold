import assert from 'node:assert/strict'
import { createBotVmToolContext } from './bot-vm-host'
import type { BotVmHost } from './bot-vm-host'
import { executeTool } from '../../agent/tools'

const calls: string[] = []
const host = { executeTool: async (_id: string, name: string) => { calls.push(name); return { ok: true, output: 'isolated result' } } } as unknown as BotVmHost
const context = createBotVmToolContext(host, 'writer', { cwd: '/fictional-host', callMcpTool: async () => { throw new Error('host callback escaped') }, runComputerUse: async () => { throw new Error('host computer escaped') }, writeManagedOutput: async () => { throw new Error('host files escaped') } })
assert.equal(context.cwd, '/workspace')
assert.equal(context.callMcpTool, undefined)
assert.equal(context.runComputerUse, undefined)
assert.equal(context.writeManagedOutput, undefined)
assert.equal(context.backend, undefined)
assert.equal((await executeTool('read_file', { path: 'a' }, context)).output, 'isolated result')
for (const tool of ['mcp__foreign', 'computer_use', 'phone_observe', 'web_fetch', 'launch_sub_agent', 'save_artifact', 'create_project']) assert.equal((await executeTool(tool, {}, context)).ok, false)
assert.deepEqual(calls, ['read_file'])
console.log('bot-vm host target dispatch cannot fall back to host tools')
