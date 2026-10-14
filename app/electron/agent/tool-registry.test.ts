import assert from 'node:assert/strict'
import {
  buildStaticToolRegistry,
  isRegistryReadOnlyTool,
  toolRegistrySchemas,
} from './tool-registry'

function names(entries: ReturnType<typeof buildStaticToolRegistry>): string[] {
  return entries.map((entry) => entry.name)
}

const agentRegistry = buildStaticToolRegistry()
const agentNames = names(agentRegistry)

assert.ok(agentNames.includes('read_file'))
assert.ok(agentNames.includes('apply_patch'))
assert.ok(agentNames.includes('web_search'))
assert.equal(isRegistryReadOnlyTool('read_file', agentRegistry), true)
assert.equal(isRegistryReadOnlyTool('web_search', agentRegistry), true)
assert.equal(isRegistryReadOnlyTool('apply_patch', agentRegistry), false)
assert.equal(isRegistryReadOnlyTool('calendar_create', agentRegistry), false)
assert.equal(isRegistryReadOnlyTool('mcp__demo__read_only_claim', agentRegistry), false)
assert.ok(toolRegistrySchemas(agentRegistry).some((schema: any) => schema.function?.name === 'read_file'))
console.log('ok static tool registry preserves agent schema and read-only metadata')

const chatRegistry = buildStaticToolRegistry({ scope: 'chat' })
const chatNames = names(chatRegistry)

assert.ok(chatNames.includes('web_search'))
assert.ok(chatNames.includes('visualize'))
assert.equal(chatNames.includes('read_file'), false)
assert.equal(chatNames.includes('run_command'), false)
assert.equal(isRegistryReadOnlyTool('web_fetch', chatRegistry), true)
console.log('ok static tool registry keeps chat tools separate from agent tools')
