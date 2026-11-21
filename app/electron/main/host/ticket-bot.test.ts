import assert from 'node:assert/strict'
import { captureTicketBot, sameTicketBot } from './ticket-bot'
import { pickTicketUserPatch } from './room-ticket-patch-policy'
import { mkTicket } from './ticket-helpers'
import { defaultSettings } from '../../store/defaults'
import type { Settings, Ticket } from '../../agent/types'

const settings = {
  ...defaultSettings(),
  providers: [{ id: 'local', kind: 'local', name: 'Local', baseUrl: 'http://127.0.0.1:8080/v1', models: ['small'] }],
  agentPresets: [{ id: 'builder', lane: 'development', createdAt: 1, updatedAt: 1, name: 'Builder', role: 'Read documentation', instructions: 'Summarize concisely', providerId: 'local', model: 'small', approvalScope: 'readonly', executionTarget: 'workspace' }],
} as Settings
const ticket = { botId: 'builder', executionMode: 'solo' } as Ticket
const bot = captureTicketBot(ticket, 'build', settings)!
assert.equal(bot.model, 'small')
assert.equal(bot.instructions, 'Summarize concisely')
assert.equal(bot.approvalScope, 'readonly')
assert.equal(captureTicketBot(ticket, 'review', settings), undefined)
assert.equal(captureTicketBot({ ...ticket, executionMode: 'room' }, 'build', settings), undefined)
assert.equal(captureTicketBot({ ...ticket, botId: null }, 'build', settings), undefined)
assert.throws(() => captureTicketBot({ ...ticket, botId: 'missing' }, 'build', settings), /unavailable/)
settings.agentPresets[0].paused = true
assert.throws(() => captureTicketBot(ticket, 'build', settings), /Resume/)
settings.agentPresets[0].paused = false
settings.agentPresets[0].executionTarget = 'vm'
assert.throws(() => captureTicketBot(ticket, 'build', settings), /VM ticket execution/)

assert.equal(sameTicketBot(undefined, undefined), true)
assert.equal(sameTicketBot(undefined, bot), false)
assert.equal(sameTicketBot(bot, undefined), false)
assert.equal(sameTicketBot(bot, { ...bot, capturedAt: bot.capturedAt + 1 }), true)
assert.equal(sameTicketBot(bot, { ...bot, instructions: 'Changed instructions' }), false)
assert.equal(sameTicketBot(bot, { ...bot, presetId: 'another' }), false)

assert.equal(pickTicketUserPatch({ botId: 'builder' }).botId, 'builder')
assert.equal(pickTicketUserPatch({ botId: null }).botId, null)
assert.throws(() => pickTicketUserPatch({ botId: { instructions: 'untrusted' } }), /botId/)
assert.equal(mkTicket('project', { botId: 'builder' }).botId, 'builder')
settings.agentPresets[0].executionTarget = 'workspace'
settings.agentPresets[0].toolScope = { mode: 'selected', names: ['read_file'] }
const scopedBot = captureTicketBot(ticket, 'build', settings)!
assert.deepEqual(scopedBot.toolScope, { mode: 'selected', names: ['read_file'] })
settings.agentPresets[0].instructions = 'New instructions'
assert.equal(scopedBot.instructions, 'Summarize concisely')
settings.agentPresets[0].archivedAt = 1
assert.throws(() => captureTicketBot(ticket, 'build', settings), /Restore/)
settings.agentPresets[0].archivedAt = null
settings.agentPresets[0].model = 'missing'
assert.throws(() => captureTicketBot(ticket, 'build', settings), /model is unavailable/)

console.log('ticket bot assignment checks passed')
