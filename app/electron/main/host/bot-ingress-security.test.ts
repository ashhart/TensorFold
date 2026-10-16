import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { defaultSettings } from '../../store'
import { store } from '../state'
import { botChannelAllowed } from './bot-control'

store.settings = defaultSettings()
store.settings.botControl.telegram = {
  ...store.settings.botControl.telegram,
  enabled: true,
  botToken: 'token',
  polling: true,
  allowedChatIds: '-100,42',
  allowedUserIds: '7',
}

assert.equal(botChannelAllowed({ channel: 'telegram', chatId: '-100', senderId: '7', text: 'rooms' }), true)
assert.equal(botChannelAllowed({ channel: 'telegram', chatId: '-100', senderId: '8', text: 'rooms' }), false, 'an allowlisted group does not authorize another member')
assert.equal(botChannelAllowed({ channel: 'telegram', chatId: '-200', senderId: '7', text: 'rooms' }), false, 'an allowlisted user still needs an allowlisted chat')
assert.equal(botChannelAllowed({ channel: 'telegram', chatId: '42', text: 'rooms' }), false, 'missing authenticated sender fails closed')

store.settings.botControl.whatsapp = {
  ...store.settings.botControl.whatsapp,
  enabled: true,
  accessToken: 'access',
  phoneNumberId: 'phone',
  verifyToken: 'verify',
  appSecret: 'app-secret',
  allowedChatIds: '447700900123',
}
assert.equal(botChannelAllowed({ channel: 'whatsapp', chatId: '447700900123', senderId: '447700900123', text: 'rooms' }), true)
assert.equal(botChannelAllowed({ channel: 'whatsapp', chatId: '447700900123', senderId: '447700900999', text: 'rooms' }), false, 'a bridge cannot supply an allowlisted chat alias for another signed sender')

for (const file of ['electron/preload.ts', 'src/lib/ipc.ts', 'electron/main/host/core-ipc.ts']) {
  const source = readFileSync(file, 'utf8')
  assert.ok(!source.includes("'bot:ingest'") && !source.includes('ingestBotControl:'), `${file} must not expose bot ingestion to renderer code`)
}

console.log('bot ingress security tests passed')
