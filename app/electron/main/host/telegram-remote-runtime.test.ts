import assert from 'node:assert/strict'
import { mkdtempSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

process.env.VONTRA_HOME = mkdtempSync(path.join(os.tmpdir(), 'tensorfold-telegram-verify-'))
const { verifyTelegramToken } = await import('./telegram-remote-runtime')
const token = '123456:synthetic_token_for_verification_only'
let calls = 0
const fetchImpl = (async (input: any, init: any) => {
  calls++
  assert.equal(String(input), `https://api.telegram.org/bot${token}/getMe`)
  assert.equal(init.method, 'GET')
  assert.equal(init.redirect, 'error')
  return new Response(JSON.stringify({ ok: true, result: { id: 123456, is_bot: true, username: 'synthetic_bot' } }), { headers: { 'content-type': 'application/json' } })
}) as typeof fetch
assert.deepEqual(await verifyTelegramToken(token, fetchImpl), { ok: true, botId: '123456', username: 'synthetic_bot', error: null })
assert.equal(calls, 1, 'verification invokes only read-only getMe, never updates or messages')
assert.equal((await verifyTelegramToken('not-a-token', fetchImpl)).ok, false)
assert.equal(calls, 1, 'malformed credentials never leave the app')
const apiError = (async () => new Response(JSON.stringify({ ok: false, description: `Invalid ${token}` }), { headers: { 'content-type': 'application/json' } })) as typeof fetch
const failure = await verifyTelegramToken(token, apiError)
assert.equal(failure.ok, false)
assert.ok(!JSON.stringify(failure).includes(token), 'provider errors cannot reveal protected tokens')
const spoof = (async () => new Response(JSON.stringify({ ok: true, result: { id: 1, is_bot: false, username: 'synthetic_person' } }), { headers: { 'content-type': 'application/json' } })) as typeof fetch
assert.equal((await verifyTelegramToken(token, spoof)).ok, false)
const oversized = (async () => new Response('{}', { headers: { 'content-type': 'application/json', 'content-length': '999999' } })) as typeof fetch
assert.equal((await verifyTelegramToken(token, oversized)).ok, false)
console.log('Telegram verification is read-only, bounded, bot-specific and redacts credentials')
