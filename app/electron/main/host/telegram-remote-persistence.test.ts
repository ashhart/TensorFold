import assert from 'node:assert/strict'
import { mkdtemp, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import type { Thread } from '../../agent/types'

const home = await mkdtemp(path.join(os.tmpdir(), 'tensorfold-telegram-retry-'))
process.env.VONTRA_HOME = home
const { defaultSettings, saveSettings, loadSettings, writeChains, closeSessionSearchIndexes } = await import('../../store')
const { store } = await import('../state')
const { createBotPollingRuntime } = await import('./bot-polling-runtime')
const { createTelegramRemoteService } = await import('../../agent/telegram-remote/service')
const { telegramAuthorizationRevision } = await import('../../agent/telegram-remote/policy')
const settings = defaultSettings()
settings.botControl.telegram = { ...settings.botControl.telegram, enabled: true, botToken: 'synthetic-token', allowedChatIds: '7 8', allowedUserIds: '7 8', polling: false,
  remote: { enabled: true, scope: 'selected', threadIds: [], roomIds: [], revision: 1, sessions: [] } }
store.settings = settings
let failNext = true
let runs = 0
const offsets: Array<string | null> = []
const replies: string[] = []
const update = { update_id: 1, message: { chat: { id: 7, type: 'private' }, from: { id: 7, is_bot: false }, text: 'Synthetic task after disk recovery' } }
const remote = createTelegramRemoteService({ settings: () => store.settings.botControl.telegram, threads: () => [], rooms: () => [],
  persist: async () => { if (failNext) { failNext = false; throw new Error('Synthetic transient protected-save failure') } await saveSettings(store.settings) },
  dispatchTurn: async () => { runs++; return { status: 'completed', text: 'Synthetic saved reply' } },
})
const polling = createBotPollingRuntime({ telegramRemote: remote,
  telegramFetch: (async (input: any, init: any) => {
    const url = new URL(String(input))
    if (url.pathname.endsWith('/getUpdates')) {
      offsets.push(url.searchParams.get('offset'))
      return Response.json({ ok: true, result: [update] })
    }
    replies.push(JSON.parse(init.body).text)
    return Response.json({ ok: true })
  }) as typeof fetch,
  botControlRuntime: () => { throw new Error('Legacy routing is forbidden in remote mode') }, resolveChannelApproval: () => false, resolveChannelApprovalByThread: () => false,
})
try {
  const failed = await polling.pollTelegramUpdates()
  assert.equal(failed.ok, false)
  assert.equal(runs, 0)
  assert.deepEqual(replies, [])
  assert.equal(store.settings.botControl.telegram.lastUpdateId, null, 'pre-dispatch persistence failure must not acknowledge the Telegram delivery')
  assert.equal(store.settings.botControl.telegram.remote!.sessions.length, 0, 'a failed owner checkpoint must be rolled back in memory too')
  assert.equal((await loadSettings()).botControl.telegram.lastUpdateId, null, 'restart retains the unacknowledged cursor after a failed admission')
  assert.equal((await polling.pollTelegramUpdates()).ok, true)
  assert.equal(runs, 1, 'the retained delivery executes once after protected persistence recovers')
  assert.deepEqual(replies, ['Synthetic saved reply'])
  assert.deepEqual(offsets, [null, null])
  assert.equal(store.settings.botControl.telegram.lastUpdateId, 1)
  const recovered = await loadSettings()
  assert.equal(recovered.botControl.telegram.lastUpdateId, 1)
  assert.equal(recovered.botControl.telegram.remote!.sessions[0].lastUpdateId, 1, 'the successful owner replay checkpoint survives protected-store reload')
  assert.equal((await polling.pollTelegramUpdates()).processed, 0, 'the successfully admitted delivery cannot execute again')
  assert.equal(runs, 1)

  const telegram = store.settings.botControl.telegram
  telegram.lastUpdateId = null
  telegram.remote!.threadIds = ['selected']
  const selected = { id: 'selected', kind: 'chat', title: 'Synthetic selected chat', timeline: [], apiMessages: [] } as unknown as Thread
  telegram.remote!.sessions = [{ ownerKey: 'telegram:7:7', authorizationRevision: telegramAuthorizationRevision(telegram), target: { kind: 'chat', id: 'selected' }, lastUpdateId: 2, updatedAt: 1, recentMessageTimes: [] }]
  const snapshot = JSON.stringify(telegram.remote!.sessions)
  const failExisting = createTelegramRemoteService({ settings: () => telegram, threads: () => [selected], rooms: () => [], persist: async () => { throw new Error('Synthetic failed existing-owner checkpoint') }, dispatchTurn: async () => { throw new Error('No dispatch after failed save') } })
  await assert.rejects(() => failExisting.ingest({ ...update, update_id: 3, message: { ...update.message, text: '/chief' } }), /failed existing-owner checkpoint/)
  assert.equal(JSON.stringify(telegram.remote!.sessions), snapshot, 'failure preserves the prior owner selection, replay cursor and limiter exactly')

  telegram.remote!.sessions = []
  telegram.remote!.threadIds = []
  const onceOnly = createTelegramRemoteService({ settings: () => store.settings.botControl.telegram, threads: () => [], rooms: () => [], persist: async () => {}, dispatchTurn: async () => { runs++; return { status: 'completed', text: 'Synthetic answer whose delivery fails' } } })
  const failedDelivery = createBotPollingRuntime({ telegramRemote: onceOnly,
    telegramFetch: (async (input: any) => String(input).includes('/getUpdates') ? Response.json({ ok: true, result: [update] }) : Response.json({ ok: false }, { status: 503 })) as typeof fetch,
    botControlRuntime: () => { throw new Error('No legacy fallback') }, resolveChannelApproval: () => false, resolveChannelApprovalByThread: () => false,
  })
  const deliveryError = await failedDelivery.pollTelegramUpdates()
  assert.equal(deliveryError.ok, false)
  assert.equal(deliveryError.processed, 1)
  assert.equal(store.settings.botControl.telegram.lastUpdateId, 1, 'post-dispatch reply failure must still acknowledge the executed delivery')
  assert.match(deliveryError.error ?? '', /delivery failed/)
  assert.equal((await failedDelivery.pollTelegramUpdates()).processed, 0)
  assert.equal(runs, 2, 'delivery failure cannot replay an already admitted model run')
  console.log('Telegram protected-save failure retains the unacknowledged delivery, rolls back owner state, retries once after recovery and preserves the prior replay barrier')
} finally {
  await Promise.allSettled([...writeChains.values()])
  closeSessionSearchIndexes()
  await rm(home, { recursive: true, force: true })
}
