import assert from 'node:assert/strict'
import { store } from '../state'
import { fanOutHostEvent, rendererWindowAvailable, sendToRenderer } from './renderer-events'

const previous = store.win
try {
  const sent: Array<[string, unknown]> = []
  store.win = {
    isDestroyed: () => false,
    webContents: {
      isDestroyed: () => false,
      send: (channel: string, payload: unknown) => { sent.push([channel, payload]) },
    },
  } as any
  assert.equal(rendererWindowAvailable(), true)
  assert.equal(sendToRenderer('safe:event', { ok: true }), true)
  assert.deepEqual(sent, [['safe:event', { ok: true }]])

  let remote = 0
  let webhook = 0
  store.win = {
    isDestroyed: () => false,
    webContents: {
      isDestroyed: () => false,
      send: () => { throw new Error('Object has been destroyed') },
    },
  } as any
  const result = fanOutHostEvent('agent:event', { secret: 'never logged' }, [
    { label: 'remote', deliver: () => { remote++ } },
    { label: 'webhook', deliver: () => { webhook++ } },
  ])
  assert.equal(result.rendererDelivered, false, 'a destruction race is contained at the renderer boundary')
  assert.equal(remote, 1, 'renderer failure cannot suppress remote fan-out')
  assert.equal(webhook, 1, 'renderer failure cannot suppress webhook fan-out')

  store.win = {
    isDestroyed: () => true,
    webContents: { isDestroyed: () => true, send: () => { throw new Error('must not call') } },
  } as any
  assert.equal(rendererWindowAvailable(), false)
  assert.equal(sendToRenderer('ignored'), false)

  let finalSink = 0
  const isolated = fanOutHostEvent('ignored', null, [
    { label: 'broken', deliver: () => { throw new Error('sink failed') } },
    { label: 'final', deliver: () => { finalSink++ } },
  ])
  assert.deepEqual(isolated.failedSinks, ['broken'])
  assert.equal(finalSink, 1, 'one downstream failure cannot suppress later sinks')
} finally {
  store.win = previous
}

console.log('renderer event lifecycle + independent fan-out tests passed')
