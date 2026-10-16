import assert from 'node:assert/strict'
import { removeWhatsAppSourceWithNativeConfirmation } from './connector-runtime'

const source = { id: 'wa-1', label: 'Support phone' }

for (const confirm of [undefined, async () => false] as const) {
  const effects: string[] = []
  await assert.rejects(
    () => removeWhatsAppSourceWithNativeConfirmation(source, confirm, {
      stop: () => effects.push('stop'),
      shred: async () => { effects.push('shred') },
      remove: async () => { effects.push('remove') },
    }),
    /cancelled|unavailable/i,
  )
  assert.deepEqual(effects, [], 'missing/declined native confirmation fails closed before bridge stop or credential shred')
}

const effects: string[] = []
await removeWhatsAppSourceWithNativeConfirmation(source, async (request) => {
  effects.push('confirm')
  assert.match(request.message, /Support phone/)
  assert.match(request.detail, /permanently shreds/i)
  return true
}, {
  stop: () => effects.push('stop'),
  shred: async () => { effects.push('shred') },
  remove: async () => { effects.push('remove') },
})
assert.deepEqual(effects, ['confirm', 'stop', 'shred', 'remove'], 'native confirmation completes before every destructive effect')

console.log('WhatsApp source removal native-confirmation ordering tests passed')
