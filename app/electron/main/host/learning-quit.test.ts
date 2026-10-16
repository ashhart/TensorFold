import assert from 'node:assert/strict'
import { createLearningQuitGuard } from './learning-quit'
async function main() {
  let busy = true
  let complete!: () => void
  let preparations = 0, quits = 0, prevented = 0, failures = 0
  const settled = new Promise<void>((resolve) => { complete = resolve })
  const guard = createLearningQuitGuard({ service: () => ({ isLearningBusy: () => busy, prepareToQuit: () => { preparations++; return settled } }), quit: () => { quits++ }, failed: () => { failures++ } })
  const event = { preventDefault: () => { prevented++ } }
  assert.equal(guard(event), true)
  assert.equal(guard(event), true)
  busy = false // Save finished, but the pending graceful-quit promise is not resolved yet.
  assert.equal(guard(event), true)
  assert.equal(preparations, 1)
  assert.equal(quits, 0)
  assert.equal(prevented, 3)
  complete(); await settled; await Promise.resolve()
  assert.equal(quits, 1)
  assert.equal(guard(event), false, 'normal teardown may proceed only after safe completion')
  const refused = createLearningQuitGuard({ service: () => ({ isLearningBusy: () => true, prepareToQuit: async () => { throw new Error('synthetic save failure') } }), quit: () => { quits++ }, failed: () => { failures++ } })
  assert.equal(refused(event), true)
  await Promise.resolve(); await Promise.resolve()
  assert.equal(failures, 1)
  assert.equal(quits, 1, 'save failure cannot force an unsafe quit')
  console.log('Living Weights graceful quit: single save fence, repeated quits, failure and normal teardown passed.')
}
main().catch((error) => { console.error(error); process.exit(1) })
