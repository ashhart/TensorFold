import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import {
  ONBOARDED_KEY,
  ONBOARDING_STEPS,
  nextOnboardingStep,
  onboardingStepIndex,
  previousOnboardingStep,
  shouldShowOnboarding,
  type OnboardingStepId,
} from './onboarding-model'

const ids = ONBOARDING_STEPS.map((step) => step.id)

assert.deepEqual(ids, ['connect', 'chat', 'assistant', 'code', 'settings'])
assert.equal(new Set(ids).size, ids.length, 'onboarding step ids must stay unique')
assert.equal(ONBOARDING_STEPS[0].title, 'Connect a model')
assert.equal(ONBOARDING_STEPS.at(-1)?.label, 'Readiness')
assert.match(ONBOARDING_STEPS[0].desc, /built-in TensorFold engine/, 'step 1 leads with the engine inside the app')

// "Connected" must come from the readiness check, never from the active provider's name alone.
const onboardingSource = readFileSync(path.join(__dirname, 'Onboarding.tsx'), 'utf8')
assert.match(onboardingSource, /const connectedLabel = activeProvider && readiness\?\.ready/)
assert.match(onboardingSource, /api\.getModelReadiness\(\{ providerId: activeProvider\.id \}\)/)
assert.match(onboardingSource, /<EngineChoice /, 'the built-in engine is offered before cloud and local providers')
assert.ok(onboardingSource.indexOf('<EngineChoice ') < onboardingSource.indexOf('testId="onboarding-provider-cloud"'))

for (const id of ids as OnboardingStepId[]) {
  assert.equal(onboardingStepIndex(id), ids.indexOf(id), `index for ${id}`)
}

assert.equal(previousOnboardingStep('connect'), 'connect', 'back clamps at first step')
assert.equal(nextOnboardingStep('connect'), 'chat')
assert.equal(nextOnboardingStep('chat'), 'assistant')
assert.equal(nextOnboardingStep('assistant'), 'code')
assert.equal(nextOnboardingStep('code'), 'settings')
assert.equal(nextOnboardingStep('settings'), 'settings', 'next clamps at final step')

const originalLocalStorage = (globalThis as any).localStorage
try {
  let stored: string | null = null
  ;(globalThis as any).localStorage = {
    getItem(key: string) { return key === ONBOARDED_KEY ? stored : null },
    setItem(key: string, value: string) { if (key === ONBOARDED_KEY) stored = value },
  }

  stored = null
  assert.equal(shouldShowOnboarding([] as any), true, 'fresh install shows onboarding')
  stored = '1'
  assert.equal(shouldShowOnboarding([{ id: 'p1' }] as any), false, 'completion flag wins even when providers exist')
} finally {
  if (originalLocalStorage === undefined) delete (globalThis as any).localStorage
  else (globalThis as any).localStorage = originalLocalStorage
}

console.log('Onboarding contract tests passed.')
