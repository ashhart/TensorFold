import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import path from 'node:path'
import {
  ONBOARDED_KEY,
  ONBOARDING_COMPLETED_EVENT,
  ONBOARDING_DRAFT_KEY,
  ONBOARDING_RESTART_EVENT,
  ONBOARDING_STEPS,
  commitOnboarding,
  nextOnboardingStep,
  onboardingStepIndex,
  onboardingProviderVisible,
  previousOnboardingStep,
  readOnboardingDraft,
  restartOnboarding,
  saveOnboardingDraft,
  shouldShowOnboarding,
  type OnboardingStepId,
} from './onboarding-model'

async function main() {
  const ids = ONBOARDING_STEPS.map((step) => step.id)
  assert.deepEqual(ids, ['mode', 'connect', 'guide'])
  assert.equal(new Set(ids).size, ids.length)
  for (const id of ids as OnboardingStepId[]) assert.equal(onboardingStepIndex(id), ids.indexOf(id))
  assert.equal(previousOnboardingStep('mode'), 'mode')
  assert.equal(nextOnboardingStep('mode'), 'connect')
  assert.equal(nextOnboardingStep('connect'), 'guide')
  assert.equal(nextOnboardingStep('guide'), 'guide')
  assert.equal(previousOnboardingStep('guide'), 'connect')
  assert.equal(onboardingProviderVisible('minimal', 'codex-cli'), false, 'Minimal never presents a cloud connection as its local server')
  assert.equal(onboardingProviderVisible('minimal', 'ollama'), false, 'Minimal is the built-in engine manager')
  assert.equal(onboardingProviderVisible('minimal', 'tensorfold'), true)
  assert.equal(onboardingProviderVisible('full', 'codex-cli'), true, 'Full retains external provider setup')

  // Readiness, rather than a provider label, remains the source of connection status.
  const source = readFileSync(path.join(__dirname, 'Onboarding.tsx'), 'utf8')
  assert.match(source, /const connectedLabel = activeProvider && readiness\?\.ready/)
  assert.match(source, /api\.getModelReadiness\(\{ providerId: activeProvider\.id \}\)/)
  assert.ok(source.indexOf('<EngineChoice ') < source.indexOf('testId="onboarding-provider-cloud"'))

  const originalStorage = (globalThis as any).localStorage
  const originalWindow = (globalThis as any).window
  try {
    const values = new Map<string, string>()
    ;(globalThis as any).localStorage = {
      getItem(key: string) { return values.get(key) ?? null },
      setItem(key: string, value: string) { values.set(key, value) },
      removeItem(key: string) { values.delete(key) },
    }
    const events = new EventTarget()
    ;(globalThis as any).window = events
    let completions = 0
    let restarts = 0
    events.addEventListener(ONBOARDING_COMPLETED_EVENT, () => { completions++ })
    events.addEventListener(ONBOARDING_RESTART_EVENT, () => { restarts++ })

    assert.equal(shouldShowOnboarding([]), true, 'fresh install sees workspace selection')
    assert.equal(shouldShowOnboarding([{ id: 'existing-provider' }] as any), true, 'connecting a provider does not finish setup')
    assert.deepEqual(readOnboardingDraft('minimal'), { mode: 'minimal', step: 'mode' })
    saveOnboardingDraft({ mode: 'minimal', step: 'connect' })
    assert.deepEqual(readOnboardingDraft(), { mode: 'minimal', step: 'connect' }, 'visiting Models preserves the pending mode and step')
    assert.equal(shouldShowOnboarding(), true, 'a draft cannot complete setup')

    for (const invalid of ['{bad json', '{"mode":"full","step":"chat"}', '{"mode":"invalid","step":"guide"}', 'null']) {
      values.set(ONBOARDING_DRAFT_KEY, invalid)
      assert.deepEqual(readOnboardingDraft('full'), { mode: 'full', step: 'mode' })
    }

    saveOnboardingDraft({ mode: 'minimal', step: 'guide' })
    let applied = false
    await assert.rejects(commitOnboarding('minimal', async () => { throw new Error('Disk full') }, () => { applied = true }), /Disk full/)
    assert.equal(applied, false)
    assert.equal(completions, 0)
    assert.equal(shouldShowOnboarding(), true, 'native persistence failure leaves setup open')
    assert.deepEqual(readOnboardingDraft(), { mode: 'minimal', step: 'guide' }, 'retry retains the chosen mode')

    let resolveSave!: (saved: { desktopMode: 'minimal' }) => void
    const pending = commitOnboarding('minimal', (mode, complete) => {
      assert.equal(mode, 'minimal')
      assert.equal(complete, true)
      return new Promise<{ desktopMode: 'minimal' }>((resolve) => { resolveSave = resolve })
    }, (saved) => {
      assert.equal(saved.desktopMode, 'minimal')
      assert.equal(completions, 0, 'apply the durable native settings before announcing completion')
      applied = true
    })
    assert.equal(shouldShowOnboarding(), true, 'pending saves cannot hide setup')
    assert.equal(completions, 0)
    resolveSave({ desktopMode: 'minimal' })
    await pending
    assert.equal(applied, true)
    assert.equal(completions, 1)
    assert.equal(values.get(ONBOARDED_KEY), '1')
    assert.equal(values.has(ONBOARDING_DRAFT_KEY), false)
    assert.equal(shouldShowOnboarding(), false)

    restartOnboarding()
    assert.equal(restarts, 1)
    assert.equal(shouldShowOnboarding(), true)
    assert.deepEqual(readOnboardingDraft('minimal'), { mode: 'minimal', step: 'mode' }, 'replay begins with mode selection')
    assert.equal(shouldShowOnboarding([], true), false, 'native completion survives renderer storage loss')
    values.set(ONBOARDED_KEY, '1')
    assert.equal(shouldShowOnboarding(), false, 'existing completed installs keep their legacy flag')

    ;(globalThis as any).localStorage = { getItem() { throw new Error('Storage blocked') }, setItem() { throw new Error('Storage blocked') }, removeItem() { throw new Error('Storage blocked') } }
    assert.equal(shouldShowOnboarding([], true), false)
    assert.equal(shouldShowOnboarding(), true)
    assert.deepEqual(readOnboardingDraft('minimal'), { mode: 'minimal', step: 'mode' })
    await commitOnboarding('full', async () => ({ desktopMode: 'full' }), () => {})
    assert.equal(completions, 2, 'native completion still announces success if local storage is unavailable')
  } finally {
    if (originalStorage === undefined) delete (globalThis as any).localStorage
    else (globalThis as any).localStorage = originalStorage
    if (originalWindow === undefined) delete (globalThis as any).window
    else (globalThis as any).window = originalWindow
  }
  console.log('Onboarding mode, durable completion, replay, and draft recovery tests passed.')
}

main().catch((error) => { console.error(error); process.exitCode = 1 })
