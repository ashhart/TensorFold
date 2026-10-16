import assert from 'node:assert/strict'
import { finishOnboardingLanding } from './onboarding-landing'
import { ONBOARDED_KEY, ONBOARDING_PRIOR_COMPLETION_KEY, onboardingPreviouslyCompleted, restartOnboarding } from '../components/onboarding-model'

const previousStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');const previousWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
const values = new Map<string, string>()
Object.defineProperty(globalThis, 'localStorage', { value: { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => values.set(key, value), removeItem: (key: string) => values.delete(key) }, configurable: true })
Object.defineProperty(globalThis, 'window', { value: new EventTarget(), configurable: true })
try {
  const calls: string[] = []
  const deps = { mode: 'full' as const, previouslyCompleted: false, savedLayout: null, persist: async (mode: string, complete: true) => { calls.push('persist');assert.equal(mode, 'full');assert.equal(complete, true);return { desktopMode: mode } }, applySaved: () => { calls.push('apply') }, applyLayout: () => { calls.push('layout') }, newChat: async () => { calls.push('idle-chat') }, home: () => { calls.push('home') }, chatFailed: () => { calls.push('chat-error') } }
  await assert.rejects(() => finishOnboardingLanding({ ...deps, persist: async () => { throw new Error('Synthetic disk failure') } }), /disk failure/)
  assert.deepEqual([...calls], [], 'native save failure cannot change layout, complete setup or create a chat')
  assert.equal(values.has(ONBOARDED_KEY), false)
  await finishOnboardingLanding(deps);assert.deepEqual([...calls], ['persist', 'apply', 'layout', 'idle-chat']);assert.equal(values.get(ONBOARDING_PRIOR_COMPLETION_KEY), '1')
  calls.length = 0;await finishOnboardingLanding({ ...deps, previouslyCompleted: true });assert.deepEqual([...calls], ['persist', 'apply', 'home'], 'existing-user setup replay does not reset layout or create work')
  calls.length = 0;await finishOnboardingLanding({ ...deps, savedLayout: 'workspace' });assert.deepEqual([...calls], ['persist', 'apply', 'idle-chat'], 'first Full still lands in Chat while honoring an explicit workspace layout')
  calls.length = 0;await finishOnboardingLanding({ ...deps, preferenceReadable: false });assert.deepEqual([...calls], ['persist', 'apply', 'idle-chat'], 'unreadable existing preference is preserved')
  calls.length = 0;await finishOnboardingLanding({ ...deps, mode: 'minimal', persist: async (mode) => { calls.push('persist');assert.equal(mode, 'minimal');return { desktopMode: mode } } });assert.deepEqual([...calls], ['persist', 'apply', 'home'], 'Minimal never creates a chat or changes the Full layout')
  calls.length = 0;await finishOnboardingLanding({ ...deps, newChat: async () => { calls.push('idle-chat');throw new Error('Synthetic create failure') } });assert.deepEqual([...calls], ['persist', 'apply', 'layout', 'idle-chat', 'home', 'chat-error'], 'saved setup stays truthful and gives an explicit retry path if idle chat creation fails')
  values.clear();values.set(ONBOARDED_KEY, '1');assert.equal(onboardingPreviouslyCompleted(), true);restartOnboarding()
  assert.equal(values.has(ONBOARDED_KEY), false);assert.equal(values.get(ONBOARDING_PRIOR_COMPLETION_KEY), '1');assert.equal(onboardingPreviouslyCompleted(), true, 'legacy local-only completion survives replay so existing layouts are preserved')
  assert.equal(onboardingPreviouslyCompleted(true), true)
  console.log('First Full idle-chat landing, preserved explicit/legacy layouts, native failure, chat retry and unchanged Minimal onboarding passed')
} finally { if (previousStorage) Object.defineProperty(globalThis, 'localStorage', previousStorage);else delete (globalThis as any).localStorage;if (previousWindow) Object.defineProperty(globalThis, 'window', previousWindow);else delete (globalThis as any).window }
