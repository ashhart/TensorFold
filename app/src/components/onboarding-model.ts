import type { IconName } from '../lib/icons'
import type { Provider } from '../lib/types'

export const ONBOARDED_KEY = 'vontra.onboarded'
export const ONBOARDING_DRAFT_KEY = 'tensorfold.onboarding-draft'
export const ONBOARDING_COMPLETED_EVENT = 'vontra:onboarding-completed'
export const ONBOARDING_RESTART_EVENT = 'tensorfold:onboarding-restart'
export const ONBOARDING_PRIOR_COMPLETION_KEY = 'tensorfold.onboarding-completed-once'
export type OnboardingDesktopMode = 'minimal' | 'full'

export function onboardingProviderVisible(mode: OnboardingDesktopMode, providerId: string): boolean {
  return mode === 'full' || providerId === 'tensorfold'
}

export function completeOnboarding(): void {
  try { localStorage.setItem(ONBOARDED_KEY, '1'); localStorage.setItem(ONBOARDING_PRIOR_COMPLETION_KEY, '1'); localStorage.removeItem(ONBOARDING_DRAFT_KEY) } catch { /* Native settings also persist completion. */ }
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(ONBOARDING_COMPLETED_EVENT))
}

export const ONBOARDING_STEPS = [
  {
    id: 'mode',
    label: 'Your workspace',
    title: 'How will you use TensorFold?',
    desc: 'Choose your starting point. You can switch between Minimal and Full in Settings whenever you need.',
    icon: 'monitor',
  },
  {
    id: 'connect',
    label: 'Your model',
    title: 'Choose a model for this Mac',
    desc: 'The built-in TensorFold engine runs supported models locally. Start with a model that fits your hardware, or use one you already have.',
    icon: 'cpu',
  },
  {
    id: 'guide',
    label: 'Get started',
    title: 'A few things before you start',
    desc: 'Your model and tools stay in reach. Here is where to find them.',
    icon: 'spark',
  },
] as const satisfies ReadonlyArray<{
  id: string
  label: string
  title: string
  desc: string
  icon: IconName
}>

export type OnboardingStepId = typeof ONBOARDING_STEPS[number]['id']
export interface OnboardingDraft { mode: OnboardingDesktopMode; step: OnboardingStepId }

export function readOnboardingDraft(defaultMode: OnboardingDesktopMode = 'full'): OnboardingDraft {
  try {
    const draft = JSON.parse(localStorage.getItem(ONBOARDING_DRAFT_KEY) ?? 'null')
    if (draft && (draft.mode === 'minimal' || draft.mode === 'full') && ONBOARDING_STEPS.some((step) => step.id === draft.step)) return { mode: draft.mode, step: draft.step }
  } catch { /* Storage may be disabled or an older draft may be unreadable. */ }
  return { mode: defaultMode, step: 'mode' }
}

export function saveOnboardingDraft(draft: OnboardingDraft): void {
  try { localStorage.setItem(ONBOARDING_DRAFT_KEY, JSON.stringify(draft)) } catch { /* The open setup still retains its selection. */ }
}

/** A failed mode save must leave the first-run flow open and unfinished. */
export async function commitOnboarding<T>(mode: OnboardingDesktopMode, persist: (mode: OnboardingDesktopMode, complete: true) => Promise<T>, applySaved: (settings: T) => void): Promise<void> {
  const saved = await persist(mode, true)
  applySaved(saved)
  completeOnboarding()
}

export function restartOnboarding(): void {
  try { if (localStorage.getItem(ONBOARDED_KEY) === '1') localStorage.setItem(ONBOARDING_PRIOR_COMPLETION_KEY, '1'); localStorage.removeItem(ONBOARDED_KEY); localStorage.removeItem(ONBOARDING_DRAFT_KEY) } catch { /* The restart event works for this session. */ }
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(ONBOARDING_RESTART_EVENT))
}
export function onboardingPreviouslyCompleted(nativeComplete = false): boolean {
  if (nativeComplete) return true
  try { return localStorage.getItem(ONBOARDED_KEY) === '1' || localStorage.getItem(ONBOARDING_PRIOR_COMPLETION_KEY) === '1' } catch { return false }
}

export function onboardingStepIndex(id: OnboardingStepId): number {
  const i = ONBOARDING_STEPS.findIndex((step) => step.id === id)
  return i < 0 ? 0 : i
}

export function nextOnboardingStep(id: OnboardingStepId): OnboardingStepId {
  return ONBOARDING_STEPS[Math.min(onboardingStepIndex(id) + 1, ONBOARDING_STEPS.length - 1)].id
}

export function previousOnboardingStep(id: OnboardingStepId): OnboardingStepId {
  return ONBOARDING_STEPS[Math.max(onboardingStepIndex(id) - 1, 0)].id
}

export function shouldShowOnboarding(_providers?: Provider[], nativeComplete = false): boolean {
  if (nativeComplete) return false
  try { return localStorage.getItem(ONBOARDED_KEY) !== '1' } catch { return true }
}
