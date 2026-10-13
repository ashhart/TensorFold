import type { IconName } from '../lib/icons'
import type { Provider } from '../lib/types'

export const ONBOARDED_KEY = 'vontra.onboarded'
export const ONBOARDING_COMPLETED_EVENT = 'vontra:onboarding-completed'

export function completeOnboarding(): void {
  try { localStorage.setItem(ONBOARDED_KEY, '1') } catch { /* session completion still works */ }
  if (typeof window !== 'undefined') window.dispatchEvent(new Event(ONBOARDING_COMPLETED_EVENT))
}

export const ONBOARDING_STEPS = [
  {
    id: 'connect',
    label: 'Connect',
    title: 'Connect a model',
    desc: 'Run a model on this Mac with the built-in TensorFold engine, or connect a cloud account, CLI subscription or another local runtime.',
    icon: 'cpu',
  },
  {
    id: 'chat',
    label: 'Chat',
    title: 'Chat stays lightweight',
    desc: 'Use Chat for direct model conversations, quick prompts, and threads that do not need a project workspace.',
    icon: 'message',
  },
  {
    id: 'assistant',
    label: 'Assistant',
    title: 'Assistant is your connected workspace',
    desc: 'Use Assistant for connectors, memory proposals, skills, voice, automations, and cross-app context.',
    icon: 'spark',
  },
  {
    id: 'code',
    label: 'Code',
    title: 'Code is the workbench',
    desc: 'Use Code for projects, worktrees, rooms, tickets, terminals, diffs, tests, and long-running jobs.',
    icon: 'code',
  },
  {
    id: 'settings',
    label: 'Readiness',
    title: 'Verify the whole harness',
    desc: 'Run Readiness before serious work so providers, CLIs, connectors, remote hosts, terminals, and security are explicit.',
    icon: 'check',
  },
] as const satisfies ReadonlyArray<{
  id: string
  label: string
  title: string
  desc: string
  icon: IconName
}>

export type OnboardingStepId = typeof ONBOARDING_STEPS[number]['id']

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

export function shouldShowOnboarding(_providers?: Provider[]): boolean {
  try { return localStorage.getItem(ONBOARDED_KEY) !== '1' } catch { return true }
}
