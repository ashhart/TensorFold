import { commitOnboarding, type OnboardingDesktopMode } from '../components/onboarding-model'
import { firstFullOnboardingLayout, type FullWorkspaceLayout } from './full-workspace-layout'

/** A new Full install opens an idle chat through the existing action, never a run. */
export async function finishOnboardingLanding<T>(input: {
  mode: OnboardingDesktopMode
  previouslyCompleted: boolean
  savedLayout: FullWorkspaceLayout | null
  preferenceReadable?: boolean
  persist(mode: OnboardingDesktopMode, complete: true): Promise<T>
  applySaved(settings: T): void
  applyLayout(layout: FullWorkspaceLayout): void
  newChat(): Promise<void>
  home(): void
  chatFailed(): void
}): Promise<void> {
  const newFull = input.mode === 'full' && !input.previouslyCompleted
  await commitOnboarding(input.mode, input.persist, (saved) => {
    input.applySaved(saved)
    const layout = newFull ? firstFullOnboardingLayout(input.previouslyCompleted, input.savedLayout, input.preferenceReadable) : null
    if (layout) input.applyLayout(layout)
  })
  if (newFull) {
    try { await input.newChat() }
    catch { input.home(); input.chatFailed() }
  } else input.home()
}
