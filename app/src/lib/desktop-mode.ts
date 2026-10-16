import { api } from './ipc'
import { useStore } from './store'
import { shouldShowOnboarding } from '../components/onboarding-model'

export async function changeDesktopMode(mode: 'minimal' | 'full'): Promise<void> {
  const completed = !shouldShowOnboarding(undefined, useStore.getState().settings?.desktopOnboardingComplete)
  const settings = await api.desktopModeSet(mode, completed ? true : undefined)
  useStore.setState({ settings })
}
