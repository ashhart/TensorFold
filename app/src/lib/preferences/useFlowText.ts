import { useStore } from '../store'
import { flowTextFor } from './flow-catalogue'

export function useFlowText() {
  const locale = useStore(state => state.settings?.uiPreferences?.locale)
  return flowTextFor(locale)
}
