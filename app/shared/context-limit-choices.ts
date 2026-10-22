import type { Settings } from '../electron/agent/types'

type Choices = Pick<Settings, 'contextLimit' | 'localContextLimit' | 'contextLimitExplicit' | 'localContextLimitExplicit'>

// Legacy saved numbers are conservatively explicit; fresh defaults carry false flags.
export function normalizeContextLimitChoices(next: Partial<Choices>, previous?: Partial<Choices>): Pick<Choices, 'contextLimitExplicit' | 'localContextLimitExplicit'> {
  function explicit(key: 'contextLimit' | 'localContextLimit', flag: 'contextLimitExplicit' | 'localContextLimitExplicit'): boolean {
    if (previous && next[key] !== previous[key]) return true
    if (previous?.[flag] === true) return true
    if (typeof next[flag] === 'boolean') return next[flag]
    return previous?.[flag] ?? typeof next[key] === 'number'
  }
  return {
    contextLimitExplicit: explicit('contextLimit', 'contextLimitExplicit'),
    localContextLimitExplicit: explicit('localContextLimit', 'localContextLimitExplicit'),
  }
}
