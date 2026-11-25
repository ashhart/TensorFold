import { createContext, useContext, type ReactNode } from 'react'
import type { BotThreadConfig } from '../../lib/types'

type CapturedComposerSettings = Pick<BotThreadConfig, 'reasoningEffort'>
const BotComposerContext = createContext<CapturedComposerSettings | undefined>(undefined)

export function BotComposerSettings({ bot, children }: { bot?: CapturedComposerSettings; children: ReactNode }) {
  return <BotComposerContext.Provider value={bot}>{children}</BotComposerContext.Provider>
}

export function useBotComposerSettings() {
  return useContext(BotComposerContext)
}
