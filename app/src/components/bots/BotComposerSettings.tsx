import { createContext, useContext, type ReactNode } from 'react'
import type { BotThreadConfig } from '../../lib/types'

const BotComposerContext = createContext<BotThreadConfig | undefined>(undefined)

export function BotComposerSettings({ bot, children }: { bot?: BotThreadConfig; children: ReactNode }) {
  return <BotComposerContext.Provider value={bot}>{children}</BotComposerContext.Provider>
}

export function useBotComposerSettings() {
  return useContext(BotComposerContext)
}
