import { useEffect, useRef, useState } from 'react'

export interface MessagingPersistedState {
  draftKey: string
  persistedKey: string | null
  enabled: boolean
}

export function liveMessagingDraftKey(source: { id: string; label: string; token?: string; channels?: string; enabled: boolean; channelPrompt?: string; credentialVersion?: number | null }): string {
  return JSON.stringify([source.id, source.label, source.token ?? '', source.channels ?? '', source.enabled, source.channelPrompt ?? '', source.credentialVersion ?? null])
}

export function telegramMessagingDraftKey(source: { enabled: boolean; botToken: string; allowedChatIds: string; allowedUserIds: string; polling: boolean; secretRotations?: Record<string, number | null> }): string {
  return JSON.stringify([source.enabled, source.botToken, source.allowedChatIds, source.allowedUserIds, source.polling, source.secretRotations?.botToken ?? null])
}

// Settings edits are local until the awaited host save completes; do not describe
// a pending disabled draft as an already disconnected server connection.
export function useMessagingDraftState<T extends { enabled?: boolean }>(enabled: boolean, save: (patch?: T) => Promise<void>, persisted?: MessagingPersistedState) {
  const committedEnabled = useRef(enabled)
  const [dirty, setDirty] = useState(false)
  useEffect(() => { if (!dirty) committedEnabled.current = enabled }, [enabled, dirty])
  return {
    // Source config equality reflects global Save/Discard; sync/status metadata and
    // other accounts cannot accidentally clear this source's unfinished edits.
    dirty: persisted ? persisted.draftKey !== persisted.persistedKey : dirty,
    persistedEnabled: persisted ? persisted.enabled : committedEnabled.current,
    edited: () => setDirty(true),
    async save(patch?: T) {
      await save(patch)
      committedEnabled.current = patch?.enabled ?? enabled
      setDirty(false)
    },
  }
}
