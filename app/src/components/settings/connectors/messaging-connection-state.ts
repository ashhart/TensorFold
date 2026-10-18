export interface MessagingVerification { ok: boolean; identity?: string; error?: string }
export interface MessagingActionState {
  phase: 'idle' | 'verifying' | 'verified' | 'loading' | 'syncing' | 'saving' | 'error'
  verified: boolean
  message: string
}

// An editor owns one controller per source; no token or cross-account result is stored here.
export function createMessagingConnectionState(changed: (state: MessagingActionState) => void) {
  let revision = 0
  let disposed = false
  let state: MessagingActionState = { phase: 'idle', verified: false, message: '' }
  const publish = (next: MessagingActionState) => { state = next; if (!disposed) changed(next) }
  return {
    snapshot: () => state,
    activate: () => { disposed = false },
    invalidate: (keepVerification = false, message = '') => { revision++; publish({ phase: keepVerification && state.verified ? 'verified' : 'idle', verified: keepVerification && state.verified, message }) },
    dispose: () => { disposed = true; revision++ },
    async verify(save: () => Promise<void>, verify: () => Promise<MessagingVerification>) {
      const ownRevision = ++revision
      publish({ phase: 'verifying', verified: false, message: 'Saving and checking this connection…' })
      try {
        await save()
        if (disposed || ownRevision !== revision) return
        const result = await verify()
        if (disposed || ownRevision !== revision) return
        publish(result.ok
          ? { phase: 'verified', verified: true, message: result.identity ? `Verified as ${result.identity}` : 'Token verified' }
          : { phase: 'error', verified: false, message: result.error || 'The service rejected this token.' })
      } catch {
        if (!disposed && ownRevision === revision) publish({ phase: 'error', verified: false, message: 'Could not save or verify this connection; check the token, permissions and network, then retry.' })
      }
    },
    async action(phase: 'loading' | 'syncing' | 'saving', work: () => Promise<string>, requiresVerification = true) {
      if (disposed || (requiresVerification && !state.verified)) return false
      const ownRevision = ++revision
      const verified = state.verified
      publish({ phase, verified, message: phase === 'syncing' ? 'Reading selected channels…' : phase === 'loading' ? 'Loading available channels…' : 'Saving…' })
      try {
        const message = await work()
        if (disposed || ownRevision !== revision) return false
        publish({ phase: verified ? 'verified' : 'idle', verified, message })
        return true
      } catch {
        if (!disposed && ownRevision === revision) publish({ phase: 'error', verified: false, message: 'This action failed; check the connection and service permissions, then verify again.' })
        return false
      }
    },
  }
}

export function messagingChannelEntries(value: string): string[] {
  return [...new Set(value.split(/[\s,;]+/).map((entry) => entry.replace(/^#/, '').trim()).filter(Boolean))]
}

export function messagingChannelScopeError(value: string): string | null {
  const entries = messagingChannelEntries(value)
  if (!entries.length) return 'Choose at least one channel before connecting; an empty list grants no access.'
  if (entries.includes('*') && value.trim() !== '*') return 'Use * alone to explicitly allow all channels, or select individual channels.'
  return null
}

export function telegramAccessError(chats: string, users: string): string | null {
  const chatIds = messagingChannelEntries(chats)
  const userIds = messagingChannelEntries(users)
  if (!chatIds.length || chatIds.some((id) => !/^-?[1-9]\d*$/.test(id))) return 'Add numeric chat IDs; groups may have a negative ID, and an empty list allows no chats.'
  if (!userIds.length || userIds.some((id) => !/^[1-9]\d*$/.test(id))) return 'Add positive numeric sender IDs; every command must match both a chat and a sender.'
  return null
}
