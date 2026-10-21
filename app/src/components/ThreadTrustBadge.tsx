import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useStore } from '../lib/store'
import { Icon } from '../lib/icons'
import { classifyProvider } from '../../shared/provider-trust'
import { api } from '../lib/ipc'
import { threadTrustSource } from '../lib/bot-labels'
import type { Thread } from '../lib/types'

// Thread header badge: work on a trusted local provider never leaves this Mac. External
// providers keep their "Planning only" treatment in the composer instead.
export function ThreadTrustBadge({ thread }: { thread: Thread }) {
  const tx = useAuthoredText()
  const provider = useStore((s) => {
    const id = thread.providerId ?? s.settings?.activeProviderId
    return s.settings?.providers.find((candidate) => candidate.id === id) ?? null
  })
  if (!provider || classifyProvider(provider) !== 'trusted-local') return null
  return (
    <span title={tx("This thread runs on a model on this Mac. Nothing is sent to a cloud provider.")} className="flex shrink-0 items-center gap-1 rounded-full border border-line px-2 py-0.5 text-[13px] leading-4 text-ink-3">
      <Icon name="monitor" size={12} className="text-ink-4" />
      {tx(threadTrustSource(api.platform))}</span>
  )
}
