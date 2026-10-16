import { useStore } from '../lib/store'
import { useEngine } from '../lib/engine'
import { Icon } from '../lib/icons'
import type { Thread } from '../lib/types'

// Thread header badge while the built-in engine serves this thread with Cortheon: its replies
// may search the web, so the thread is no longer fully on this Mac.
export function CortheonBadge({ thread }: { thread: Thread }) {
  const builtIn = useStore((s) => {
    const id = thread.providerId ?? s.settings?.activeProviderId
    return s.settings?.providers.find((candidate) => candidate.id === id)?.source === 'tensorfold'
  })
  const { snapshot } = useEngine()
  if (!builtIn || thread.orchestratorHostId || thread.workspace?.kind === 'ssh' || !snapshot?.status.cortheon?.active) return null
  return (
    <span title="Cortheon is on: before answering, the engine may send search queries built from this chat to a web search service and read public pages. Turn it off in AI settings or Models." className="flex shrink-0 items-center gap-1 rounded-full border border-line px-2 py-0.5 text-[13px] leading-4 text-info-ink">
      <Icon name="globe" size={12} className="text-info" />
      Cortheon
    </span>
  )
}
