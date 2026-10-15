import { useMemo, useState } from 'react'
import { buildThreadRecap } from '../../lib/thread-recap'
import type { Thread } from '../../lib/types'
import { PanelSection } from './primitives'

// Generated recap (VST-040): a deterministic "what happened here" summary from the timeline.
// Refreshes as the thread grows; collapsible so it stays out of the way.
export function RecapSection({ thread }: { thread: Thread }) {
  const recap = useMemo(() => buildThreadRecap(thread.timeline ?? []), [thread.timeline])
  const [open, setOpen] = useState(false)
  if (!recap.text) return null
  return (
    <>
      <div className="mx-4 border-t border-line" />
      <PanelSection
        title="Recap"
        right={<button onClick={() => setOpen((v) => !v)} className="text-[12px] font-medium text-ink-3 transition-colors hover:text-ink">{open ? 'Hide' : 'Show'}</button>}
      >
        {recap.headline && <div className="px-1 text-[13px] leading-snug text-ink-2">{recap.headline}</div>}
        <div className="mt-1 px-1 text-[12px] text-ink-4">
          {[recap.turns ? `${recap.turns} turn${recap.turns === 1 ? '' : 's'}` : '', recap.filesChanged.length ? `${recap.filesChanged.length} changed` : '', recap.commands ? `${recap.commands} command${recap.commands === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ')}
        </div>
        {open && (
          <div className="mt-2 space-y-1.5 px-1">
            {recap.filesChanged.length > 0 && (
              <div className="text-[13px] text-ink-3"><span className="font-medium text-ink-2">Changed:</span> {recap.filesChanged.slice(0, 10).join(', ')}{recap.filesChanged.length > 10 ? `, +${recap.filesChanged.length - 10}` : ''}</div>
            )}
            {recap.lastError && <div className="text-[13px] text-danger">Last error: {recap.lastError}</div>}
          </div>
        )}
      </PanelSection>
    </>
  )
}
