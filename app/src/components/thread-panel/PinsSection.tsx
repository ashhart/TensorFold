import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { useMemo } from 'react'
import { useStore } from '../../lib/store'
import { parseAnnotations, serializeAnnotations, togglePinnedDone, unpinMessage } from '../../lib/thread-annotations'
import { Icon } from '../../lib/icons'
import type { Thread } from '../../lib/types'
import { PanelSection } from './primitives'

// Pinned messages (VST-040): the user marks key turns; this lists them with a done toggle
// and scroll-to. Persisted in the serialized thread.annotations blob.
export function PinsSection({ thread }: { thread: Thread }) {
  const tx = useAuthoredText()
  const patchThread = useStore((s) => s.patchThread)
  const setView = useStore((s) => s.setView)
  const annotations = useMemo(() => parseAnnotations(thread.annotations), [thread.annotations])
  if (annotations.pinned.length === 0) return null
  const label = (itemId: string) => {
    const item = thread.timeline?.find((i) => i.id === itemId)
    const text = (item?.text || item?.title || item?.command || 'Pinned message').replace(/\s+/g, ' ').trim()
    return text.length > 64 ? `${text.slice(0, 63)}…` : text
  }
  const persist = (next: ReturnType<typeof parseAnnotations>) => void patchThread(thread.id, { annotations: serializeAnnotations(next) })
  return (
    <>
      <div className="mx-4 border-t border-line" />
      <PanelSection title={tx("Pinned")} right={<span className="rounded bg-ink-5 px-1.5 text-[12px] tabular-nums text-ink-3">{annotations.pinned.length}</span>}>
        <div className="space-y-0.5">
          {annotations.pinned.map((pin) => (
            <div key={pin.id} className="group flex items-center gap-2 rounded-lg px-1.5 py-1 text-[13px] transition-colors hover:bg-ink-5">
              <button onClick={() => persist(togglePinnedDone(annotations, pin.itemId))} className={clsx('flex h-4 w-4 shrink-0 items-center justify-center rounded border', pin.done ? 'border-positive bg-positive/10 text-positive' : 'border-line text-transparent')} title={pin.done ? tx("Done") : tx("Mark done")}>
                <Icon name="check" size={11} />
              </button>
              <button onClick={() => { setView({ name: 'thread', id: thread.id }); requestAnimationFrame(() => document.getElementById(`ti-${pin.itemId}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' })) }} className={clsx('min-w-0 flex-1 truncate text-left', pin.done ? 'text-ink-4 line-through' : 'text-ink-2 hover:text-ink')}>
                {pin.label || label(pin.itemId)}
              </button>
              <button onClick={() => persist(unpinMessage(annotations, pin.itemId))} className="hidden h-5 w-5 shrink-0 items-center justify-center rounded text-ink-4 transition-colors hover:bg-ink-5 hover:text-ink group-hover:flex" title={tx("Unpin")} aria-label={tx("Unpin")}>
                <Icon name="x" size={12} />
              </button>
            </div>
          ))}
        </div>
      </PanelSection>
    </>
  )
}
