import { useEffect, useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import type { Thread } from '../../lib/types'
import { PanelSection } from './primitives'

// Notepad: free-form per-thread notes, saved with the thread (debounced).
export function NotepadSection({ thread }: { thread: Thread }) {
  const patchThread = useStore((s) => s.patchThread)
  const [draft, setDraft] = useState(thread.notepad ?? '')
  const savedRef = useRef(thread.notepad ?? '')
  // Reset the draft only when switching threads — not on every patch echo.
  useEffect(() => { setDraft(thread.notepad ?? ''); savedRef.current = thread.notepad ?? '' }, [thread.id])
  useEffect(() => {
    if (draft === savedRef.current) return
    const t = setTimeout(() => { savedRef.current = draft; void patchThread(thread.id, { notepad: draft }) }, 600)
    return () => clearTimeout(t)
  }, [draft, thread.id, patchThread])
  return (
    <PanelSection title="Notepad">
      <textarea
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        placeholder="Type here"
        rows={4}
        className="scroll-thin w-full resize-y rounded-lg border border-line bg-transparent px-2.5 py-2 text-[13px] leading-snug text-ink placeholder:text-ink-4 focus:border-line-strong focus:outline-none"
      />
    </PanelSection>
  )
}
