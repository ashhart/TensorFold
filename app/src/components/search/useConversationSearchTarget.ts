import { scrollTranscriptTarget } from './scroll-transcript-target'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useLayoutEffect, useRef, useState, type RefObject } from 'react'
import type { TimelineItem } from '../../lib/types'
import type { TurnRow } from '../../lib/turn-blocks'
import { consumeConversationSearchTarget, REVEAL_CONVERSATION_MATCH, type SearchRevealRequest } from './search-events'

/** An exact ID is a navigation hint, never a transcript mutation or model instruction. */
export function useConversationSearchTarget(input: {
  threadId: string
  timeline: TimelineItem[]
  rows: TurnRow[]
  scrollRef: RefObject<HTMLDivElement | null>
  followRef: { current: boolean }
  revealWork(key: string): void
}) {
  const tx = useAuthoredText()
  const [target, setTarget] = useState<SearchRevealRequest | null>(null)
  const [notice, setNotice] = useState('')
  const latest = useRef(input); latest.current = input
  useLayoutEffect(() => {
    setTarget(null); setNotice('')
    const consume = () => {
      const value = consumeConversationSearchTarget(latest.current.threadId)
      if (!value) return
      latest.current.followRef.current = false
      setTarget(value)
    }
    consume()
    window.addEventListener(REVEAL_CONVERSATION_MATCH, consume)
    return () => { window.removeEventListener(REVEAL_CONVERSATION_MATCH, consume) }
  }, [input.threadId])
  useLayoutEffect(() => {
    if (!target || target.threadId !== input.threadId) return
    if (!input.timeline.some(item => item.id === target.itemId)) { setNotice(tx('This message is no longer in the active conversation. Search again to find its current version.')); setTarget(null); return }
    input.followRef.current = false
    const row = input.rows.find(row => row.type === 'work' && row.turn.work.some(item => item.id === target.itemId))
    if (row) input.revealWork(row.turn.key)
    let frame = 0, attempts = 0
    const reveal = () => {
      const node = document.getElementById(`ti-${target.itemId}`)
      if (node && input.scrollRef.current?.contains(node)) {
        node.tabIndex = -1; node.focus({ preventScroll: true }); scrollTranscriptTarget(input.scrollRef.current, node)
        setNotice(target.origin === 'outline' ? tx('Outline item opened.') : target.origin === 'quote' ? tx('Quoted message opened.') : tx('Search match opened.')); return
      }
      if (++attempts < 40) frame = requestAnimationFrame(reveal)
      else setNotice(tx('This recorded item cannot be displayed here. Open its recorded activity or search again.'))
    }
    frame = requestAnimationFrame(reveal)
    return () => cancelAnimationFrame(frame)
  }, [target?.nonce, input.threadId])
  return { itemId: target?.threadId === input.threadId ? target.itemId : null, notice, clear: () => { setTarget(null); setNotice('') } }
}
