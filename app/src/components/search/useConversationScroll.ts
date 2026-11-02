import { useCallback, useLayoutEffect, useState, type RefObject } from 'react'
import { captureTranscriptPosition, restoreTranscriptPosition, transcriptAtEnd } from '../../lib/transcript-scroll'
import { transcriptViewMemory } from '../../lib/transcript-view-memory'

export function useConversationScroll(input: {
  threadId: string
  revision: string
  scrollRef: RefObject<HTMLDivElement | null>
  followRef: { current: boolean }
}) {
  const { threadId, revision, scrollRef, followRef } = input
  const [following, setFollowing] = useState(true)
  const remember = useCallback(() => {
    const element = scrollRef.current
    if (element?.dataset.conversationId === threadId) {
      transcriptViewMemory.set(threadId, { position: captureTranscriptPosition(element, followRef.current) })
    }
  }, [threadId, scrollRef, followRef])

  useLayoutEffect(() => {
    const element = scrollRef.current
    if (!element) return
    const saved = transcriptViewMemory.get(threadId)?.position
    followRef.current = saved?.following ?? true
    setFollowing(followRef.current)
    restoreTranscriptPosition(element, saved)
    remember()
    return remember
  }, [threadId, scrollRef, followRef, remember])

  useLayoutEffect(() => {
    const element = scrollRef.current
    if (element && followRef.current) {
      element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight)
      remember()
    }
  }, [threadId, revision, scrollRef, followRef, remember])

  useLayoutEffect(() => {
    const element = scrollRef.current
    if (!element || typeof ResizeObserver === 'undefined') return
    const observer = new ResizeObserver(() => {
      if (followRef.current) {
        element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight)
        remember()
      }
    })
    observer.observe(element)
    if (element.firstElementChild) observer.observe(element.firstElementChild)
    return () => observer.disconnect()
  }, [threadId, scrollRef, followRef, remember])

  const onScroll = () => {
    const element = scrollRef.current
    if (!element) return
    followRef.current = transcriptAtEnd(element)
    setFollowing(followRef.current)
    remember()
  }
  const jumpToLatest = () => {
    const element = scrollRef.current
    if (!element) return
    followRef.current = true
    setFollowing(true)
    element.scrollTop = Math.max(0, element.scrollHeight - element.clientHeight)
    remember()
  }
  return { onScroll, following, jumpToLatest }
}
