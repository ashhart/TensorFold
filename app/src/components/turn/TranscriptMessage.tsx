import { memo, useCallback } from 'react'
import { requestMessageQuote } from '../../lib/message-quotes'
import type { Thread, TimelineItem } from '../../lib/types'
import { SentUserMessage } from '../SentUserMessage'
import { AssistantMessage } from '../timeline'
import { TurnStatsLine } from './TurnStatsLine'

interface Props {
  threadId: string
  orchestratorHostId?: Thread['orchestratorHostId']
  item: TimelineItem
  busy: boolean
  active: boolean
  showStats: boolean
}

// Stable item and owner props keep old message bodies out of each new streaming render.
export const TranscriptMessage = memo(function TranscriptMessage({
  threadId, orchestratorHostId, item, busy, active, showStats,
}: Props) {
  const quote = useCallback((selection?: string) => {
    void requestMessageQuote({ threadId, itemId: item.id, ...(selection ? { selection } : {}) })
  }, [threadId, item.id])
  const onQuote = orchestratorHostId ? undefined : quote
  if (item.kind === 'user') {
    return <SentUserMessage thread={{ id: threadId, orchestratorHostId }} item={item} busy={busy} onQuote={onQuote} />
  }
  if (item.kind !== 'assistant') return null
  return <AssistantMessage text={item.text || ''} active={active} meta={item.meta} onQuote={onQuote}
    footer={showStats && item.stats ? <TurnStatsLine stats={item.stats} /> : undefined} />
})
