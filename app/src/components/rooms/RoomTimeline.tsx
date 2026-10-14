import clsx from 'clsx'
import { basename } from '../../lib/format'
import { Icon } from '../../lib/icons'
import { FileGlyph } from '../../lib/fileicon'
import type { RoomEventItem } from '../../lib/types'
import { FILE_TOOL_VERB, Avatar } from './helpers'

const VERB_LABEL: Record<string, string> = { speak: 'says', ask: 'asks', assign: 'assigns', object: 'objects', claim: 'claims', check: 'checks', vote: 'votes', yield: 'yields', final: 'concludes' }

type RoomTimelineItem =
  | { kind: 'event'; event: RoomEventItem }
  | { kind: 'final_block_group'; event: RoomEventItem; count: number; targets: string[] }

function isFinalBlockEvent(event: RoomEventItem): boolean {
  return event.kind === 'check' && event.seat === 'moderator' && /^Final blocked:/i.test(event.text)
}

function finalBlockMessage(text: string): string {
  return text.replace(/^Final blocked:\s*/i, '').trim()
}

function compactRoomTimeline(events: RoomEventItem[]): RoomTimelineItem[] {
  const items: RoomTimelineItem[] = []
  for (const event of events) {
    if (isFinalBlockEvent(event)) {
      const previous = items[items.length - 1]
      if (previous?.kind === 'final_block_group' && previous.event.text === event.text) {
        previous.count += 1
        if (event.target && !previous.targets.includes(event.target)) previous.targets.push(event.target)
        continue
      }
      items.push({ kind: 'final_block_group', event, count: 1, targets: event.target && event.target !== 'all' ? [event.target] : [] })
      continue
    }
    items.push({ kind: 'event', event })
  }
  return items
}

export function RoomNoticeEvent({ event }: { event: RoomEventItem }) {
  return (
    <div className="rounded-xl border border-line bg-card-2 px-3.5 py-2.5">
      <div className="mb-1 flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wide text-ink-3">
        <Icon name="spark" size={12} />
        {event.seat === 'moderator' ? 'Moderator' : event.seat}
      </div>
      <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-ink-3">{event.text}</div>
    </div>
  )
}

export function RoomFinalBlockEvent({ event, count, targets }: { event: RoomEventItem; count: number; targets: string[] }) {
  return (
    <div className="rounded-xl border border-warn/30 bg-warn/10 px-3.5 py-3">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-warn/15 text-warn">
          <Icon name="warning" size={15} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-[13px] font-semibold text-ink">Final gate blocked</div>
            {count > 1 && <span className="rounded-md bg-warn/15 px-1.5 py-0.5 text-[12px] font-medium text-warn">{count} repeated checks</span>}
            {targets.length > 0 && <span className="rounded-md bg-ink-5 px-1.5 py-0.5 text-[12px] text-ink-3">to {targets.join(', ')}</span>}
          </div>
          <div className="mt-1 whitespace-pre-wrap text-[13px] leading-relaxed text-ink-2">{finalBlockMessage(event.text)}</div>
        </div>
      </div>
    </div>
  )
}

export function roomEventSnippet(text: string, max = 150): string {
  const compact = text.replace(/\s+/g, ' ').trim()
  return compact.length > max ? `${compact.slice(0, Math.max(0, max - 3))}...` : compact
}

export function RoomHandoffStrip({ events, live }: { events: RoomEventItem[]; live?: { seat: string; text: string } }) {
  const handoffs = [...events]
    .reverse()
    .filter((event) => ['ask', 'assign', 'claim'].includes(event.kind) && event.seat !== 'you')
    .slice(0, 3)

  if (!handoffs.length && !live) return null

  return (
    <div className="rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
      <div className="mb-2 flex items-center gap-2">
        <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-accent-soft text-accent">
          <Icon name="branch" size={14} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="text-[13px] font-semibold text-ink">Teamwork</div>
          <div className="truncate text-[12px] text-ink-4">{live ? `${live.seat} is shaping the next move` : 'Latest handoffs and claims'}</div>
        </div>
      </div>
      <div className="grid gap-1.5 sm:grid-cols-3">
        {handoffs.map((event) => {
          const isClaim = event.kind === 'claim'
          const target = event.target && event.target !== 'all' ? event.target : 'room'
          return (
            <div key={event.id} className="min-w-0 rounded-lg border border-line bg-card-2 px-2.5 py-2">
              <div className="mb-1 flex min-w-0 items-center gap-1.5 text-[12px] font-medium text-ink-3">
                <Avatar name={event.seat} size={16} />
                <span className="truncate text-ink-2">{event.seat}</span>
                {isClaim ? (
                  <span className="shrink-0 text-positive">claimed</span>
                ) : (
                  <>
                    <Icon name="arrowRight" size={10} className="shrink-0 text-ink-4" />
                    <span className="truncate">{target}</span>
                  </>
                )}
              </div>
              <div className="line-clamp-2 text-[13px] leading-snug text-ink-3">{roomEventSnippet(event.text, 110)}</div>
            </div>
          )
        })}
        {!handoffs.length && live && (
          <div className="min-w-0 rounded-lg border border-line bg-card-2 px-2.5 py-2 text-[13px] text-ink-3">{live.seat} is drafting a response.</div>
        )}
      </div>
    </div>
  )
}

export function RoomHandoffEvent({ event }: { event: RoomEventItem }) {
  const isAsk = event.kind === 'ask'
  const target = event.target && event.target !== 'all' ? event.target : 'room'
  return (
    <div className="rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
      <div className="flex items-start gap-3">
        <Avatar name={event.seat} />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5 text-[13px]">
            <span className="font-semibold text-ink">{event.seat}</span>
            <span className={clsx('flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[12px] font-medium', isAsk ? 'bg-accent-soft text-accent' : 'bg-ink-5 text-ink-2')}>
              <Icon name={isAsk ? 'message' : 'branch'} size={11} />
              {isAsk ? 'asks' : 'assigns'}
            </span>
            <Icon name="arrowRight" size={12} className="text-ink-4" />
            <span className="font-medium text-ink-2">{target}</span>
          </div>
          <div className="mt-1 whitespace-pre-wrap text-[14px] leading-relaxed text-ink-2">{event.text}</div>
        </div>
      </div>
    </div>
  )
}

export function RoomClaimEvent({ event }: { event: RoomEventItem }) {
  return (
    <div className="rounded-xl border border-line bg-card px-3.5 py-2.5">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-positive/12 text-positive">
          <Icon name="check" size={14} />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-1.5 text-[13px]">
            <span className="font-semibold text-ink">{event.seat}</span>
            <span className="text-ink-3">claimed work</span>
          </div>
          <div className="mt-0.5 whitespace-pre-wrap text-[13px] leading-relaxed text-ink-2">{event.text}</div>
        </div>
      </div>
    </div>
  )
}

export function RoomYieldEvent({ event }: { event: RoomEventItem }) {
  return (
    <div className="flex items-center gap-2 pl-1 text-[13px] text-ink-4">
      <Avatar name={event.seat} size={18} />
      <span className="font-medium text-ink-3">{event.seat}</span>
      <span>yielded</span>
      {event.text && <span className="min-w-0 truncate text-ink-4">{roomEventSnippet(event.text, 120)}</span>}
    </div>
  )
}

export function RoomTimeline({ events, live }: { events: RoomEventItem[]; live?: { seat: string; text: string } }) {
  const timelineItems = compactRoomTimeline(events)
  return (
    <>
      {timelineItems.map((item) => {
        if (item.kind === 'final_block_group') return <RoomFinalBlockEvent key={item.event.id} event={item.event} count={item.count} targets={item.targets} />
        const e = item.event
        if (e.kind === 'task') return (
          <div key={e.id} className="rounded-xl border border-line bg-card-2 px-4 py-3">
            <div className="mb-1 text-[12px] font-semibold uppercase tracking-wide text-ink-3">Task</div>
            <div className="whitespace-pre-wrap text-[14px] text-ink">{e.text}</div>
          </div>
        )
        if (e.kind === 'error') return (
          <div key={e.id} className="flex items-start gap-2 rounded-xl border border-danger/25 bg-danger/8 px-3.5 py-2.5 text-[13px] text-ink-2"><Icon name="warning" size={15} className="mt-0.5 shrink-0 text-danger" />{e.text}</div>
        )
        if (e.kind === 'notice') return <RoomNoticeEvent key={e.id} event={e} />
        if (e.kind === 'final') return (
          <div key={e.id} className="rounded-xl border border-positive/30 bg-positive/8 px-4 py-3">
            <div className="mb-1 flex items-center gap-1.5 text-[12px] font-semibold uppercase tracking-wide text-positive"><Icon name="check" size={13} />Final · {e.seat}</div>
            <div className="whitespace-pre-wrap text-[14px] leading-relaxed text-ink">{e.text}</div>
          </div>
        )
        if (e.kind === 'ask' || e.kind === 'assign') return <RoomHandoffEvent key={e.id} event={e} />
        if (e.kind === 'claim') return <RoomClaimEvent key={e.id} event={e} />
        if (e.kind === 'yield') return <RoomYieldEvent key={e.id} event={e} />
        if (e.kind === 'tool') {
          const verb = e.tool ? FILE_TOOL_VERB[e.tool] : undefined
          return (
            <div key={e.id} className="flex items-center gap-2 text-[13px] fade-in">
              <Avatar name={e.seat} size={20} />
              <span className="font-medium text-ink-2">{e.seat}</span>
              {verb && e.path ? (
                <span className="flex min-w-0 flex-1 items-center gap-1.5 text-ink-3">{verb}<FileGlyph path={e.path} size={13} /><span className="truncate text-ink-2">{basename(e.path)}</span></span>
              ) : (
                <span className="flex min-w-0 flex-1 items-center gap-1.5 truncate text-ink-3"><Icon name={e.tool === 'run_command' ? 'terminal' : 'grid'} size={12} />{e.text}</span>
              )}
            </div>
          )
        }
        return (
          <div key={e.id} className="flex gap-3">
            <Avatar name={e.seat} />
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-[13px]">
                <span className="font-semibold text-ink">{e.seat}</span>
                <span className="text-ink-3">{VERB_LABEL[e.kind] ?? e.kind}</span>
                {e.target && e.target !== 'all' && e.target !== e.seat && <span className="text-ink-3">→ {e.target}</span>}
                {e.replyTo && <span className="text-ink-4" title={`Replies to ${e.replyTo}`}>↩ re #{e.replyTo.slice(0, 8)}</span>}
              </div>
              <div className="mt-0.5 whitespace-pre-wrap text-[14px] leading-relaxed text-ink-2">{e.text}</div>
            </div>
          </div>
        )
      })}
      {live && (
        <div className="flex gap-3 fade-in">
          <Avatar name={live.seat} />
          <div className="min-w-0 flex-1">
            <div className="text-[13px] font-semibold text-ink">{live.seat}</div>
            <div className="mt-0.5 whitespace-pre-wrap text-[14px] leading-relaxed text-ink-2">{live.text}<span className="caret" /></div>
          </div>
        </div>
      )}
    </>
  )
}
