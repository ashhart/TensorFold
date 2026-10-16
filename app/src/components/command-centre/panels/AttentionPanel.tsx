import type { Dispatch, SetStateAction } from 'react'
import clsx from 'clsx'
import { relativeTime } from '../../../lib/format'
import { Button } from '../../ui'
import { DisclosurePanel, EmptyState, Panel, StatusIcon, ToneBadge } from '../ui'
import type { AttentionRow, AwayBriefRow } from '../model'

interface AttentionPanelProps {
  attentionRows: AttentionRow[]
  selected: AttentionRow | null
  setSelectedId: Dispatch<SetStateAction<string>>
  awayBriefRows: AwayBriefRow[]
  awayOpen: boolean
  toggleAwayOpen: () => void
}

export function AttentionPanel(props: AttentionPanelProps) {
  const { attentionRows, selected, setSelectedId, awayBriefRows, awayOpen, toggleAwayOpen } = props

  return (
    <>
      <Panel title="Attention Inbox" icon="bell" action={`${attentionRows.length} open`}>
        <div className="divide-y divide-line">
          {attentionRows.length === 0 ? <EmptyState icon="check" title="Inbox clear" detail="No connector, ticket, or automation signals need action." /> : attentionRows.map((row) => (
            <button
              key={row.id}
              onClick={() => { setSelectedId(row.id); row.onOpen?.() }}
              className={clsx('flex w-full items-start gap-3 px-3 py-2.5 text-left transition-colors hover:bg-card-2', selected?.id === row.id && 'bg-card-2')}
            >
              <StatusIcon icon={row.icon} tone={row.tone} />
              <span className="min-w-0 flex-1">
                <span className="flex items-center gap-2">
                  <span className="truncate text-[13px] font-medium text-ink-3">{row.source}</span>
                  <ToneBadge tone={row.tone} />
                  <span className="ml-auto shrink-0 text-[12px] text-ink-4">{row.age}</span>
                </span>
                <span className="mt-0.5 block truncate text-[13px] font-medium text-ink">{row.title}</span>
                <span className="mt-0.5 line-clamp-2 text-[13px] leading-snug text-ink-3">{row.detail}</span>
                {row.sourceIds?.length ? <span className="mt-1 block truncate font-mono text-[11px] text-ink-4">{row.sourceIds.join(' / ')}</span> : null}
              </span>
            </button>
          ))}
        </div>
      </Panel>

      <DisclosurePanel title="Since You Were Away" icon="clock" badge={awayBriefRows.length ? `24h / ${awayBriefRows.length}` : 'quiet'} isOpen={awayOpen} onToggle={toggleAwayOpen}>
        <div className="divide-y divide-line">
          {awayBriefRows.length === 0 ? (
            <EmptyState icon="check" title="No notable changes" detail="No failures, approvals, connector events, or active work changed in the last 24 hours." />
          ) : awayBriefRows.map((row) => (
            <div key={row.id} className="px-3 py-3">
              <div className="flex items-start gap-3">
                <StatusIcon icon={row.icon} tone={row.tone} />
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <div className="truncate text-[13px] font-semibold text-ink">{row.title}</div>
                    <span className="ml-auto shrink-0 text-[12px] text-ink-4">{relativeTime(row.createdAt)}</span>
                  </div>
                  <div className="mt-0.5 truncate text-[13px] text-ink-4">{row.meta}</div>
                  <div className="mt-1.5 line-clamp-2 text-[13px] leading-snug text-ink-2">{row.detail}</div>
                  {row.onOpen && (
                    <div className="mt-2 flex justify-end">
                      <Button variant={row.tone === 'critical' ? 'default' : 'ghost'} size="xs" icon="arrowRight" onClick={row.onOpen}>{row.actionLabel ?? 'Open'}</Button>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      </DisclosurePanel>
    </>
  )
}
