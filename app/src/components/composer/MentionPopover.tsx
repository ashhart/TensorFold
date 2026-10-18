import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'
import { Icon } from '../../lib/icons'
import type { BuiltInCapability } from '../../lib/capabilities'
import { CapabilityGlyph } from './CapabilityGlyph'

export interface MentionPopoverProps {
  /** Where the popover should anchor relative to the input (home = below, thread = above). */
  anchor: 'home' | 'thread'
  /** The capabilities that match the current @-query. */
  capabilityItems: BuiltInCapability[]
  /** The files that match the current @-query. */
  items: string[]
  /** The currently-highlighted index across capabilities then files. */
  mIdx: number
  /** Update the highlighted index (hover/keyboard). */
  onHoverIndex: (index: number) => void
  /** Insert the chosen capability into the composer. */
  onSelectCapability: (capability: BuiltInCapability) => void
  /** Insert the chosen file mention into the composer. */
  onSelectMention: (file: string) => void
}

export function MentionPopover({
  anchor,
  capabilityItems,
  items,
  mIdx,
  onHoverIndex,
  onSelectCapability,
  onSelectMention,
}: MentionPopoverProps) {
  const tx = useAuthoredText()
  return (
    <div
      className={clsx(
        'absolute left-0 z-50 w-[min(440px,100%)] overflow-hidden rounded-xl border border-line bg-card p-1 shadow-pop fade-in',
        anchor === 'home' ? 'top-full mt-2' : 'bottom-full mb-2',
      )}
      style={{ boxShadow: 'var(--shadow-pop)' }}
    >
      {capabilityItems.length > 0 && (
        <>
          <div className="px-2.5 pb-1 pt-1.5 text-[13px] font-medium text-ink-3">{tx("Capabilities")}</div>
          {capabilityItems.map((capability, i) => (
            <button
              type="button"
              key={capability.id}
              onMouseEnter={() => onHoverIndex(i)}
              onMouseDown={(e) => { e.preventDefault(); onSelectCapability(capability) }}
              className={clsx('flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left transition-colors', i === mIdx ? 'bg-ink-5 text-ink' : 'text-ink-2')}
            >
              <CapabilityGlyph capability={capability} />
              <span className="shrink-0 text-[13px] font-medium">{capability.name}</span>
              <span className="truncate text-[13px] text-ink-3">{capability.description}</span>
            </button>
          ))}
        </>
      )}
      <div className={clsx('px-2.5 pb-1 pt-2 text-[13px] font-medium text-ink-3', capabilityItems.length > 0 && 'border-t border-line')}>{tx("Files")}</div>
      {items.map((f, i) => (
        <button
          type="button"
          key={f}
          onMouseEnter={() => onHoverIndex(capabilityItems.length + i)}
          onMouseDown={(e) => { e.preventDefault(); onSelectMention(f) }}
          className={clsx('flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors', capabilityItems.length + i === mIdx ? 'bg-ink-5 text-ink' : 'text-ink-2')}
        >
          <Icon name="file" size={14} className="shrink-0 text-ink-3" />
          <span className="truncate">{f.includes('/') ? <><span className="text-ink-3">{f.slice(0, f.lastIndexOf('/') + 1)}</span>{f.slice(f.lastIndexOf('/') + 1)}</> : f}</span>
        </button>
      ))}
      {items.length === 0 && <div className="px-2.5 py-1.5 text-[13px] text-ink-3">{tx("Type to search for files")}</div>}
    </div>
  )
}
