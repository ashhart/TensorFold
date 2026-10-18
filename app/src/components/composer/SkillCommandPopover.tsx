import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import clsx from 'clsx'

export interface SkillCommandPopoverProps {
  /** Where the popover should anchor relative to the input (home = below, thread = above). */
  anchor: 'home' | 'thread'
  /** Whether the trigger was `$` (skills) or `/` (commands). */
  kind: '/' | '$'
  /** The combined list of matching skills or commands. */
  pickItems: { name: string; description?: string; argHint?: string }[]
  /** The currently-highlighted index. */
  pIdx: number
  /** Update the highlighted index (hover/keyboard). */
  onHoverIndex: (index: number) => void
  /** Insert the chosen skill/command into the composer. */
  onSelect: (name: string) => void
}

export function SkillCommandPopover({
  anchor,
  kind,
  pickItems,
  pIdx,
  onHoverIndex,
  onSelect,
}: SkillCommandPopoverProps) {
  const tx = useAuthoredText()
  return (
    <div
      className={clsx(
        'absolute left-0 z-50 w-[min(440px,100%)] overflow-hidden rounded-xl border border-line bg-card p-1 shadow-pop fade-in',
        anchor === 'home' ? 'top-full mt-2' : 'bottom-full mb-2',
      )}
      style={{ boxShadow: 'var(--shadow-pop)' }}
    >
      <div className="px-2.5 pb-1 pt-1.5 text-[13px] font-medium text-ink-3">{kind === '$' ? tx("Skills") : tx("Commands")}</div>
      {pickItems.map((it, i) => (
        <button
          type="button"
          key={it.name}
          onMouseEnter={() => onHoverIndex(i)}
          onMouseDown={(e) => { e.preventDefault(); onSelect(it.name) }}
          className={clsx('flex w-full items-center gap-2 rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors', i === pIdx ? 'bg-ink-5 text-ink' : 'text-ink-2')}
        >
          <span className="shrink-0 font-mono text-ink-3">{kind}{it.name}</span>
          {it.argHint && <span className="shrink-0 font-mono text-[12px] text-ink-4">{it.argHint}</span>}
          {it.description && <span className="ml-auto truncate pl-3 text-[12px] text-ink-3">{it.description}</span>}
        </button>
      ))}
    </div>
  )
}
