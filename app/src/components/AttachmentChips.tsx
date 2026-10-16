import { Icon } from '../lib/icons'
import type { Attachment } from '../lib/types'

// Compact attachment chips — used both in the composers (with a remove button) and
// on sent user bubbles / room tasks (read-only, no onRemove).
export function AttachmentChips({ items, onRemove, className = '' }: { items?: Attachment[]; onRemove?: (id: string) => void; className?: string }) {
  if (!items || !items.length) return null
  return (
    <div className={`flex flex-wrap gap-1.5 ${className}`}>
      {items.map((a) => (
        <div key={a.id} className="group flex items-center gap-1.5 rounded-lg border border-line bg-card-2 py-1 pl-1 pr-2 text-[13px] text-ink-2">
          {a.kind === 'image' && a.dataUrl
            ? <img src={a.dataUrl} alt={a.name} className="h-6 w-6 rounded object-cover" />
            : <span className="flex h-6 w-6 items-center justify-center rounded bg-ink-5 text-ink-3"><Icon name="fileText" size={13} /></span>}
          <span className="max-w-[160px] truncate" title={a.name}>{a.name}</span>
          {onRemove && (
            <button type="button" onClick={() => onRemove(a.id)} className="text-ink-4 transition-colors hover:text-danger" aria-label={`Remove ${a.name}`}>
              <Icon name="x" size={12} />
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
