import type { Attachment } from '../../lib/types'
import type { QuickAskWindowChoice } from '../../../shared/quick-ask'
import { AskDialog } from './AskDialog'
import { Icon } from '../../lib/icons'

export type AskText = (source: string, params?: Record<string, string | number | null | undefined>) => string
export interface AskContextGroup { id: string; image?: Attachment; text?: Attachment; label: string; ids: string[] }

export function askContextGroups(attachments: Attachment[]): AskContextGroup[] {
  const groups = new Map<string, AskContextGroup>()
  for (const item of attachments) {
    const id = /^(window-[a-f0-9-]{36})-(?:image|text)$/.exec(item.id)?.[1] ?? item.id
    const group = groups.get(id) ?? { id, label: item.name, ids: [] }
    group.ids.push(item.id)
    if (item.kind === 'image') { group.image = item; group.label = item.name }
    else group.text = item
    groups.set(id, group)
  }
  return [...groups.values()]
}

export function WindowTargetIcon() {
  return <svg width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.4" aria-hidden="true">
    <path d="M7 3H4a1 1 0 0 0-1 1v3m10-4h3a1 1 0 0 1 1 1v3M3 13v3a1 1 0 0 0 1 1h3m10-4v3a1 1 0 0 1-1 1h-3" />
    <rect x="6" y="6" width="8" height="8" rx="1" /><path d="M6 8h8" />
  </svg>
}

export function AskContextChips({ groups, blocked, preview, remove, tx }: { groups: AskContextGroup[]; blocked: boolean; preview(group: AskContextGroup): void; remove(ids: string[]): void; tx: AskText }) {
  return <div className="ask-contexts" aria-label={tx('Attached context')}>
    {groups.map(group => <div className="ask-context-chip" key={group.id}>
      <button onClick={() => preview(group)} className="ask-context-open" title={group.label}>
        {group.image ? <img src={group.image.dataUrl} alt="" /> : <Icon name="file" size={14} />}
        <span>{group.label}</span>
      </button>
      <button className="ask-icon-button" disabled={blocked} aria-label={tx('Remove {name}', { name: group.label })} onClick={() => remove(group.ids)}>×</button>
    </div>)}
  </div>
}

export function AskContextPreview({ group, close, tx }: { group: AskContextGroup; close(): void; tx: AskText }) {
  return <AskDialog title={group.label} onClose={close}>
    <p className="mb-3 text-xs text-ink-3">{tx('Only this selected context will be attached. Review it before sending.')}</p>
    {group.image && <img className="max-h-[250px] w-full rounded-lg object-contain" src={group.image.dataUrl} alt={group.label} />}
    {group.text && <details className="mt-3" open={!group.image}>
      <summary className="cursor-pointer text-xs text-ink-2">{tx('Visible text')}</summary>
      <pre className="mt-2 max-h-[200px] overflow-auto whitespace-pre-wrap break-words text-xs leading-relaxed">{group.text.text}</pre>
    </details>}
  </AskDialog>
}

export function AskWindowChoices({ windows, choose, close, blocked, tx }: { windows: QuickAskWindowChoice[]; choose(token: string): void; close(): void; blocked: boolean; tx: AskText }) {
  return <AskDialog title={tx('Choose a window')} onClose={close}>
    <p className="mb-4 text-xs text-ink-3">{tx('Choose one window to attach. Other windows are not sent.')}</p>
    <div className="grid grid-cols-2 gap-3">{windows.map(window => <button key={window.token} className="rounded-lg border border-line p-2 text-left hover:border-accent" disabled={blocked} onClick={() => choose(window.token)}>
      <img className="h-24 w-full rounded object-contain" src={window.thumbnail} alt="" />
      <span className="mt-2 block truncate text-xs">{window.title}</span>
    </button>)}</div>
  </AskDialog>
}
