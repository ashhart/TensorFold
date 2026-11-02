import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Button } from '../ui'
import type { ReturnTypeOfGmailInbox } from './gmail-inbox-view'

export function GmailInboxStatus({ gmail, selectedKey }: { gmail: ReturnTypeOfGmailInbox; selectedKey?: string }) {
  const tx = useAuthoredText(), selected = selectedKey?.startsWith('gmail:')
  const enabled = gmail.snapshot.accounts.filter(account => account.enabled)
  if (!enabled.length && !gmail.error) return null
  return <section aria-label={tx('Gmail in Inbox')} className="space-y-2 border-b border-line bg-card px-4 py-3 text-[13px] sm:px-6">
    <div className="flex flex-wrap items-center justify-between gap-2"><p className="text-ink-2">{tx('Gmail in Inbox')} · {tx('Up to 20 recent Inbox messages per selected account')}</p><div className="flex flex-wrap gap-2"><Button size="sm" disabled={!!gmail.busy} onClick={() => void gmail.sync()}>{gmail.busy === 'sync' ? tx('Syncing Gmail…') : tx('Sync Gmail')}</Button>{selected && <Button size="sm" disabled={!!gmail.busy} onClick={() => void gmail.read(selectedKey!)}>{gmail.busy === 'read' ? tx('Reading…') : tx('Read Gmail conversation')}</Button>}</div></div>
    {enabled.map(account => <p key={account.id} className="break-words text-[12px] text-ink-3">{account.email} · {!account.connected ? tx('Reconnect needed') : account.syncedAt === null ? tx('Not synced') : `${tx('Last sync')}: ${new Date(account.syncedAt).toLocaleString()}`}{account.error && <span className="ml-2 text-warn">{tx(account.error)}</span>}</p>)}
    {gmail.error && <p role="alert" className="text-danger">{tx(gmail.error)}</p>}
  </section>
}
