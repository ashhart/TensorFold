import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { Button } from '../ui'
import { useMailConnections } from './useMailConnections'

export function MailConnectionHealthPanel() {
  const tx = useAuthoredText(), health = useMailConnections()
  if (health.hostOnly) return <p className="text-[13px] text-ink-2">{tx('Manage mail connection health on the host computer.')}</p>
  return <section aria-label={tx('Mail connection health')} className="space-y-3 rounded-xl border border-line bg-card p-4">
    <div className="flex flex-wrap items-center justify-between gap-2"><h3 className="text-[14px] font-medium text-ink">{tx('Mail connection health')}</h3><Button size="sm" disabled={health.busy.length > 0} onClick={() => void health.refresh().catch(() => { })}>{tx('Refresh status')}</Button></div>
    <p className="text-[13px] text-ink-2">{tx('Authorization and mailbox access are separate. Test connection reads mailbox metadata only and never sends a message.')}</p>
    {health.error && <p role="alert" className="text-[13px] text-danger">{tx(health.error)}</p>}
    {!health.snapshot && !health.error && <p role="status" className="text-[13px] text-ink-3">{tx('Loading mail connection health…')}</p>}
    {health.snapshot?.accounts.length === 0 && <p className="text-[13px] text-ink-3">{tx('Connect a Google or Microsoft account to test mailbox access.')}</p>}
    {health.snapshot?.accounts.map(account => {
      const checking = health.busy.includes(`${account.provider}:${account.id}`)
      return <article key={`${account.provider}:${account.id}`} className="space-y-2 rounded-lg border border-line p-3">
        <div className="flex flex-wrap items-start justify-between gap-2"><div className="min-w-0"><p className="break-all text-[13px] font-medium text-ink">{account.email}</p><p className="text-[12px] text-ink-3">{account.provider === 'gmail' ? 'Gmail' : 'Outlook / Microsoft 365'}</p></div><Button size="sm" disabled={checking || account.authorization !== 'connected' || !account.read} onClick={() => void health.test({ provider: account.provider, id: account.id, revision: account.revision })}>{checking ? tx('Testing connection…') : tx('Test connection')}</Button></div>
        <p role="status" className={`text-[13px] ${account.state === 'healthy' ? 'text-ink-2' : 'text-ink-3'}`}>{tx(account.message)}</p>
        <p className="text-[12px] text-ink-3">{account.read ? tx('Mailbox reading granted') : tx('Mailbox reading not granted')} · {account.send ? tx('Reviewed sending granted') : tx('Sending not granted')}</p>
        <p className="text-[12px] text-ink-3">{tx('Last mailbox test')}: {account.checkedAt === null ? tx('Not tested') : new Date(account.checkedAt).toLocaleString()}</p>
        {account.lastSuccessAt !== null && <p className="text-[12px] text-ink-3">{tx('Last successful test read')}: {new Date(account.lastSuccessAt).toLocaleString()}</p>}
        {account.provider === 'gmail' && <Button size="sm" disabled={checking || !account.inbox && (account.authorization !== 'connected' || !account.read)} onClick={() => void health.setInbox(account.id, account.revision, !account.inbox)}>{account.inbox ? tx('Remove Gmail from Inbox') : tx('Add Gmail to Inbox')}</Button>}
      </article>
    })}
  </section>
}
