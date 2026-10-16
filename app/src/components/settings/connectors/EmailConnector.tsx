import { shortPath } from '../../../lib/format'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Field, Input, Segmented, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { ConnectorStatusChip } from './ConnectorStatusChip'

export function EmailConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const {
    draft,
    setDraft,
    commit,
    emailBusy,
    connectorStatus,
    addEmailSource,
    addImapAccount,
    syncEmail,
    patchImapAccount,
    removeEmailSource,
    updateEmailSource,
  } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  return (
    <Section title="Email connector" desc="Local-first inbox access. Add folders with .eml/.emlx files, a Maildir, or a live IMAP account. Outbound replies are approval-gated.">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{draft.email.sources.length} source{draft.email.sources.length === 1 ? '' : 's'} configured</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.email.lastSyncAt ? `Last sync ${new Date(draft.email.lastSyncAt).toLocaleString()}` : 'Not synced yet'}</div>
            <ConnectorStatusChip status={connectorStatusMap.email} />
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" icon="folder" onClick={addEmailSource} disabled={!!emailBusy}>{emailBusy === 'add' ? 'Adding' : 'Add folder'}</Button>
            <Button variant="default" size="sm" icon="mail" onClick={addImapAccount} disabled={!!emailBusy}>Add IMAP account</Button>
            <Button variant="primary" size="sm" icon="mail" onClick={syncEmail} disabled={!!emailBusy || draft.email.sources.length === 0}>{emailBusy === 'sync' ? 'Syncing' : 'Sync email'}</Button>
          </div>
        </div>

        {draft.email.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            Add a local email export folder or an IMAP account to turn the Email connector from a contract into a live source.
          </div>
        ) : draft.email.sources.map((source) => source.kind === 'imap' ? (
          // IMAP account: host/port/tls/user/password. Password is masked to the
          // renderer (••••) and secret-protected at rest; typing a new value replaces it.
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="outlook" size={32} />
              <div className="min-w-0 flex-1">
                <Input
                  value={source.label}
                  onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))}
                  onBlur={(e) => patchImapAccount(source.id, { label: e.target.value })}
                  className="h-8"
                />
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
              </div>
              <Toggle label={`Enable ${source.label || "email source"}`} checked={source.enabled} onChange={(enabled) => patchImapAccount(source.id, { enabled })} />
              <button onClick={() => removeEmailSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove IMAP account">
                <Icon name="trash" size={15} />
              </button>
            </div>
            <div className="mt-3 grid grid-cols-1 gap-2.5 sm:grid-cols-[1.4fr_0.7fr_0.7fr]">
              <Field label="IMAP host">
                <Input value={source.host ?? ''} onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, host: e.target.value } : s) } }))} onBlur={(e) => patchImapAccount(source.id, { host: e.target.value })} placeholder="outlook.office365.com" className="h-8" spellCheck={false} />
              </Field>
              <Field label="Port">
                <Input type="number" value={String(source.port ?? 993)} onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, port: Number(e.target.value) } : s) } }))} onBlur={(e) => patchImapAccount(source.id, { port: Number(e.target.value) })} className="h-8" />
              </Field>
              <Field label="TLS" hint={source.tls === false ? 'Off' : 'On'}>
                <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2">
                  <span className="flex-1 text-[13px] text-ink-3">Use TLS</span>
                  <Toggle label={`Use TLS for ${source.label || "IMAP account"}`} checked={source.tls !== false} onChange={(tls) => patchImapAccount(source.id, { tls })} />
                </div>
              </Field>
              <Field label="Username">
                <Input value={source.user ?? ''} onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, user: e.target.value } : s) } }))} onBlur={(e) => patchImapAccount(source.id, { user: e.target.value })} placeholder="you@example.com" className="h-8" spellCheck={false} />
              </Field>
              <Field label="Password" hint="secret">
                <Input type="password" value={source.password ?? ''} onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, password: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="app password" className="h-8" spellCheck={false} />
              </Field>
              <Field label="Mailbox">
                <Input value={source.mailbox ?? ''} onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, mailbox: e.target.value } : s) } }))} onBlur={(e) => patchImapAccount(source.id, { mailbox: e.target.value })} placeholder="INBOX" className="h-8" />
              </Field>
            </div>
            <div className="mt-2.5">
              <Field label="Channel prompt" hint="optional"><Input value={source.channelPrompt ?? ''} onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, channelPrompt: e.target.value } : s) } }))} onBlur={(e) => patchImapAccount(source.id, { channelPrompt: e.target.value })} placeholder="Extra guidance applied only to this inbox (ephemeral)" className="h-8" spellCheck={false} /></Field>
            </div>
            <div className="mt-2 text-right text-[12px] text-ink-4">{source.lastSyncAt ? `Synced ${new Date(source.lastSyncAt).toLocaleString()}` : 'Not synced'} · password is encrypted at rest</div>
          </div>
        ) : (
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="email" size={32} />
              <div className="min-w-0 flex-1">
                <Input
                  value={source.label}
                  onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))}
                  onBlur={(e) => updateEmailSource(source.id, { label: e.target.value })}
                  className="h-8"
                />
                <div className="mt-1 truncate text-[13px] text-ink-3" title={source.path}>{shortPath(source.path, 5)}</div>
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
              </div>
              <Toggle label={`Enable ${source.label || "email source"}`} checked={source.enabled} onChange={(enabled) => updateEmailSource(source.id, { enabled })} />
              <button onClick={() => removeEmailSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove email source">
                <Icon name="trash" size={15} />
              </button>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <Segmented
                value={source.kind}
                onChange={(kind) => updateEmailSource(source.id, { kind })}
                options={[{ value: 'eml_dir', label: 'EML folder', icon: 'folder' }, { value: 'maildir', label: 'Maildir', icon: 'mail' }]}
              />
              <span className="text-[12px] text-ink-3">{source.lastSyncAt ? `Synced ${new Date(source.lastSyncAt).toLocaleString()}` : 'Not scanned'}</span>
            </div>
            <div className="mt-2.5">
              <Field label="Channel prompt" hint="optional"><Input value={source.channelPrompt ?? ''} onChange={(e) => setDraft((d) => ({ ...d, email: { ...d.email, sources: d.email.sources.map((s) => s.id === source.id ? { ...s, channelPrompt: e.target.value } : s) } }))} onBlur={(e) => updateEmailSource(source.id, { channelPrompt: e.target.value })} placeholder="Extra guidance applied only to this inbox (ephemeral)" className="h-8" spellCheck={false} /></Field>
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
