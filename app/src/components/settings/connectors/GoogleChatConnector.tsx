import { currentIntlLocale } from '../../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../../OriginalDiagnostic'
import { useAuthoredText } from '../../../lib/preferences/useAuthoredText'
import { useState } from 'react'
import clsx from 'clsx'
import { api } from '../../../lib/ipc'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Field, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { ConnectorStatusChip } from './ConnectorStatusChip'
import type { GoogleChatSource } from '../../../lib/types'

export function GoogleChatConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const { draft, setDraft, commit, connectorStatus } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  const [googlechatBusy, setGooglechatBusy] = useState('')
  const [googlechatVerify, setGooglechatVerify] = useState<{ ok: boolean; mode?: 'webhook' | 'token'; space?: string | null; error?: string } | null>(null)
  const addGoogleChatSource = async () => {
    const next = await api.addGoogleChatSource('Google Chat space')
    setDraft((d) => ({ ...d, googlechat: next }))
  }
  const removeGoogleChatSource = async (id: string) => {
    const next = await api.removeGoogleChatSource(id)
    setDraft((d) => ({ ...d, googlechat: next }))
  }
  const commitGoogleChatSourcePatch = (id: string, patch: Partial<GoogleChatSource>) => {
    const next = { ...draft, googlechat: { ...draft.googlechat, sources: draft.googlechat.sources.map((s) => s.id === id ? { ...s, ...patch, id: s.id, updatedAt: Date.now() } : s) } }
    setDraft(next)
    void commit(next)
  }
  const verifyGoogleChatSource = async (id: string) => {
    setGooglechatBusy('verify-' + id)
    try { setGooglechatVerify(await api.verifyGoogleChat(id).catch(() => ({ ok: false, error: 'Verification failed' }))) }
    finally { setGooglechatBusy('') }
  }

  return (
    <Section title={tx("Google Chat connector")} desc={tx("Google Chat spaces via an incoming-webhook URL (send) and an optional service-account / OAuth2 token (read). Paste a space's webhook URL from Apps & integrations; reads need a bearer token.")}>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{tx("Configured sources: {count}", { count: draft.googlechat.sources.length })}</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.googlechat.lastSyncAt ? tx("Last sync {value0}", { value0: new Date(draft.googlechat.lastSyncAt).toLocaleString(currentIntlLocale()) }) : tx("Not synced yet")}</div>
            <ConnectorStatusChip status={connectorStatusMap.googlechat} />
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" icon="plus" onClick={() => void addGoogleChatSource()} disabled={!!googlechatBusy}>{tx("Add space")}</Button>
          </div>
        </div>
        {draft.googlechat.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            {tx("Add a Google Chat incoming-webhook URL to send (and a service-account token to read).")}</div>
        ) : draft.googlechat.sources.map((source) => (
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="googlechat" size={32} />
              <div className="min-w-0 flex-1">
                <Input value={source.label} onChange={(e) => setDraft((d) => ({ ...d, googlechat: { ...d.googlechat, sources: d.googlechat.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))} onBlur={() => commit(draft)} className="h-8" />
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{<OriginalDiagnostic>{source.lastError}</OriginalDiagnostic>}</div>}
              </div>
              <Toggle label={tx("Enable {value0}", { value0: source.label || tx("Google Chat source") })} checked={source.enabled} onChange={(enabled) => commitGoogleChatSourcePatch(source.id, { enabled })} />
              <button onClick={() => void removeGoogleChatSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label={tx("Remove Google Chat source")}><Icon name="trash" size={15} /></button>
            </div>
            <div className="mt-3 space-y-2.5">
              <Field label={tx("Webhook URL")} hint={tx("secret")}><Input type="password" value={source.webhookUrl} onChange={(e) => setDraft((d) => ({ ...d, googlechat: { ...d.googlechat, sources: d.googlechat.sources.map((s) => s.id === source.id ? { ...s, webhookUrl: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="https://chat.googleapis.com/v1/spaces/…/messages?key=…&token=…" className="h-8" spellCheck={false} /></Field>
              <Field label={tx("Access token")} hint={tx("optional, secret — enables reading")}><Input type="password" value={source.accessToken ?? ''} onChange={(e) => setDraft((d) => ({ ...d, googlechat: { ...d.googlechat, sources: d.googlechat.sources.map((s) => s.id === source.id ? { ...s, accessToken: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder={tx("service-account / OAuth2 bearer token")} className="h-8" spellCheck={false} /></Field>
              <Field label={tx("Allowed spaces")} hint={tx("optional")}><Input value={source.allowedChannels} onChange={(e) => setDraft((d) => ({ ...d, googlechat: { ...d.googlechat, sources: d.googlechat.sources.map((s) => s.id === source.id ? { ...s, allowedChannels: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder={tx("spaces/AAAA, spaces/BBBB (blank = all)")} className="h-8" spellCheck={false} /></Field>
              <Field label={tx("Channel prompt")} hint={tx("optional")}><Input value={source.channelPrompt ?? ''} onChange={(e) => setDraft((d) => ({ ...d, googlechat: { ...d.googlechat, sources: d.googlechat.sources.map((s) => s.id === source.id ? { ...s, channelPrompt: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder={tx("Extra guidance applied only to this Google Chat source (ephemeral)")} className="h-8" spellCheck={false} /></Field>
              <div className="flex gap-2">
                <Button variant="default" size="sm" icon="shield" onClick={() => void verifyGoogleChatSource(source.id)} disabled={!!googlechatBusy}>{googlechatBusy === 'verify-' + source.id ? tx("Verifying") : tx("Verify")}</Button>
                <Button variant="primary" size="sm" icon="message" onClick={async () => { setGooglechatBusy('sync-' + source.id); try { await api.syncGoogleChat(source.id) } finally { setGooglechatBusy('') } }} disabled={!!googlechatBusy}>{googlechatBusy === 'sync-' + source.id ? tx("Syncing") : tx("Sync Google Chat")}</Button>
              </div>
              {googlechatVerify && <div className={clsx('text-[13px]', googlechatVerify.ok ? 'text-positive' : 'text-danger')}>{googlechatVerify.ok ? tx("Verified ({value0}{value1})", { value0: googlechatVerify.mode, value1: googlechatVerify.space ? ` · ${googlechatVerify.space}` : '' }) : googlechatVerify.error}</div>}
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
