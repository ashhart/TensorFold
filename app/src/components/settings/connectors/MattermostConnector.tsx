import { useState } from 'react'
import clsx from 'clsx'
import { api } from '../../../lib/ipc'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Field, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { ConnectorStatusChip } from './ConnectorStatusChip'
import type { MattermostSource } from '../../../lib/types'

export function MattermostConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const { draft, setDraft, commit, connectorStatus } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  const [mattermostBusy, setMattermostBusy] = useState('')
  const [mattermostVerify, setMattermostVerify] = useState<{ ok: boolean; user?: string; error?: string } | null>(null)
  const addMattermostSource = async () => {
    const next = await api.addMattermostSource('Mattermost server')
    setDraft((d) => ({ ...d, mattermost: next }))
  }
  const removeMattermostSource = async (id: string) => {
    const next = await api.removeMattermostSource(id)
    setDraft((d) => ({ ...d, mattermost: next }))
  }
  const commitMattermostSourcePatch = (id: string, patch: Partial<MattermostSource>) => {
    const next = { ...draft, mattermost: { ...draft.mattermost, sources: draft.mattermost.sources.map((s) => s.id === id ? { ...s, ...patch, id: s.id, updatedAt: Date.now() } : s) } }
    setDraft(next)
    void commit(next)
  }
  const verifyMattermostSource = async (id: string) => {
    setMattermostBusy('verify-' + id)
    try { setMattermostVerify(await api.verifyMattermost(id).catch(() => ({ ok: false, error: 'Verification failed' }))) }
    finally { setMattermostBusy('') }
  }

  return (
    <Section title="Mattermost connector" desc="Live Mattermost via a Personal Access / bot token. Reads channels/threads and can reply (approval-gated). Create a token in your Mattermost account settings, paste it with the server URL.">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{draft.mattermost.sources.length} source{draft.mattermost.sources.length === 1 ? '' : 's'} configured</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.mattermost.lastSyncAt ? `Last sync ${new Date(draft.mattermost.lastSyncAt).toLocaleString()}` : 'Not synced yet'}</div>
            <ConnectorStatusChip status={connectorStatusMap.mattermost} />
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" icon="plus" onClick={() => void addMattermostSource()} disabled={!!mattermostBusy}>Add server</Button>
          </div>
        </div>
        {draft.mattermost.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            Add a Mattermost server URL + access token to read channels/threads and reply.
          </div>
        ) : draft.mattermost.sources.map((source) => (
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="mattermost" size={32} />
              <div className="min-w-0 flex-1">
                <Input value={source.label} onChange={(e) => setDraft((d) => ({ ...d, mattermost: { ...d.mattermost, sources: d.mattermost.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))} onBlur={() => commit(draft)} className="h-8" />
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
              </div>
              <Toggle label={`Enable ${source.label || "Mattermost source"}`} checked={source.enabled} onChange={(enabled) => commitMattermostSourcePatch(source.id, { enabled })} />
              <button onClick={() => void removeMattermostSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove Mattermost source"><Icon name="trash" size={15} /></button>
            </div>
            <div className="mt-3 space-y-2.5">
              <Field label="Server URL" hint="required"><Input value={source.baseUrl} onChange={(e) => setDraft((d) => ({ ...d, mattermost: { ...d.mattermost, sources: d.mattermost.sources.map((s) => s.id === source.id ? { ...s, baseUrl: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="https://mm.example.com" className="h-8" spellCheck={false} /></Field>
              <Field label="Private network" hint={source.allowPrivateNetwork ? 'Allowed' : 'Blocked'}>
                <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2">
                  <span className="flex-1 text-[13px] leading-snug text-ink-3">Allow an intentional self-hosted RFC1918 or loopback server. Cloud metadata and link-local addresses stay blocked.</span>
                  <Toggle label={`Allow private network for ${source.label || "Mattermost source"}`} checked={source.allowPrivateNetwork === true} onChange={(allowPrivateNetwork) => commitMattermostSourcePatch(source.id, { allowPrivateNetwork })} />
                </div>
              </Field>
              <Field label="Access token" hint="secret"><Input type="password" value={source.token} onChange={(e) => setDraft((d) => ({ ...d, mattermost: { ...d.mattermost, sources: d.mattermost.sources.map((s) => s.id === source.id ? { ...s, token: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="Personal Access or bot token" className="h-8" spellCheck={false} /></Field>
              <Field label="Allowed channels" hint="optional"><Input value={source.allowedChannels} onChange={(e) => setDraft((d) => ({ ...d, mattermost: { ...d.mattermost, sources: d.mattermost.sources.map((s) => s.id === source.id ? { ...s, allowedChannels: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="channel slug, channel id (blank = all)" className="h-8" spellCheck={false} /></Field>
              <Field label="Channel prompt" hint="optional"><Input value={source.channelPrompt ?? ''} onChange={(e) => setDraft((d) => ({ ...d, mattermost: { ...d.mattermost, sources: d.mattermost.sources.map((s) => s.id === source.id ? { ...s, channelPrompt: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="Extra guidance applied only to this Mattermost source (ephemeral)" className="h-8" spellCheck={false} /></Field>
              <div className="flex gap-2">
                <Button variant="default" size="sm" icon="shield" onClick={() => void verifyMattermostSource(source.id)} disabled={!!mattermostBusy}>{mattermostBusy === 'verify-' + source.id ? 'Verifying' : 'Verify token'}</Button>
                <Button variant="primary" size="sm" icon="message" onClick={async () => { setMattermostBusy('sync-' + source.id); try { await api.syncMattermost(source.id) } finally { setMattermostBusy('') } }} disabled={!!mattermostBusy}>{mattermostBusy === 'sync-' + source.id ? 'Syncing' : 'Sync Mattermost'}</Button>
              </div>
              {mattermostVerify && <div className={clsx('text-[13px]', mattermostVerify.ok ? 'text-positive' : 'text-danger')}>{mattermostVerify.ok ? `Connected as ${mattermostVerify.user}` : mattermostVerify.error}</div>}
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
