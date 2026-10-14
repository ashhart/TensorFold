import { useState } from 'react'
import clsx from 'clsx'
import { api } from '../../../lib/ipc'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Field, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { ConnectorStatusChip } from './ConnectorStatusChip'
import type { DiscordSource } from '../../../lib/types'

export function DiscordConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const { draft, setDraft, commit, connectorStatus } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  const [discordBusy, setDiscordBusy] = useState('')
  const [discordVerify, setDiscordVerify] = useState<{ ok: boolean; bot?: string; guilds?: { id: string; name: string }[]; error?: string } | null>(null)
  const addDiscordSource = async () => {
    const next = await api.addDiscordSource('Discord server')
    setDraft((d) => ({ ...d, discord: next }))
  }
  const removeDiscordSource = async (id: string) => {
    const next = await api.removeDiscordSource(id)
    setDraft((d) => ({ ...d, discord: next }))
  }
  const commitDiscordSourcePatch = (id: string, patch: Partial<DiscordSource>) => {
    const next = { ...draft, discord: { ...draft.discord, sources: draft.discord.sources.map((s) => s.id === id ? { ...s, ...patch, id: s.id, updatedAt: Date.now() } : s) } }
    setDraft(next)
    void commit(next)
  }
  const verifyDiscordSource = async (id: string) => {
    setDiscordBusy('verify-' + id)
    try { setDiscordVerify(await api.verifyDiscord(id).catch(() => ({ ok: false, error: 'Verification failed' }))) }
    finally { setDiscordBusy('') }
  }

  return (
    <Section title="Discord connector" desc="Live Discord via bot token. Reads channels/threads and can reply (approval-gated). Create an app at discord.com/developers, invite the bot, then paste the token.">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{draft.discord.sources.length} source{draft.discord.sources.length === 1 ? '' : 's'} configured</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.discord.lastSyncAt ? `Last sync ${new Date(draft.discord.lastSyncAt).toLocaleString()}` : 'Not synced yet'}</div>
            <ConnectorStatusChip status={connectorStatusMap.discord} />
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" icon="plus" onClick={() => void addDiscordSource()} disabled={!!discordBusy}>Add bot</Button>
          </div>
        </div>
        {draft.discord.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            Add a Discord bot token to read channels/threads and reply.
          </div>
        ) : draft.discord.sources.map((source) => (
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="discord" size={32} />
              <div className="min-w-0 flex-1">
                <Input value={source.label} onChange={(e) => setDraft((d) => ({ ...d, discord: { ...d.discord, sources: d.discord.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))} onBlur={() => commit(draft)} className="h-8" />
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
              </div>
              <Toggle label={`Enable ${source.label || "Discord source"}`} checked={source.enabled} onChange={(enabled) => commitDiscordSourcePatch(source.id, { enabled })} />
              <button onClick={() => void removeDiscordSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove Discord source"><Icon name="trash" size={15} /></button>
            </div>
            <div className="mt-3 space-y-2.5">
              <Field label="Bot token" hint="secret"><Input type="password" value={source.token} onChange={(e) => setDraft((d) => ({ ...d, discord: { ...d.discord, sources: d.discord.sources.map((s) => s.id === source.id ? { ...s, token: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="bot token from the Developer Portal" className="h-8" spellCheck={false} /></Field>
              <Field label="Allowed channels" hint="optional"><Input value={source.allowedChannels} onChange={(e) => setDraft((d) => ({ ...d, discord: { ...d.discord, sources: d.discord.sources.map((s) => s.id === source.id ? { ...s, allowedChannels: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="#general, channel id (blank = all)" className="h-8" spellCheck={false} /></Field>
              <Field label="Channel prompt" hint="optional"><Input value={source.channelPrompt ?? ''} onChange={(e) => setDraft((d) => ({ ...d, discord: { ...d.discord, sources: d.discord.sources.map((s) => s.id === source.id ? { ...s, channelPrompt: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="Extra guidance applied only to this Discord source (ephemeral)" className="h-8" spellCheck={false} /></Field>
              <div className="flex gap-2">
                <Button variant="default" size="sm" icon="shield" onClick={() => void verifyDiscordSource(source.id)} disabled={!!discordBusy}>{discordBusy === 'verify-' + source.id ? 'Verifying' : 'Verify token'}</Button>
                <Button variant="primary" size="sm" icon="message" onClick={async () => { setDiscordBusy('sync-' + source.id); try { await api.syncDiscord(source.id) } finally { setDiscordBusy('') } }} disabled={!!discordBusy}>{discordBusy === 'sync-' + source.id ? 'Syncing' : 'Sync Discord'}</Button>
              </div>
              {discordVerify && <div className={clsx('text-[13px]', discordVerify.ok ? 'text-positive' : 'text-danger')}>{discordVerify.ok ? `Connected as ${discordVerify.bot}${discordVerify.guilds?.length ? ` (${discordVerify.guilds.length} servers)` : ''}` : discordVerify.error}</div>}
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
