import { shortPath } from '../../../lib/format'
import { BrandIcon } from '../../../lib/brand'
import { Icon } from '../../../lib/icons'
import { Button, Field, Input, Toggle } from '../../ui'
import { Section, type SettingsSectionContext } from '../shared'
import { ConnectorStatusChip } from './ConnectorStatusChip'

export function SlackConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const {
    draft,
    setDraft,
    commit,
    slackBusy,
    connectorStatus,
    addSlackSource,
    syncSlack,
    updateSlackSource,
    removeSlackSource,
  } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  // Slack live + Discord helpers: defined locally (api + commit in scope) so the
  // big Settings.tsx ctx doesn't have to carry every connector action.
  const addLiveSlackSource = () => {
    const id = `slack_${Math.random().toString(36).slice(2, 10)}`
    const source = { id, label: 'Slack workspace', path: '', kind: 'api_token' as const, token: '', channels: '', enabled: true, createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null }
    setDraft((d) => ({ ...d, slack: { ...d.slack, sources: [source, ...d.slack.sources] } }))
  }

  return (
    <Section title="Slack connector" desc="Local Slack exports or a live workspace (bot token). A live token reads channels/threads and can reply, approval-gated.">
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{draft.slack.sources.length} source{draft.slack.sources.length === 1 ? '' : 's'} configured</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.slack.lastSyncAt ? `Last sync ${new Date(draft.slack.lastSyncAt).toLocaleString()}` : 'Not synced yet'}</div>
            <ConnectorStatusChip status={connectorStatusMap.slack} />
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" icon="folder" onClick={addSlackSource} disabled={!!slackBusy}>{slackBusy === 'add' ? 'Adding' : 'Add export'}</Button>
            <Button variant="default" size="sm" icon="plus" onClick={() => addLiveSlackSource()} disabled={!!slackBusy}>Add live token</Button>
            <Button variant="primary" size="sm" icon="message" onClick={syncSlack} disabled={!!slackBusy || draft.slack.sources.length === 0}>{slackBusy === 'sync' ? 'Syncing' : 'Sync Slack'}</Button>
          </div>
        </div>

        {draft.slack.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            Add a local Slack export folder, or a live bot token (api.slack.com app) to read channels and reply.
          </div>
        ) : draft.slack.sources.map((source) => source.kind === 'api_token' ? (
          // Live Slack workspace token: xoxb-... + allowed channels.
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="slack" size={32} />
              <div className="min-w-0 flex-1">
                <Input value={source.label} onChange={(e) => setDraft((d) => ({ ...d, slack: { ...d.slack, sources: d.slack.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))} onBlur={(e) => updateSlackSource(source.id, { label: e.target.value })} className="h-8" />
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
              </div>
              <Toggle label={`Enable ${source.label || "Slack source"}`} checked={source.enabled} onChange={(enabled) => updateSlackSource(source.id, { enabled })} />
              <button onClick={() => removeSlackSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove Slack source"><Icon name="trash" size={15} /></button>
            </div>
            <div className="mt-3 space-y-2.5">
              <Field label="Bot token" hint="secret"><Input type="password" value={source.token ?? ''} onChange={(e) => setDraft((d) => ({ ...d, slack: { ...d.slack, sources: d.slack.sources.map((s) => s.id === source.id ? { ...s, token: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="xoxb-..." className="h-8" spellCheck={false} /></Field>
              <Field label="Allowed channels" hint="optional"><Input value={source.channels ?? ''} onChange={(e) => setDraft((d) => ({ ...d, slack: { ...d.slack, sources: d.slack.sources.map((s) => s.id === source.id ? { ...s, channels: e.target.value } : s) } }))} onBlur={(e) => updateSlackSource(source.id, { channels: e.target.value })} placeholder="#general, C0... (blank = all the bot sees)" className="h-8" spellCheck={false} /></Field>
              <Field label="Channel prompt" hint="optional"><Input value={source.channelPrompt ?? ''} onChange={(e) => setDraft((d) => ({ ...d, slack: { ...d.slack, sources: d.slack.sources.map((s) => s.id === source.id ? { ...s, channelPrompt: e.target.value } : s) } }))} onBlur={(e) => updateSlackSource(source.id, { channelPrompt: e.target.value })} placeholder="Extra guidance applied only to this Slack source (ephemeral)" className="h-8" spellCheck={false} /></Field>
            </div>
          </div>
        ) : (
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="slack" size={32} />
              <div className="min-w-0 flex-1">
                <Input
                  value={source.label}
                  onChange={(e) => setDraft((d) => ({ ...d, slack: { ...d.slack, sources: d.slack.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))}
                  onBlur={(e) => updateSlackSource(source.id, { label: e.target.value })}
                  className="h-8"
                />
                <div className="mt-1 truncate text-[13px] text-ink-3" title={source.path}>{shortPath(source.path, 5)}</div>
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{source.lastError}</div>}
              </div>
              <Toggle label={`Enable ${source.label || "Slack source"}`} checked={source.enabled} onChange={(enabled) => updateSlackSource(source.id, { enabled })} />
              <button onClick={() => removeSlackSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label="Remove Slack source">
                <Icon name="trash" size={15} />
              </button>
            </div>
            <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
              <span className="rounded-lg border border-line bg-card-2 px-2.5 py-1.5 text-[13px] text-ink-3">Slack export folder</span>
              <span className="text-[12px] text-ink-3">{source.lastSyncAt ? `Synced ${new Date(source.lastSyncAt).toLocaleString()}` : 'Not scanned'}</span>
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
