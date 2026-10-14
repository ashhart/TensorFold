import { useEffect, useState } from 'react'
import clsx from 'clsx'
import { api } from '../lib/ipc'
import { Icon } from '../lib/icons'
import { Button, Input, Toggle } from './ui'
import type { WebhookEndpoint, WebhookSnapshot, WebhookCapturePolicy, WebhookDirection, WebhookEventKind, WebhookTriggerPolicy } from '../lib/types'

const CAPTURE: WebhookCapturePolicy[] = ['metadata', 'redacted_payload', 'ttl_payload']
const TRIGGER: WebhookTriggerPolicy[] = ['notify', 'suggest', 'approval', 'run_automation']
const DIRECTION: WebhookDirection[] = ['inbound', 'outbound']
const EVENTS: WebhookEventKind[] = ['agent', 'room', 'ticket', 'automation', 'mission', 'connector']

function editableWebhook(endpoint: WebhookEndpoint): Partial<WebhookEndpoint> {
  return {
    id: endpoint.id,
    name: endpoint.name,
    direction: endpoint.direction,
    enabled: endpoint.enabled,
    scope: endpoint.scope,
    url: endpoint.url,
    path: endpoint.path,
    secretRef: endpoint.secretRef,
    replayWindowSeconds: endpoint.replayWindowSeconds,
    capturePolicy: endpoint.capturePolicy,
    triggerPolicy: endpoint.triggerPolicy,
    runAutomationId: endpoint.runAutomationId,
    eventKinds: endpoint.eventKinds,
    approvalRequired: endpoint.approvalRequired,
  }
}

// Webhooks (build.md Phase 2). Inbound is loopback-only; outbound is signed and history-backed.
// Secrets are masked here: entered or rotated, never read back.
export function WebhooksPanel() {
  const [snap, setSnap] = useState<WebhookSnapshot | null>(null)
  const [draft, setDraft] = useState<Partial<WebhookEndpoint> | null>(null)
  const [busy, setBusy] = useState(false)

  const reload = () => { api.getWebhooks().then(setSnap).catch(() => {}) }
  useEffect(reload, [])

  const run = async (fn: () => Promise<WebhookSnapshot>) => { setBusy(true); try { setSnap(await fn()) } finally { setBusy(false) } }
  const save = async () => { if (!draft) return; await run(() => api.saveWebhook(draft)); setDraft(null) }
  const genSecret = () => setDraft((d) => ({ ...d, secretRef: Array.from(crypto.getRandomValues(new Uint8Array(24))).map((b) => b.toString(16).padStart(2, '0')).join('') }))
  const toggleDraftEvent = (kind: WebhookEventKind) => setDraft((d) => {
    if (!d) return d
    const current = d.eventKinds ?? []
    const next = current.includes(kind) ? current.filter((item) => item !== kind) : [...current, kind]
    return { ...d, eventKinds: next }
  })

  const endpoints = snap?.endpoints ?? []
  const listener = snap?.listener
  const inboundCount = endpoints.filter((w) => w.direction === 'inbound').length
  const outboundCount = endpoints.filter((w) => w.direction === 'outbound').length

  return (
    <div className="mt-6 rounded-2xl border border-line bg-card-2/40 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-[13px] font-medium uppercase tracking-[0.05em] text-ink-3">Webhooks</h2>
          <p className="mt-1 text-[13px] text-ink-3">
            {listener?.listening
              ? <span className="text-positive">● Listening on 127.0.0.1:{listener.port} · {listener.routes} route{listener.routes === 1 ? '' : 's'}</span>
              : <span>{inboundCount} inbound · {outboundCount} outbound · enable an inbound endpoint to start the local listener.</span>}
          </p>
        </div>
        {!draft && <Button variant="default" size="sm" icon="plus" onClick={() => setDraft({ name: '', direction: 'inbound', path: '', url: '', replayWindowSeconds: 300, capturePolicy: 'redacted_payload', triggerPolicy: 'notify', eventKinds: [], approvalRequired: true, enabled: false })}>Add endpoint</Button>}
      </div>

      {draft && (
        <div className="mt-3 space-y-2.5 rounded-xl border border-accent/40 bg-card p-3">
          <div className="text-[13px] font-medium text-ink">{draft.id ? 'Edit endpoint' : `New ${draft.direction || 'inbound'} endpoint`}</div>
          <div className="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
            <Input value={draft.name || ''} onChange={(e) => setDraft({ ...draft, name: e.target.value })} placeholder="Name (e.g. GitHub CI)" />
            <label className="block">
              <div className="mb-1 text-[12px] text-ink-3">Direction</div>
              <select value={draft.direction || 'inbound'} onChange={(e) => {
                const direction = e.target.value as WebhookDirection
                setDraft(direction === 'outbound'
                  ? { ...draft, direction, path: undefined, runAutomationId: null, approvalRequired: true }
                  : { ...draft, direction, url: undefined, eventKinds: [], approvalRequired: false })
              }} className="w-full rounded-lg border border-line bg-card-2 px-2.5 py-2 text-[13px] text-ink outline-none">
                {DIRECTION.map((d) => <option key={d} value={d}>{d}</option>)}
              </select>
            </label>
            {(draft.direction || 'inbound') === 'outbound' ? (
              <Input value={draft.url || ''} onChange={(e) => setDraft({ ...draft, url: e.target.value })} placeholder="https://example.com/vontra" spellCheck={false} />
            ) : (
              <Input value={draft.path || ''} onChange={(e) => setDraft({ ...draft, path: e.target.value })} placeholder="/hooks/github" spellCheck={false} />
            )}
            <div className="flex gap-2">
              <Input type="password" value={draft.secretRef || ''} onChange={(e) => setDraft({ ...draft, secretRef: e.target.value })} placeholder="HMAC signing secret" spellCheck={false} />
              <Button variant="ghost" size="sm" onClick={genSecret} title="Generate a random secret">Generate</Button>
            </div>
            <Input type="number" value={String(draft.replayWindowSeconds ?? 300)} onChange={(e) => setDraft({ ...draft, replayWindowSeconds: parseInt(e.target.value, 10) || 300 })} placeholder={(draft.direction || 'inbound') === 'outbound' ? 'Retry window (s)' : 'Replay window (s)'} />
            <label className="block">
              <div className="mb-1 text-[12px] text-ink-3">Capture policy</div>
              <select value={draft.capturePolicy || 'redacted_payload'} onChange={(e) => setDraft({ ...draft, capturePolicy: e.target.value as WebhookCapturePolicy })} className="w-full rounded-lg border border-line bg-card-2 px-2.5 py-2 text-[13px] text-ink outline-none">
                {CAPTURE.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
            {(draft.direction || 'inbound') === 'inbound' && (
              <>
                <label className="block">
                  <div className="mb-1 text-[12px] text-ink-3">On delivery</div>
                  <select value={draft.triggerPolicy || 'notify'} onChange={(e) => setDraft({ ...draft, triggerPolicy: e.target.value as WebhookTriggerPolicy })} className="w-full rounded-lg border border-line bg-card-2 px-2.5 py-2 text-[13px] text-ink outline-none">
                    {TRIGGER.map((t) => <option key={t} value={t}>{t}</option>)}
                  </select>
                </label>
                {draft.triggerPolicy === 'run_automation' && (
                  <Input value={draft.runAutomationId || ''} onChange={(e) => setDraft({ ...draft, runAutomationId: e.target.value })} placeholder="Automation id to run" spellCheck={false} />
                )}
              </>
            )}
            {(draft.direction || 'inbound') === 'outbound' && (
              <>
                <label className="flex items-center justify-between gap-3 rounded-lg border border-line bg-card-2 px-3 py-2">
                  <span>
                    <span className="block text-[13px] font-medium text-ink-2">Require approval</span>
                    <span className="mt-0.5 block text-[12px] text-ink-3">New destinations stay blocked until approved.</span>
                  </span>
                  <Toggle label="Require approval" checked={draft.approvalRequired !== false} onChange={(approvalRequired) => setDraft({ ...draft, approvalRequired })} />
                </label>
                <div className="sm:col-span-2">
                  <div className="mb-1 text-[12px] text-ink-3">Outbound events</div>
                  <div className="flex flex-wrap gap-1.5">
                    {EVENTS.map((kind) => (
                      <button
                        type="button"
                        key={kind}
                        onClick={() => toggleDraftEvent(kind)}
                        className={clsx('rounded-lg border px-2 py-1 text-[13px] transition-colors', (draft.eventKinds ?? []).includes(kind) ? 'border-accent bg-accent text-accent-ink' : 'border-line bg-card-2 text-ink-3 hover:text-ink')}
                      >
                        {kind}
                      </button>
                    ))}
                  </div>
                  <div className="mt-1 text-[12px] text-ink-4">{(draft.eventKinds ?? []).length ? 'Only selected events deliver.' : 'No filter selected, all outbound events deliver after approval.'}</div>
                </div>
              </>
            )}
          </div>
          <p className="text-[12px] text-ink-4">
            {(draft.direction || 'inbound') === 'outbound'
              ? <>TensorFold signs compact event payloads with HMAC-SHA256 and records delivery history. Use https URLs, except localhost during development.</>
              : <>Senders sign <span className="font-mono">{'`${timestamp}.` + body'}</span> with HMAC-SHA256 (header <span className="font-mono">x-vontra-signature</span>, hex) and send <span className="font-mono">x-vontra-timestamp</span>. Payload values are never executed.</>}
          </p>
          <div className="flex justify-end gap-2">
            <Button variant="ghost" size="sm" onClick={() => setDraft(null)}>Cancel</Button>
            <Button variant="primary" size="sm" icon="check" onClick={save} disabled={busy}>Save</Button>
          </div>
        </div>
      )}

      <div className="mt-3 space-y-1.5">
        {endpoints.map((w) => {
          const approved = w.direction !== 'outbound' || w.approvalRequired === false || !!w.approvedAt
          const eventLabel = w.direction === 'outbound' ? (w.eventKinds?.length ? w.eventKinds.join(', ') : 'all events') : w.triggerPolicy
          return (
          <div key={w.id} data-testid={`webhook-${w.id}`} className="group rounded-xl border border-line bg-card px-3 py-2.5">
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 text-[13px] font-medium text-ink">
                  <Icon name={w.direction === 'outbound' ? 'send' : 'plug'} size={14} className="text-ink-3" />
                  {w.name}
                  <span className="rounded-md border border-line bg-card-2 px-1.5 py-0.5 text-[12px] uppercase tracking-[0.06em] text-ink-3">{w.direction}</span>
                  {w.direction === 'outbound' && <span className={clsx('rounded-md border px-1.5 py-0.5 text-[12px]', approved ? 'border-positive/30 bg-positive/10 text-positive' : 'border-warn/40 bg-warn/10 text-warn')}>{approved ? 'approved' : 'needs approval'}</span>}
                  <span className="truncate font-mono text-[12px] text-ink-3">{w.direction === 'outbound' ? w.url : w.path}</span>
                </div>
                <div className="mt-0.5 flex flex-wrap items-center gap-2 text-[12px] text-ink-4">
                  <span>{w.direction === 'outbound' ? eventLabel : w.triggerPolicy}</span>·<span>{w.capturePolicy}</span>·<span>{w.direction === 'outbound' ? 'retry 3x' : `replay ${w.replayWindowSeconds}s`}</span>{!w.secretRef && <span className="text-warn">· no secret set</span>}
                </div>
              </div>
              {w.direction === 'outbound' && !approved && (
                <button aria-label={`Approve ${w.name}`} onClick={() => run(() => api.approveWebhook(w.id))} className="text-[13px] text-warn transition-colors hover:text-ink" title="Approve outbound destination">Approve</button>
              )}
              <button aria-label={`Test ${w.name}`} onClick={() => run(() => api.testWebhook(w.id))} className="text-[13px] text-ink-3 transition-colors hover:text-ink" title="Send a sample event">Test</button>
              <Toggle label={`Enable ${w.name} webhook`} checked={w.enabled} onChange={(v) => run(() => api.saveWebhook({ id: w.id, enabled: v }))} />
              <button onClick={() => setDraft(editableWebhook(w))} className="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-ink-5 hover:text-ink"><Icon name="pencil" size={14} /></button>
              <button onClick={() => run(() => api.deleteWebhook(w.id))} className="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 opacity-0 transition-all hover:text-danger group-hover:opacity-100"><Icon name="trash" size={14} /></button>
            </div>
            {!!w.deliveries?.length && (
              <div className="mt-2 space-y-1 border-t border-line pt-2">
                {w.deliveries.slice(0, 3).map((d) => (
                  <div key={d.id} className="flex items-center gap-2 text-[12px] text-ink-4">
                    <span className={clsx('h-1.5 w-1.5 rounded-full', d.status === 'delivered' ? 'bg-positive' : d.status === 'failed' ? 'bg-danger' : 'bg-ink-4')} />
                    <span className="font-medium text-ink-3">{d.eventKind}</span>
                    <span>{d.status}</span>
                    <span>{d.httpStatus ?? d.error ?? 'pending'}</span>
                    <span>{new Date(d.createdAt).toLocaleTimeString()}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )})}
        {endpoints.length === 0 && !draft && (
          <div className={clsx('flex flex-col items-center justify-center rounded-xl border border-dashed border-line py-8 text-center')}>
            <span className="flex h-10 w-10 items-center justify-center rounded-xl bg-ink-5 text-ink-3"><Icon name="plug" size={20} /></span>
            <div className="mt-2.5 text-[13px] font-medium text-ink-2">No webhooks yet</div>
            <div className="mt-0.5 max-w-[360px] text-[13px] text-ink-3">Add a signed localhost route or an outbound destination for automation, room, mission, and connector events.</div>
          </div>
        )}
      </div>
    </div>
  )
}
