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
import type { MatrixSource } from '../../../lib/types'

export function MatrixConnector({ ctx }: { ctx: SettingsSectionContext }) {
  const tx = useAuthoredText()
  const { draft, setDraft, commit, connectorStatus } = ctx
  const connectorStatusMap = connectorStatus ?? {}

  const [matrixBusy, setMatrixBusy] = useState('')
  const [matrixVerify, setMatrixVerify] = useState<{ ok: boolean; userId?: string; error?: string } | null>(null)
  const addMatrixSource = async () => {
    const next = await api.addMatrixSource('Matrix account')
    setDraft((d) => ({ ...d, matrix: next }))
  }
  const removeMatrixSource = async (id: string) => {
    const next = await api.removeMatrixSource(id)
    setDraft((d) => ({ ...d, matrix: next }))
  }
  const commitMatrixSourcePatch = (id: string, patch: Partial<MatrixSource>) => {
    const next = { ...draft, matrix: { ...draft.matrix, sources: draft.matrix.sources.map((s) => s.id === id ? { ...s, ...patch, id: s.id, updatedAt: Date.now() } : s) } }
    setDraft(next)
    void commit(next)
  }
  const verifyMatrixSource = async (id: string) => {
    setMatrixBusy('verify-' + id)
    try { setMatrixVerify(await api.verifyMatrix(id).catch(() => ({ ok: false, error: 'Verification failed' }))) }
    finally { setMatrixBusy('') }
  }

  return (
    <Section title={tx("Matrix connector")} desc={tx("Live Matrix via a homeserver access token. Reads rooms/threads and can reply (approval-gated). Paste your homeserver URL and an access token (Settings → Help & About in Element).")}>
      <div className="space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card px-3.5 py-3 shadow-card">
          <div>
            <div className="text-[13px] font-medium text-ink">{tx("Configured sources: {count}", { count: draft.matrix.sources.length })}</div>
            <div className="mt-0.5 text-[13px] text-ink-3">{draft.matrix.lastSyncAt ? tx("Last sync {value0}", { value0: new Date(draft.matrix.lastSyncAt).toLocaleString(currentIntlLocale()) }) : tx("Not synced yet")}</div>
            <ConnectorStatusChip status={connectorStatusMap.matrix} />
          </div>
          <div className="flex gap-2">
            <Button variant="default" size="sm" icon="plus" onClick={() => void addMatrixSource()} disabled={!!matrixBusy}>{tx("Add account")}</Button>
          </div>
        </div>
        {draft.matrix.sources.length === 0 ? (
          <div className="rounded-xl border border-dashed border-line px-3.5 py-5 text-center text-[13px] text-ink-3">
            {tx("Add a Matrix homeserver URL + access token to read rooms/threads and reply.")}</div>
        ) : draft.matrix.sources.map((source) => (
          <div key={source.id} className="rounded-xl border border-line bg-card p-3.5 shadow-card">
            <div className="flex items-start gap-3">
              <BrandIcon brand="matrix" size={32} />
              <div className="min-w-0 flex-1">
                <Input value={source.label} onChange={(e) => setDraft((d) => ({ ...d, matrix: { ...d.matrix, sources: d.matrix.sources.map((s) => s.id === source.id ? { ...s, label: e.target.value } : s) } }))} onBlur={() => commit(draft)} className="h-8" />
                {source.lastError && <div className="mt-1 text-[13px] text-danger">{<OriginalDiagnostic>{source.lastError}</OriginalDiagnostic>}</div>}
              </div>
              <Toggle label={tx("Enable {value0}", { value0: source.label || tx("Matrix source") })} checked={source.enabled} onChange={(enabled) => commitMatrixSourcePatch(source.id, { enabled })} />
              <button onClick={() => void removeMatrixSource(source.id)} className="flex h-8 w-8 items-center justify-center rounded-lg text-ink-3 transition-colors hover:bg-danger/10 hover:text-danger" aria-label={tx("Remove Matrix source")}><Icon name="trash" size={15} /></button>
            </div>
            <div className="mt-3 space-y-2.5">
              <Field label={tx("Homeserver URL")} hint={tx("required")}><Input value={source.baseUrl} onChange={(e) => setDraft((d) => ({ ...d, matrix: { ...d.matrix, sources: d.matrix.sources.map((s) => s.id === source.id ? { ...s, baseUrl: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder="https://matrix.org" className="h-8" spellCheck={false} /></Field>
              <Field label={tx("Private network")} hint={source.allowPrivateNetwork ? tx("Allowed") : tx("Blocked")}>
                <div className="flex items-center gap-3 rounded-lg border border-line bg-card-2 px-3 py-2">
                  <span className="flex-1 text-[13px] leading-snug text-ink-3">{tx("Allow an intentional self-hosted RFC1918 or loopback homeserver. Cloud metadata and link-local addresses stay blocked.")}</span>
                  <Toggle label={tx("Allow private network for {value0}", { value0: source.label || tx("Matrix source") })} checked={source.allowPrivateNetwork === true} onChange={(allowPrivateNetwork) => commitMatrixSourcePatch(source.id, { allowPrivateNetwork })} />
                </div>
              </Field>
              <Field label={tx("Access token")} hint={tx("secret")}><Input type="password" value={source.token} onChange={(e) => setDraft((d) => ({ ...d, matrix: { ...d.matrix, sources: d.matrix.sources.map((s) => s.id === source.id ? { ...s, token: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder={tx("homeserver access token")} className="h-8" spellCheck={false} /></Field>
              <Field label={tx("Allowed rooms")} hint={tx("optional")}><Input value={source.allowedRooms} onChange={(e) => setDraft((d) => ({ ...d, matrix: { ...d.matrix, sources: d.matrix.sources.map((s) => s.id === source.id ? { ...s, allowedRooms: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder={tx("!room:server, #alias:server (blank = all joined)")} className="h-8" spellCheck={false} /></Field>
              <Field label={tx("Channel prompt")} hint={tx("optional")}><Input value={source.channelPrompt ?? ''} onChange={(e) => setDraft((d) => ({ ...d, matrix: { ...d.matrix, sources: d.matrix.sources.map((s) => s.id === source.id ? { ...s, channelPrompt: e.target.value } : s) } }))} onBlur={() => commit(draft)} placeholder={tx("Extra guidance applied only to this Matrix source (ephemeral)")} className="h-8" spellCheck={false} /></Field>
              <div className="flex gap-2">
                <Button variant="default" size="sm" icon="shield" onClick={() => void verifyMatrixSource(source.id)} disabled={!!matrixBusy}>{matrixBusy === 'verify-' + source.id ? tx("Verifying") : tx("Verify token")}</Button>
                <Button variant="primary" size="sm" icon="message" onClick={async () => { setMatrixBusy('sync-' + source.id); try { await api.syncMatrix(source.id) } finally { setMatrixBusy('') } }} disabled={!!matrixBusy}>{matrixBusy === 'sync-' + source.id ? tx("Syncing") : tx("Sync Matrix")}</Button>
              </div>
              {matrixVerify && <div className={clsx('text-[13px]', matrixVerify.ok ? 'text-positive' : 'text-danger')}>{matrixVerify.ok ? tx("Connected as {value0}", { value0: matrixVerify.userId }) : matrixVerify.error}</div>}
            </div>
          </div>
        ))}
      </div>
    </Section>
  )
}
