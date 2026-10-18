import { currentIntlLocale } from '../../lib/preferences/current-locale'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { useEffect, useRef, useState } from 'react'
import type { WorkspaceBackupApi, WorkspaceBackupInventory, WorkspaceBackupPreview, WorkspaceBackupSelection, WorkspaceRestoreMode } from '../../../shared/workspace-backup'
import { api } from '../../lib/ipc'
import { Button, Input } from '../ui'

const describeCounts = (preview: WorkspaceBackupPreview) => `${preview.counts.conversations} conversations, ${preview.counts.projects} projects, ${preview.counts.bots} bots, ${preview.counts.drafts} unsent drafts and ${preview.counts.attachments} attachments`

/** Native file selection, encryption and restore authority all remain in main. */
export function WorkspaceBackupPanel({ client = api, disabled = false }: { client?: WorkspaceBackupApi; disabled?: boolean }) {
  const tx = useAuthoredText()
  const [inventory, setInventory] = useState<WorkspaceBackupInventory | null>(null)
  const [mode, setMode] = useState<'export' | 'restore'>('export')
  const [restoreMode, setRestoreMode] = useState<WorkspaceRestoreMode>('merge')
  const [selection, setSelection] = useState<WorkspaceBackupSelection>({ settings: true, drafts: true, attachments: true, conversationIds: [], projectIds: [], botIds: [] })
  const [preview, setPreview] = useState<WorkspaceBackupPreview | null>(null)
  const [password, setPassword] = useState('')
  const [confirmed, setConfirmed] = useState(false)
  const [busy, setBusy] = useState(false)
  const [cancelling, setCancelling] = useState(false)
  const [error, setError] = useState('')
  const [status, setStatus] = useState('')
  const [visible, setVisible] = useState(50)
  const mounted = useRef(true)
  const epoch = useRef(0)
  const cancelRequested = useRef(false)
  useEffect(() => {
    mounted.current = true
    const request = ++epoch.current
    void client.workspaceBackupList().then((next) => { if (mounted.current && request === epoch.current) setInventory(next) },
      (caught) => { if (mounted.current && request === epoch.current) setError(String((caught as Error)?.message ?? caught)) })
    return () => { mounted.current = false; epoch.current++; void client.workspaceBackupCancel().catch(() => {}) }
  }, [client])
  const unavailable = disabled || !inventory?.available
  const invalidate = () => { setPreview(null); setConfirmed(false); setError(''); setStatus('') }
  const chooseMode = (next: 'export' | 'restore') => { invalidate(); setMode(next); setPassword(''); void client.workspaceBackupCancel().catch(() => {}) }
  const patch = (next: Partial<WorkspaceBackupSelection>) => { invalidate(); setSelection(previous => ({ ...previous, ...next })) }
  const run = async (label: string, work: () => Promise<void>) => {
    if (busy || unavailable) return
    const request = ++epoch.current
    cancelRequested.current = false; setCancelling(false); setBusy(true); setError(''); setStatus(label)
    try { await work() }
    catch (caught) {
      if (mounted.current && request === epoch.current) {
        const message = String((caught as Error)?.message ?? caught)
        if (cancelRequested.current && /session closed|unavailable; preview again/iu.test(message)) setStatus('Operation cancelled.')
        else { setError(message); setStatus('') }
        setPreview(null); setConfirmed(false)
      }
    } finally {
      if (mounted.current && request === epoch.current) { setBusy(false); setCancelling(false); setPassword('') }
    }
  }
  const makePreview = () => run(mode === 'export' ? 'Preparing a local backup preview…' : 'Checking the selected encrypted backup…', async () => {
    const next = mode === 'export' ? await client.workspaceBackupPreviewExport(selection) : await client.workspaceBackupPreviewRestore({ password, mode: restoreMode })
    if (!mounted.current) return
    setPreview(next); setConfirmed(false)
    setStatus(next ? 'Review the contents and warnings before continuing.' : 'File selection cancelled.')
  })
  const finish = () => preview && run(mode === 'export' ? 'Encrypting and saving the backup…' : 'Restoring the reviewed items; your prior workspace is held for recovery…', async () => {
    if (mode === 'export') {
      const result = await client.workspaceBackupExport({ approvalId: preview.approvalId, password })
      if (!mounted.current) return
      setStatus(result.saved ? `Encrypted backup saved (${Math.ceil((result.bytes ?? 0) / 1024).toLocaleString(currentIntlLocale())} KB).` : 'File selection cancelled.')
      if (!result.saved) return
    } else {
      const result = await client.workspaceBackupApplyRestore({ approvalId: preview.approvalId })
      if (!mounted.current) return
      setStatus(result.cancelled ? 'Restore cancelled; the prior workspace and unsent drafts are preserved.' : 'Reviewed items restored; the workspace is refreshing.')
    }
    setPreview(null); setConfirmed(false)
  })
  const cancel = async () => {
    cancelRequested.current = true
    if (busy) { setCancelling(true); setStatus('Cancellation requested; waiting for the operation and any recovery to finish…') }
    else { invalidate(); setPassword(''); setStatus('Preview cancelled.') }
    try { await client.workspaceBackupCancel() } catch (caught) { setError(String((caught as Error)?.message ?? caught)) }
  }
  const toggleId = (key: 'conversationIds' | 'projectIds' | 'botIds', id: string) => patch({ [key]: selection[key]?.includes(id) ? selection[key]!.filter(value => value !== id) : [...(selection[key] ?? []), id] })
  const focus = 'focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent'
  return <section aria-label={tx("Encrypted workspace backup")} aria-busy={busy} className="space-y-4 rounded-xl border border-line bg-card p-4">
    <div><h3 className="text-[14px] font-medium text-ink">{tx("Encrypted workspace backup")}</h3><p className="mt-1 max-w-prose text-[13px] leading-relaxed text-ink-3">{tx("Save selected conversations, settings, paused bots and unfinished messages in a password-protected file you can restore later.")}</p></div>
    <div className="flex flex-wrap gap-2" aria-label={tx("Backup operation")}>
      <Button aria-pressed={mode === 'export'} disabled={busy} onClick={() => chooseMode('export')} className={focus}>{tx("Create backup")}</Button>
      <Button aria-pressed={mode === 'restore'} disabled={busy} onClick={() => chooseMode('restore')} className={focus}>{tx("Restore backup")}</Button>
    </div>
    {!inventory && !error && <p role="status" className="text-[13px] text-ink-3">{tx("Checking workspace backup access…")}</p>}
    {inventory && !inventory.available && <p role="status" className="text-[13px] text-ink-3">{inventory.reason}</p>}
    {disabled && <p className="text-[13px] text-ink-3">{tx("Save or discard pending settings changes before backing up or restoring the workspace.")}</p>}
    {inventory?.available && mode === 'export' && <fieldset disabled={busy || unavailable} className="min-w-0 space-y-3">
      <legend className="mb-2 text-[13px] font-medium text-ink-2">{tx("Choose what to save")}</legend>
      <div className="grid gap-2 text-[13px] text-ink-2 sm:grid-cols-2">
        <label className="flex items-start gap-2"><input className={focus} type="checkbox" checked={!!selection.settings} onChange={event => patch({ settings: event.target.checked })} />{tx("Preferences without credentials")}</label>
        <label className="flex items-start gap-2"><input className={focus} type="checkbox" checked={!!selection.drafts} onChange={event => patch({ drafts: event.target.checked })} />{tx("All saved unfinished messages")}</label>
        <label className="flex items-start gap-2 sm:col-span-2"><input className={focus} type="checkbox" checked={!!selection.attachments} onChange={event => patch({ attachments: event.target.checked })} />{tx("All retained managed attachment content")}</label>
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2"><span className="text-[13px] text-ink-2">{tx("Saved conversations")}</span><Button size="xs" variant="ghost" onClick={() => patch({ conversationIds: selection.conversationIds?.length === inventory.conversations.length ? [] : inventory.conversations.map(item => item.id) })}>{selection.conversationIds?.length === inventory.conversations.length && inventory.conversations.length ? tx("Clear conversation selection") : tx("Select all conversations")}</Button></div>
      {!inventory.conversations.length && <p className="text-[13px] text-ink-3">{tx("No saved conversations are available; incognito chats are excluded.")}</p>}
      <div className="max-h-48 space-y-2 overflow-auto rounded-lg border border-line p-3">{inventory.conversations.slice(0, visible).map(item => <label key={item.id} className="flex items-start gap-2 text-[13px] text-ink-2"><input className={focus} type="checkbox" checked={!!selection.conversationIds?.includes(item.id)} onChange={() => toggleId('conversationIds', item.id)} /><span className="min-w-0 break-words">{item.title || tx("Untitled conversation")}{item.archived ? tx(" (archived)") : ''}</span></label>)}</div>
      {(['projects', 'bots'] as const).map(collection => <div key={collection} className="space-y-2"><p className="text-[13px] text-ink-2">{collection === 'projects' ? tx("Projects and saved notes") : tx("Bots, restored paused")}</p>{inventory[collection].slice(0, visible).map(item => <label key={item.id} className="flex items-start gap-2 text-[13px] text-ink-2"><input className={focus} type="checkbox" checked={!!selection[collection === 'projects' ? 'projectIds' : 'botIds']?.includes(item.id)} onChange={() => toggleId(collection === 'projects' ? 'projectIds' : 'botIds', item.id)} /><span className="min-w-0 break-words">{item.name}</span></label>)}</div>)}
      {Math.max(inventory.conversations.length, inventory.projects.length, inventory.bots.length) > visible && <Button size="xs" onClick={() => setVisible(value => value + 50)}>{tx("Show more saved items")}</Button>}
    </fieldset>}
    {inventory?.available && mode === 'restore' && <label className="block text-[13px] text-ink-2">{tx("When saved identities already exist")}<select className={`mt-1 w-full min-w-0 rounded-lg border border-line bg-card px-3 py-2 text-[13px] text-ink ${focus}`} value={restoreMode} disabled={busy || unavailable} onChange={event => { invalidate(); setRestoreMode(event.target.value as WorkspaceRestoreMode) }}><option value="merge">{tx("Keep existing records")}</option><option value="replace">{tx("Replace matching records from the backup")}</option></select></label>}
    {inventory?.available && (mode === 'restore' || preview) && <label className="block text-[13px] text-ink-2">{mode === 'export' ? tx("Choose a backup password, at least 8 characters") : tx("Backup password")}<Input type="password" autoComplete={mode === 'export' ? 'new-password' : 'current-password'} maxLength={1024} value={password} disabled={busy || unavailable} onChange={event => setPassword(event.target.value)} className="mt-1" /></label>}
    {preview && <div aria-label={tx("Reviewed backup contents")} className="space-y-3 border-t border-line pt-3">
      <p className="text-[13px] font-medium leading-relaxed text-ink">{describeCounts(preview)}.</p>
      <p className="text-[12px] text-ink-3">{(preview.counts.blobBytes / 1024 / 1024).toFixed(1)} {tx("MB of managed file content; ")}{preview.conflicts.length} {tx("existing identities overlap.")}</p>
      <ul className="list-disc space-y-1 pl-5 text-[13px] leading-relaxed text-ink-3">{preview.warnings.map((warning, index) => <li key={index} className="break-words">{warning}</li>)}</ul>
      {mode === 'restore' && <label className="flex items-start gap-2 text-[13px] text-ink-2"><input type="checkbox" className={focus} checked={confirmed} disabled={busy} onChange={event => setConfirmed(event.target.checked)} />{tx("I reviewed these items, file warnings and the existing-record choice.")}</label>}
      <Button variant={mode === 'restore' ? 'danger' : 'primary'} disabled={busy || unavailable || (mode === 'export' ? password.length < 8 : !confirmed)} onClick={() => void finish()} className={focus}>{mode === 'export' ? tx("Save encrypted backup") : tx("Restore reviewed items")}</Button>
    </div>}
    <div className="flex flex-wrap gap-2">
      {!preview && <Button variant="primary" disabled={busy || unavailable || (mode === 'restore' && password.length < 8)} onClick={() => void makePreview()} className={focus}>{mode === 'export' ? tx("Preview selected backup") : tx("Choose backup and preview")}</Button>}
      {(preview || busy) && <Button disabled={cancelling} onClick={() => void cancel()} className={focus}>{cancelling ? tx("Cancelling…") : tx("Cancel")}</Button>}
    </div>
    {status && <p role="status" aria-live="polite" className="break-words text-[13px] leading-relaxed text-ink-2">{status}</p>}
    {error && <p role="alert" className="break-words text-[13px] leading-relaxed text-danger">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</p>}
    <p className="max-w-prose text-[12px] leading-relaxed text-ink-3">{tx("Keep the password separately; it cannot be recovered here. Credentials, incognito chats, source folders, model downloads and virtual machine disks are excluded. Readable conversation export is a separate action.")}</p>
  </section>
}
