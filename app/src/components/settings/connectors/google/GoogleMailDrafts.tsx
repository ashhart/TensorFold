import { OriginalDiagnostic } from '../../../OriginalDiagnostic'
import { useAuthoredText } from '../../../../lib/preferences/useAuthoredText'
import { useEffect, useState } from 'react'
import { api } from '../../../../lib/ipc'
import { Button, Field, Input } from '../../../ui'
import type { GoogleWorkspaceAccount, GoogleWorkspaceEmailContent, GoogleWorkspaceLocalDraft } from '../../../../../shared/google-workspace'
import { useGoogleAccountRequest } from './useGoogleAccountRequest'

const blank = { to: '', cc: '', bcc: '', subject: '', text: '' }
export function GoogleMailDrafts({ account, refresh, onDirtyChange }: { account: GoogleWorkspaceAccount; refresh: () => Promise<void>; onDirtyChange?: (dirty: boolean) => void }) {
  const tx = useAuthoredText()
  const requests = useGoogleAccountRequest(account)
  const [drafts, setDrafts] = useState<GoogleWorkspaceLocalDraft[]>([])
  const [id, setId] = useState<string | undefined>()
  const [fields, setFields] = useState(blank)
  const [notice, setNotice] = useState('')
  const [prepared, setPrepared] = useState('')
  const [dirty, setDirty] = useState(false)
  useEffect(() => { onDirtyChange?.(dirty) }, [dirty])
  useEffect(() => () => onDirtyChange?.(false), [account.id])
  useEffect(() => { void requests.run('drafts', () => api.googleWorkspaceDrafts(account.id), setDrafts) }, [account.id, account.revision])
  const recipients = (value: string) => value.split(/[,;\n]/).map((item) => item.trim()).filter(Boolean)
  const email: GoogleWorkspaceEmailContent = { to: recipients(fields.to), cc: recipients(fields.cc), bcc: recipients(fields.bcc), subject: fields.subject, text: fields.text }
  const valid = email.to.length > 0 && [...email.to, ...(email.cc ?? []), ...(email.bcc ?? [])].every((address) => /^[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+$/.test(address)) && !!(fields.subject.trim() || fields.text.trim())
  const fingerprint = JSON.stringify(email)
  const edit = (key: keyof typeof blank, value: string) => { setFields((current) => ({ ...current, [key]: value })); setNotice(''); setDirty(true) }
  const choose = (draftId: string) => {
    const draft = drafts.find((item) => item.id === draftId)
    setId(draft?.id); setFields(draft ? { to: draft.to.join(', '), cc: draft.cc?.join(', ') ?? '', bcc: draft.bcc?.join(', ') ?? '', subject: draft.subject, text: draft.text } : blank)
    setNotice(''); setPrepared(''); setDirty(false)
  }
  return <section className="space-y-3 rounded-xl border border-line bg-card p-4" aria-label={tx("Gmail local drafts")}>
    <div><h3 className="text-[14px] font-medium text-ink">{tx("Write an email")}</h3><p className="mt-1 text-[12px] text-ink-3">{tx("Drafts are saved locally in TensorFold for this account; they are not Gmail drafts, and saving never sends an email.")}</p></div>
    <div className="flex flex-wrap gap-2"><select aria-label={tx("Saved local email drafts")} value={id ?? ''} onChange={(event) => choose(event.target.value)} disabled={!!requests.busy || dirty} className="min-w-0 flex-1 rounded-lg border border-line bg-card-2 px-3 py-2 text-[13px] text-ink"><option value="">{tx("New draft")}</option>{drafts.map((draft) => <option key={draft.id} value={draft.id}>{draft.subject || tx("No subject")} · {draft.to.join(', ')}</option>)}</select><Button size="sm" disabled={!!requests.busy || dirty} onClick={() => choose('')}>{tx("New draft")}</Button></div>
    <div className="grid gap-3 sm:grid-cols-2">{(['to', 'cc', 'bcc'] as const).map((key) => <Field key={key} label={key === 'to' ? tx("To") : key.toUpperCase()} hint={key === 'to' ? tx("required") : tx("optional")}><Input aria-label={tx("Email {value0}", { value0: key })} value={fields[key]} disabled={!!requests.busy} onChange={(event) => edit(key, event.target.value)} placeholder={tx("Plain email addresses, separated by commas")} /></Field>)}</div>
    <Field label={tx("Subject")}><Input aria-label={tx("Email subject")} value={fields.subject} disabled={!!requests.busy} onChange={(event) => edit('subject', event.target.value)} /></Field>
    <label className="block text-[13px] text-ink-2">{tx("Message")}<textarea aria-label={tx("Email message")} value={fields.text} disabled={!!requests.busy} onChange={(event) => edit('text', event.target.value)} rows={6} className="mt-1 w-full resize-y rounded-lg border border-line bg-card-2 p-3 text-[13px] leading-relaxed text-ink" /></label>
    {!valid && <p className="text-[12px] text-ink-3">{tx("Add a plain recipient address and a subject or message before saving or preparing a send.")}</p>}
    <div className="flex flex-wrap gap-2"><Button size="sm" disabled={!!requests.busy || !valid} onClick={() => requests.run('save', () => api.googleWorkspaceSaveDraft(account.id, email, id), (saved) => { setId(saved.id); setDrafts((current) => [saved, ...current.filter((item) => item.id !== saved.id)]); setDirty(false); setNotice(tx("Saved locally; nothing was sent to Google.")) })}>{requests.busy === 'save' ? tx("Saving…") : tx("Save local draft")}</Button>{id && <Button size="sm" variant="danger" disabled={!!requests.busy} onClick={() => requests.run('delete', () => api.googleWorkspaceDeleteDraft(account.id, id), () => { setDrafts((current) => current.filter((item) => item.id !== id)); choose(''); setNotice(tx("Local draft deleted; Gmail was unchanged.")) })}>{tx("Delete local draft")}</Button>}<Button size="sm" variant="primary" disabled={!!requests.busy || !valid || !account.capabilities.includes('gmail.send') || prepared === fingerprint} onClick={() => requests.run('prepare', () => api.googleWorkspacePrepareAction(account.id, { kind: 'gmail.send', email }), () => { setPrepared(fingerprint); setNotice(tx("Ready for exact-content review below; no email has been sent.")); void refresh().catch(() => setNotice(tx("The review was prepared; refresh Google connections to load it before approving."))) })}>{requests.busy === 'prepare' ? tx("Preparing…") : tx("Review email before sending")}</Button></div>
    {!account.capabilities.includes('gmail.send') && <p className="text-[12px] text-ink-3">{tx("Reconnect with Send reviewed emails access to prepare a send; local drafts remain available.")}</p>}
    {dirty && <div className="flex flex-wrap items-center gap-2"><p role="status" className="text-[12px] text-warn">{tx("Unsaved email edits; save locally before switching accounts or leaving this page.")}</p><Button size="sm" disabled={!!requests.busy} onClick={() => choose(id ?? '')}>{tx("Discard unsaved edits")}</Button></div>}
    {requests.error && <p role="alert" className="text-[13px] text-danger">{<OriginalDiagnostic>{requests.error}</OriginalDiagnostic>}</p>}{notice && <p role="status" className="text-[13px] text-ink-2">{notice}</p>}
  </section>
}
