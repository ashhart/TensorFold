import { useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import { useAuthoredText } from '../../lib/preferences/useAuthoredText'
import { resolveAssistantModel } from '../../lib/assistant-model'
import { useConversationDraft } from '../../lib/use-conversation-draft'
import { desktopDraftPort } from '../../lib/desktop-drafts'
import { OriginalDiagnostic } from '../OriginalDiagnostic'
import { Button } from '../ui'
import { Composer } from './Composer'
import { WorkActivityMenu } from '../work-activity/WorkActivityMenu'
import type { useGoogleWorkspace } from '../settings/connectors/google/useGoogleWorkspace'

export function AssistantOverview({ google, onGoogle, onInbox }: {
  google: ReturnType<typeof useGoogleWorkspace>; onGoogle: () => void; onInbox: () => void
}) {
  const tx = useAuthoredText()
  const settings = useStore(state => state.settings)
  const launch = useStore(state => state.launch)
  const setView = useStore(state => state.setView)
  const threads = useStore(state => state.threads)
  const openThread = useStore(state => state.openThread)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef(false)
  const textarea = useRef<HTMLTextAreaElement>(null)
  const draft = useConversationDraft({ surface: 'assistant', conversationId: 'assistant-overview', hostId: null, incognito: false }, desktopDraftPort)
  const model = settings ? resolveAssistantModel(settings) : null
  const accounts = google.snapshot?.accounts ?? []
  const chooseModel = () => setView({ name: 'models' })
  const start = async () => {
    if (!draft.ready || !draft.text.trim() || pending.current) return
    if (!settings || !model?.providerId || !model.model) {
      setError(tx('Choose a model in Models before starting an assistant task. Your draft is still here.')); return
    }
    pending.current = true; setBusy(true); setError('')
    const submitted = draft.capture()
    try {
      const text = ['Assistant task.', 'User request:', draft.text.trim(), '',
        'Work draft-first. Do not send, delete, purchase, schedule, or change external state without an explicit approval step.'].join('\n')
      const result = await launch({ text, cwd: settings.defaultCwd || '',
        providerId: model.providerId, providerAccountId: model.providerAccountId, model: model.model,
        approvalMode: settings.approvalMode, projectId: null, persistActiveModel: false })
      if (!result.ok) throw new Error(result.error || tx('The assistant could not start. Your draft is still here.'))
      await draft.clearAccepted(submitted)
    } catch (reason) { setError(reason instanceof Error ? reason.message : tx('The assistant could not start. Your draft is still here.')) }
    finally { pending.current = false; setBusy(false) }
  }
  const suggest = (text: string) => {
    if (!draft.ready) return
    if (draft.text.trim() && draft.text !== text && !confirm(tx('Replace your current draft with this suggestion?'))) return
    draft.setText(text); textarea.current?.focus()
  }
  const recent = [...threads].filter(thread => !thread.archivedAt).sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 5)
  return <div className="mx-auto max-w-[1000px] space-y-5 p-4 sm:p-6">
    <header><div className="flex flex-wrap items-center justify-between gap-2"><h2 className="text-[22px] font-semibold text-ink">{tx('What would you like to get done?')}</h2><WorkActivityMenu variant="toolbar" /></div><p className="mt-2 text-[13px] text-ink-2">{tx('Start a task, read connected conversations, or manage Google mail and calendar.')}</p></header>
    <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-line bg-card p-3">
      <div className="min-w-0"><p className="text-[12px] text-ink-3">{tx('Assistant model')}</p><p className="break-words text-[13px] font-medium text-ink">{model?.model || tx('Choose a model')}</p></div>
      <Button size="sm" onClick={chooseModel}>{tx('Choose a model')}</Button>
    </div>
    {error && <p role="alert" className="text-[13px] text-danger"><OriginalDiagnostic>{error}</OriginalDiagnostic></p>}
    <div className="overflow-hidden rounded-xl border border-line">
      <Composer taskText={draft.text} setTaskText={draft.setText} canDirectReply={false} selectedItem={null} selectedSource="all"
        contextLabel={tx('No inbox conversation is attached')}
        quickActions={[]} taskBusy={busy} sendingReply={false} onSendReply={() => {}} onStartTask={() => { void start() }}
        onDraftTask={suggest} textareaRef={textarea} draftKey={draft.scopeKey} draftReady={draft.ready}
        draftStatus={draft.error ?? (!draft.ready ? 'Loading this conversation draft…' : !desktopDraftPort ? 'Draft kept for this session; use the desktop app to save across restarts.' : draft.saving || draft.dirty ? 'Saving draft…' : 'Draft saved on this device')}
        draftProblem={!!draft.error} sessionOnly={!desktopDraftPort} onRetryDraft={() => { void draft.retry().catch(() => {}) }}
        sendKey={settings?.uiPreferences?.sendKey} intent="task" />
    </div>
    <div className="grid gap-3 sm:grid-cols-2">
      <section className="rounded-xl border border-line bg-card p-4"><h3 className="text-[14px] font-medium text-ink">{tx('Inbox')}</h3><p className="my-2 text-[13px] text-ink-2">{tx('Read conversations, prepare replies, and review actions here.')}</p><Button size="sm" onClick={onInbox}>{tx('Open inbox')}</Button></section>
      <section className="rounded-xl border border-line bg-card p-4"><h3 className="text-[14px] font-medium text-ink">{tx('Google Workspace')}</h3>
        <p className="mt-2 text-[12px] text-ink-3">{tx('Google account tools require a model running locally on this computer.')}</p>
        {google.hostOnly ? <p className="my-2 text-[13px] text-ink-2">{tx('Manage Google accounts locally in TensorFold on the host computer.')}</p>
          : google.error ? <p role="status" className="my-2 text-[13px] text-warn">{tx('Google account status is unavailable.')}</p>
          : !google.snapshot ? <p role="status" className="my-2 text-[13px] text-ink-3">{tx('Loading Google connections…')}</p>
            : accounts.length ? <ul className="my-2 space-y-1">{accounts.map(account => <li key={account.id} className="break-words text-[12px] text-ink-2">{account.email} · {tx(account.status === 'connected' ? 'Connected' : account.status === 'needs-reauthorization' ? 'Reconnect needed' : 'Disconnected')}</li>)}</ul>
              : <p className="my-2 text-[13px] text-ink-2">{tx('Connect your Google account')}</p>}
        <Button size="sm" onClick={onGoogle}>{tx('Open Google Workspace')}</Button>
      </section>
    </div>
    {recent.length > 0 && <section><h3 className="mb-2 text-[14px] font-medium text-ink">{tx('Recent tasks')}</h3><ul className="space-y-1">{recent.map(thread => <li key={thread.id}><button className="flex w-full items-center justify-between gap-3 rounded-lg border border-line bg-card px-3 py-2 text-left text-[13px] text-ink hover:bg-surface focus-visible:outline-2 focus-visible:outline-accent" onClick={() => openThread(thread.id)}><span className="truncate">{thread.title}</span><span className="shrink-0 text-[11px] text-ink-3">{thread.status}</span></button></li>)}</ul></section>}
  </div>
}
