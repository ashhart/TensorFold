import { useGmailInbox } from './assistant/useGmailInbox'
import { GmailInboxStatus } from './assistant/GmailInboxStatus'
import { mergeGmailInboxSnapshot } from './assistant/gmail-inbox-view'
import { gmailDraftBinding, gmailTaskContext, resolveInboxApproval } from './assistant/gmail-inbox-client'
import { OriginalDiagnostic } from './OriginalDiagnostic'
import { useAuthoredText } from '../lib/preferences/useAuthoredText'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useStore } from '../lib/store'
import { api } from '../lib/ipc'
import { discardPendingDraft } from '../lib/draft-approval'
import type { ConnectorEvent, ConnectorId } from '../lib/types'
import { resolveAssistantModel } from '../lib/assistant-model'
import { MessageCardView, bubbleToCard } from './assistant/MessageCard'
import { eventToBubble, mergeBubbles, groupEventsIntoThreads, combineItemsByThread, type BubbleRow } from './assistant/conversation'
import { PendingApprovalsBar } from './assistant/PendingApprovalsBar'
import { OutboundApprovalDialog } from './assistant/OutboundApprovalDialog'
import { itemFromAttention, itemFromEvent, mergeSourceRows, sourceFromConnector, sourceName } from './assistant/mappings'
import { assistantTaskContext, emailThreadContext, quickActionsFor } from './assistant/task-context'
import { AssistantHeader } from './assistant/AssistantHeader'
import { ImessageFdaBanner, WhatsappQrBanner } from './assistant/Banners'
import { TriageRail } from './assistant/TriageRail'
import { Composer } from './assistant/Composer'
import type { AssistantItem, SourceKey } from './assistant/types'
import { assistantDraftScope } from '../../shared/assistant-drafts'
import { useConversationDraft } from '../lib/use-conversation-draft'
import { desktopDraftPort } from '../lib/desktop-drafts'
import { insertTextareaText } from '../lib/textarea-editing'
import { useAssistantSnapshot } from './assistant/useAssistantSnapshot'
import { ConversationHeading, EmptyInbox, InboxLoadState } from './assistant/InboxStates'
import { isAwaitingApproval } from './assistant/pending-approvals'
import { Button } from './ui'
import { AssistantWorkspace } from './assistant/AssistantWorkspace'

export function Assistant() {
  return <AssistantWorkspace><AssistantInbox /></AssistantWorkspace>
}

function AssistantInbox() {
  const tx = useAuthoredText()
  const launch = useStore((s) => s.launch)
  const settings = useStore((s) => s.settings)
  const setView = useStore((s) => s.setView)
  const setAssistantSource = useStore((s) => s.setAssistantSource)

  const selectedSource = useStore((s) => s.assistantSource) as SourceKey
  const [selectedItemId, setSelectedItemId] = useState('')
  const { snapshot: baseSnapshot, loading, error: refreshError, refreshedAt, paused, refresh: refreshSnapshot } = useAssistantSnapshot()
  const [error, setError] = useState('')
  const [taskBusy, setTaskBusy] = useState(false)
  const [sendingReply, setSendingReply] = useState(false)
  const actionPending = useRef(false)
  const [resolvingEmail, setResolvingEmail] = useState(false)
  const [syncingEmail, setSyncingEmail] = useState(false)
  const [notice, setNotice] = useState('')
  const [reviewEvent, setReviewEvent] = useState<ConnectorEvent | null>(null)
  const [reviewBusy, setReviewBusy] = useState(false)
  const gmail = useGmailInbox()
  const snapshot = useMemo(() => mergeGmailInboxSnapshot(baseSnapshot, gmail), [baseSnapshot, gmail.snapshot, gmail.opened])
  const [composerIntent, setComposerIntent] = useState<'reply' | 'task'>('reply')
  const manageSources = () => setView({ name: 'settings', section: 'connectors' })

  const refresh = async () => {
    setError('')
    await Promise.all([refreshSnapshot(), gmail.refresh()])
  }

  const sourceRows = useMemo(() => {
    const rows = snapshot ? snapshot.connectors.map(sourceFromConnector) : []
    return mergeSourceRows(rows)
  }, [snapshot])

  const attentionItems = useMemo(() => {
    if (!snapshot) return []
    const mapped = snapshot.attention?.length
      ? snapshot.attention.map(itemFromAttention)
      : groupEventsIntoThreads(snapshot.events).slice(0, 50).map(({ event, count }) => ({
          ...itemFromEvent(event),
          id: (event.threadKey && event.threadKey.trim()) || event.id,
          count,
        }))

    if (snapshot.attention?.length) mapped.push(...groupEventsIntoThreads(snapshot.events.filter(event => event.payload?.gmailInbox === true)).map(({ event, count }) => ({ ...itemFromEvent(event), id: event.threadKey || event.id, count })))
    return combineItemsByThread(mapped).map((item) => {
      const latest = snapshot.events.find((event) => event.id === item.eventId)
      if (!item.threadKey || !latest || !['outbound', 'internal'].includes(latest.direction)) return item
      const incoming = snapshot.events.filter((event) => event.threadKey === item.threadKey && event.direction === 'inbound').sort((a, b) => b.createdAt - a.createdAt)[0]
      return incoming ? { ...item, from: incoming.actor || incoming.pluginName } : item
    })
  }, [snapshot])

  const filteredItems = useMemo(() => (
    selectedSource === 'all' ? attentionItems : attentionItems.filter((item) => item.source === selectedSource)
  ), [attentionItems, selectedSource])

  const selectedItem = filteredItems.find((item) => item.id === selectedItemId) ?? filteredItems[0] ?? null
  const selectedEvent = selectedItem?.eventId ? snapshot?.events.find((event) => event.id === selectedItem.eventId) ?? null : null
  const gmailBinding = useMemo(() => selectedItem?.threadKey?.startsWith('gmail:') ? gmailDraftBinding(selectedItem.threadKey) : null, [selectedItem?.threadKey])
  const draftContext = useMemo(() => {
    if (gmailBinding) return { scope: gmailBinding.scope, problem: null }
    try {
      const event = selectedEvent ?? (selectedItem ? { id: selectedItem.eventId ?? selectedItem.id, connectorId: selectedItem.source, threadKey: selectedItem.threadKey, sources: selectedItem.sources } : null)
      return { scope: assistantDraftScope(selectedSource, event, settings), problem: null }
    } catch {
      return { scope: { surface: 'assistant' as const, conversationId: `assistant-unavailable:${selectedSource}:${(selectedEvent?.id ?? selectedItem?.id ?? 'overview').slice(0, 512)}`, hostId: null, incognito: false }, problem: tx("This conversation account is unavailable or changed; this draft stays in this session until you refresh the source.") }
    }
  }, [selectedSource, selectedItem, selectedEvent, settings, tx, gmailBinding])
  const draft = useConversationDraft(draftContext.scope, gmailBinding ? gmailBinding.port : draftContext.problem ? null : desktopDraftPort)
  const { text: taskText, setText: setTaskText } = draft
  const composerRef = useRef<HTMLTextAreaElement>(null)
  // A two-way comms thread (not docs/calendar) can take a direct reply (connector:reply).
  const canDirectReply = !!selectedItem?.threadKey && (['email', 'slack', 'discord', 'whatsapp', 'imessage'] as ConnectorId[]).includes(selectedItem.source)
  useEffect(() => { setComposerIntent(canDirectReply ? 'reply' : 'task') }, [draft.scopeKey, canDirectReply])
  const pendingEvents = useMemo(() => snapshot?.events.filter(isAwaitingApproval) ?? [], [snapshot])
  const selectedPending = pendingEvents.filter((event) => event.threadKey === selectedItem?.threadKey && (event.outbound || event.outboundError || /send_requested|reply_requested|automation_result_ready/.test(event.action)))

  useEffect(() => {
    if (filteredItems[0] && !filteredItems.some((item) => item.id === selectedItemId)) {
      setSelectedItemId(filteredItems[0].id)
    }
  }, [filteredItems, selectedItemId])

  const selectedSourceRow = selectedSource === 'all' ? null : sourceRows.find((source) => source.id === selectedSource) ?? null
  const openSignals = filteredItems.length
  const urgentCount = filteredItems.filter((item) => item.priority === 'urgent').length

  const conversationBubbles: BubbleRow[] = useMemo(() => {
    const rows: BubbleRow[] = []
    if (snapshot) {

      const scoped = selectedItem?.threadKey
        ? snapshot.events.filter((e) => e.threadKey === selectedItem.threadKey)
        : selectedSource === 'all'
          ? snapshot.events.slice(0, 12)
          : snapshot.events.filter((e) => e.connectorId === selectedSource).slice(0, 12)
      for (const e of scoped) rows.push(eventToBubble({
        id: e.id, actor: e.actor, title: e.title, summary: e.summary, detail: e.summary,
        text: typeof e.payload?.text === 'string' ? e.payload.text : undefined,
        action: e.action, direction: e.direction, connectorId: e.connectorId,
        threadKey: e.threadKey ?? undefined, createdAt: e.createdAt,
        approvalStatus: e.approvalStatus ?? undefined,
        outbound: e.outbound, delivery: e.delivery,
      }))
    }
    return mergeBubbles(rows)
  }, [snapshot, selectedItem?.threadKey, selectedSource])

  const conversationCards = useMemo(() => conversationBubbles.map(bubbleToCard), [conversationBubbles])

  const quickActions = quickActionsFor(selectedItem?.source ?? selectedSource)

  const draftTask = (text: string, intent: 'reply' | 'task' = 'task') => {
    if (!draft.ready) return false
    if (taskText.trim() && taskText !== text && !confirm(tx('Replace your current draft with this suggestion?'))) return false
    const textarea = composerRef.current
    setTaskText(textarea ? insertTextareaText(textarea, text, 0, textarea.value.length) : text)
    setComposerIntent(intent)
    textarea?.focus()
    setNotice('')
    return true
  }

  const sendReply = async () => {
    const text = taskText.trim()
    const key = selectedItem?.threadKey
    if (!draft.ready || !text || !key || actionPending.current) return
    if (!gmailBinding && !confirm(tx("Send this reply to {value0} via {value1}?", { value0: selectedItem!.from, value1: sourceName(selectedItem!.source) }))) return
    actionPending.current = true
    const submitted = draft.capture()
    setSendingReply(true)
    setNotice('')
    try {
      if (gmailBinding) {
        await api.gmailInboxReply(key, text)
        await draft.clearAccepted(submitted)
        setNotice(tx('Gmail reply prepared for review. Nothing was sent.'))
        await refresh(); return
      }
      const r = await api.replyToThread(key, text)
      if (r.ok) { await draft.clearAccepted(submitted); setNotice(tx("Sent to {value0}.", { value0: selectedItem!.from })); void refresh() }
      else setNotice(r.error || tx("Could not send the reply."))
    } catch (e: any) {
      setNotice(e?.message ?? tx("Could not send the reply."))
    } finally {
      actionPending.current = false
      setSendingReply(false)
    }
  }

  const startTask = async () => {
    const text = taskText.trim()
    if (!draft.ready || !text || actionPending.current) return
    const selection = settings ? resolveAssistantModel(settings) : null
    if (!settings || !selection?.providerId || !selection.model) {
      setNotice(tx("Choose a model in Models before starting an assistant task. Your draft is still here."))
      return
    }
    actionPending.current = true
    const submitted = draft.capture()
    setTaskBusy(true)
    setNotice('')
    try {
      {
        const scope = selectedSource === 'all' ? 'all assistant sources' : sourceName(selectedSource)
        let sourceContext = assistantTaskContext(scope, selectedItem, selectedEvent, filteredItems.filter(item => !item.threadKey?.startsWith('gmail:')), snapshot?.briefing)
        if (gmailBinding && selectedItem?.threadKey) {
          sourceContext = `${sourceContext}\n\n${gmailTaskContext(await api.gmailInboxThread(selectedItem.threadKey))}`
        } else if (selectedItem?.source === 'email' && selectedItem.threadKey) {
          try {
            const thread = await api.readEmailThread(selectedItem.threadKey)
            sourceContext = `${sourceContext}\n\n${emailThreadContext(thread)}`
          } catch (err) {
            sourceContext = `${sourceContext}\n\n<email_thread_read_error>${err instanceof Error ? err.message : String(err)}</email_thread_read_error>`
          }
        }
        const result = await launch({
          text: [
            `Assistant task for ${scope}.`,
            sourceContext,
            '',
            'User request:',
            text,
            '',
            'Work draft-first. Do not send, delete, purchase, schedule, or change external state without an explicit approval step.',
          ].join('\n'),
          cwd: settings.defaultCwd || '',

          providerId: selection.providerId,
          providerAccountId: selection.providerAccountId,
          model: selection.model,
          approvalMode: settings.approvalMode,
          projectId: null,

          sourceThreadKey: selectedItem?.threadKey,
          persistActiveModel: false,
        })
        if (!result.ok) throw new Error(result.error || tx("The assistant could not start. Your draft is still here."))
      }
      await draft.clearAccepted(submitted)
    } catch (err) {
      setNotice(err instanceof Error ? err.message : String(err))
    } finally {
      actionPending.current = false
      setTaskBusy(false)
    }
  }

  const syncEmail = async () => {
    setSyncingEmail(true)
    setError('')
    setNotice('')
    try {
      const result = await api.syncEmail()
      await refresh()
      setNotice(tx("Email synced: {value0} messages scanned, {value1} new events.", { value0: result.messages, value1: result.events.length }))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setSyncingEmail(false)
    }
  }

  const [syncingSources, setSyncingSources] = useState(false)
  const syncAllSources = async () => {
    setSyncingSources(true)
    setNotice('')
    const results: string[] = []
    try {
      if (settings?.email.sources.some((s) => s.enabled)) {
        try { const r = await api.syncEmail(); results.push(`${r.events.length} new email`) } catch {  }
      }
      if (settings?.imessage.enabled) {

        const access = await api.imessageAccess().catch(() => ({ state: 'error', detail: tx("Could not check iMessage access.") }))
        if (access.state === 'needs_full_disk_access') {
          setNeedsFdaAccess(true)
          results.push('iMessage: needs Full Disk Access')
        } else {
          try { const r = await api.syncImessage(); results.push(`${r.messages} iMessage`) } catch (e: any) { results.push(`iMessage: ${e?.message ?? 'sync failed'}`) }
        }
      }
      await refresh()
      setNotice(results.length ? tx("Synced: {value0}.", { value0: results.join(', ') }) : tx("Nothing to sync — enable a connector in Sources."))
    } finally {
      setSyncingSources(false)
    }
  }

  const [whatsappQrs, setWhatsappQrs] = useState<Record<string, string>>({})
  useEffect(() => {
    const off = api.onWhatsAppQr(({ sourceId, qr }) => setWhatsappQrs((q) => ({ ...q, [sourceId]: qr })))
    const offStatus = api.onWhatsAppStatus(({ sourceId, paired }) => {
      if (paired) setWhatsappQrs((q) => { const next = { ...q }; delete next[sourceId]; return next })
    })
    return () => { off(); offStatus() }
  }, [])

  const [needsFdaAccess, setNeedsFdaAccess] = useState(false)
  const openFdaPane = () => { void api.openFullDiskAccess().catch(() => {}) }

  const resolveSelectedEmail = async () => {
    if (!selectedItem?.threadKey) return
    setResolvingEmail(true)
    setError('')
    setNotice('')
    try {
      await api.resolveEmailThread({ id: selectedItem.threadKey, note: `Resolved from Assistant: ${selectedItem.title}` })
      await refresh()
      setNotice(tx("Email thread marked resolved locally."))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setResolvingEmail(false)
    }
  }

  const approveDraft = (id: string) => {
    const event = snapshot?.events.find((e) => e.id === id && isAwaitingApproval(e))
    if (!event) { setNotice(tx("This draft has no pending approval to resolve.")); return }
    setError(''); setNotice(''); setReviewEvent(event)
  }
  const confirmDraft = async (reviewDigest: string) => {
    if (!reviewEvent || reviewBusy) return
    setReviewBusy(true); setError(''); setNotice('')
    try {
      const result = await resolveInboxApproval(reviewEvent, 'approve', reviewDigest)
      setReviewEvent(null)
      await refresh()
      setNotice(result.delivery?.transport === 'gmail' && result.delivery.status === 'queued' ? tx('Gmail accepted this message for sending.') : result.delivery?.status === 'delivered' ? tx("Message delivered.") : result.delivery?.status === 'uncertain' ? tx("Delivery is uncertain; check the destination before sending again.") : result.delivery?.status === 'failed' ? result.delivery.error || tx("Message delivery failed.") : tx("Draft approved; no delivery receipt yet."))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      await refresh()
    } finally { setReviewBusy(false) }
  }
  const discardDraft = async (id: string) => {
    const event = snapshot?.events.find((e) => e.id === id && e.requiresApproval)
    if (!event) {
      setError(tx("This draft has no pending approval to discard."))
      return
    }
    setError(''); setNotice('')
    try {
      await discardPendingDraft(() => resolveInboxApproval(event, 'deny'), refresh, event.id)
      setNotice(tx("Draft discarded."))
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    }
  }
  // "Edit" a draft: load its text into the composer so you can revise + resend.
  const draftTextFor = (id: string): string => {
    const row = conversationBubbles.find((r) => r.id === id)
    return row?.text ?? ''
  }
  const editDraft = (id: string) => {
    const text = draftTextFor(id)
    if (!text) { setNotice(tx("Nothing to edit in this draft.")); return }
    if (draftTask(text, 'reply')) setNotice(tx("Draft loaded into the composer — revise and send."))
  }
  // Ask the assistant to redraft a pending reply in a different register.
  const redraftDraft = (id: string, how: string) => {
    const text = draftTextFor(id)
    draftTask(`Redraft this reply to be ${how}. Keep the same intent and recipient; reply with the new draft only.\n\n---\n${text}`)
  }

  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas">
      {reviewEvent && <OutboundApprovalDialog event={reviewEvent} busy={reviewBusy} error={error} onConfirm={(digest) => void confirmDraft(digest)} onClose={() => setReviewEvent(null)} />}
      <main className="scroll-thin flex min-h-0 flex-1 flex-col overflow-y-auto">
        <GmailInboxStatus gmail={gmail} selectedKey={selectedItem?.threadKey} />
        <AssistantHeader
          snapshot={snapshot}
          selectedSourceRow={selectedSourceRow}
          selectedItem={selectedItem}
          selectedSource={selectedSource}
          openSignals={openSignals}
          urgentCount={urgentCount}
          loading={loading}
          refreshError={refreshError}
          refreshedAt={refreshedAt}
          paused={paused}
          syncingSources={syncingSources}
          syncingEmail={syncingEmail}
          onSyncAll={syncAllSources}
          onRefresh={refresh}
          onSyncEmail={syncEmail}
          onBriefMe={draftTask}
          onManageSources={manageSources}
        />

        {refreshError && snapshot && <div role="status" className="flex shrink-0 flex-wrap items-center justify-between gap-3 border-b border-warn/30 bg-warn/10 px-4 py-3 text-[13px] text-ink-2 sm:px-6">
          <div><p className="font-medium">{tx('Inbox check failed. Your conversation and draft are still here.')}</p><details className="mt-1 text-[12px] text-ink-3"><summary className="cursor-pointer">{tx('Connection details')}</summary><p className="mt-1 break-words"><OriginalDiagnostic>{refreshError}</OriginalDiagnostic></p></details></div>
          <Button size="sm" icon="refresh" disabled={loading} onClick={refresh}>{loading ? tx('Retrying…') : tx('Retry now')}</Button>
        </div>}
        {error && <div role="alert" className="shrink-0 border-b border-danger/25 bg-danger/10 px-4 py-3 text-[13px] text-danger sm:px-6">{<OriginalDiagnostic>{error}</OriginalDiagnostic>}</div>}
        {notice && <div role="status" className="flex shrink-0 items-start justify-between gap-3 border-b border-line bg-card-2 px-4 py-2 text-[13px] text-ink-2 sm:px-6"><span>{notice}</span><button type="button" className="shrink-0 rounded px-1 underline focus-visible:outline-2 focus-visible:outline-accent" onClick={() => setNotice('')}>{tx('Dismiss')}</button></div>}

        {}
        {Object.keys(whatsappQrs).length > 0 && (
          <WhatsappQrBanner qr={Object.values(whatsappQrs)[0]} />
        )}

        {}
        {needsFdaAccess && (
          <ImessageFdaBanner onOpen={() => { void openFdaPane() }} onDismiss={() => setNeedsFdaAccess(false)} />
        )}

        {}
        <PendingApprovalsBar
          events={snapshot?.events ?? []}
          onResolved={() => { void refresh() }}
          onOpen={(threadKey) => { setAssistantSource('all'); setSelectedItemId(threadKey) }}
        />

        {!snapshot ? <InboxLoadState loading={loading} error={refreshError} onRetry={refresh} /> : <div className="flex min-h-0 flex-1 flex-col md:flex-row">
          <TriageRail
            items={filteredItems}
            selectedItemId={selectedItem?.id ?? null}
            onSelect={setSelectedItemId}
            selectedSource={selectedSource}
            sources={sourceRows}
            onSourceChange={setAssistantSource}
            pendingThreadKeys={pendingEvents.map((event) => event.threadKey || event.id)}
          />

          {}
          <div className="flex min-h-0 min-w-0 flex-1 flex-col bg-canvas">
            {selectedItem && <ConversationHeading item={selectedItem} messageCount={conversationCards.length} pending={selectedPending.length} onReview={() => approveDraft(selectedPending[0].id)} />}
            {selectedItem ? <MessageCardView
              key={selectedItem.id}
              items={conversationCards}
              emptyTitle={tx('No messages in this conversation yet')}
              emptyDetail={tx('Ask TensorFold to prepare a reply or check the inbox again.')}
              onApprove={approveDraft}
              onEdit={editDraft}
              onDiscard={discardDraft}
              onRedraft={redraftDraft}
            /> : <EmptyInbox onManageSources={manageSources} onPrepareBriefing={() => { setComposerIntent('task'); composerRef.current?.focus() }} />}

            <Composer
              sendKey={settings?.uiPreferences?.sendKey}
              draftKey={draft.scopeKey}
              textareaRef={composerRef}
              draftReady={draft.ready && !!snapshot}
              draftStatus={draftContext.problem ?? draft.error ?? (!draft.ready || !snapshot ? 'Loading this conversation draft…' : !desktopDraftPort ? 'Draft kept for this session; use the desktop app to save across restarts.' : draft.saving || draft.dirty ? 'Saving draft…' : 'Draft saved on this device')}
              draftProblem={!!(draftContext.problem || draft.error)}
              sessionOnly={!desktopDraftPort || !!draftContext.problem}
              intent={composerIntent}
              onIntentChange={setComposerIntent}
              onRetryDraft={() => { void draft.retry().catch(() => {}) }}
              taskText={taskText}
              setTaskText={setTaskText}
              canDirectReply={canDirectReply}
              selectedItem={selectedItem}
              selectedSource={selectedSource}
              quickActions={quickActions}
              taskBusy={taskBusy}
              sendingReply={sendingReply}
              reviewReply={!!gmailBinding}
              onSendReply={sendReply}
              onStartTask={startTask}
              onDraftTask={draftTask}
            />
          </div>
        </div>}
      </main>
    </div>
  )
}
