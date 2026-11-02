import { documentEvidenceTool } from './document-evidence-tool'
import path from 'node:path'
import { emailDraftEvent, emailSourceRef, emailSourceId, emailThreadResolvedEvent, readEmailMessage, readEmailThread, scanEmailMessages, summarizeEmail } from './email-connector'
import { readSlackMessage, scanSlackMessages, slackSourceId, slackSourceRef, summarizeSlack } from './slack-connector'
import { calendarMutationEvent, calendarSourceId, calendarSourceRef, readCalendarEvent, scanCalendarEvents, summarizeCalendarDay } from './calendar-connector'
import { documentSourceId, documentSourceRef, readDocumentItem, readTableItem, scanDocumentItems, scanTableItems, summarizeDocuments, summarizeTables } from './document-connector'
import { browserSourceId, browserSourceRef, readBrowserItem, scanBrowserItems, summarizeBrowser } from './browser-connector'
import { connectorName, formatMarkdownTable, formatToolNumber, normalizePersonalMessageConnector, parseStringList, personalMessageDraftEvent } from './tool-labels'
import type { CalendarMutationInput, ConnectorEvent, ConnectorSourceRef, EmailDraftInput } from './types'
import type { ToolContext, ToolResult } from './tool-context'

type PersonalMessageConnectorId = Extract<ConnectorEvent['connectorId'], 'telegram' | 'whatsapp' | 'imessage'>

// Connector reads and draft-producing actions retain their host-controlled ports.
export async function executeConnectorTool(name: string, args: Record<string, any>, ctx: ToolContext): Promise<ToolResult> {
  switch (name) {
    case 'email_list': {
      if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
      const { messages } = await scanEmailMessages(ctx.emailSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        unreadOnly: Boolean(args.unreadOnly),
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
      })
      const lines = messages.map((message) => {
        const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
        return `- [${emailSourceId(message)}] ${message.id} ${message.unread ? '[unread]' : '[read]'} ${when} | ${message.from || 'Unknown'} | ${message.subject || '(no subject)'} | ${message.snippet}`
      })
      return { ok: true, output: lines.length ? lines.join('\n') : 'No email messages matched.\n\nSources: none.', title: 'Email list', meta: `${messages.length} messages`, sources: messages.map(emailSourceRef) }
    }

    case 'email_read': {
      if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
      const message = await readEmailMessage(ctx.emailSettings, String(args.id || ''))
      const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
      const header = [
        `ID: ${message.id}`,
        `Source ID: ${emailSourceId(message)}`,
        `Source: ${message.sourceLabel}`,
        `From: ${message.from || 'Unknown'}`,
        `To: ${message.to || '(not listed)'}`,
        `Date: ${when}`,
        `Subject: ${message.subject || '(no subject)'}`,
      ].join('\n')
      return { ok: true, output: `${header}\n\n${message.body}`, title: message.subject || 'Email', path: message.id, meta: `${message.size} bytes`, sources: [emailSourceRef(message)] }
    }

    case 'email_thread': {
      if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
      const thread = await readEmailThread(ctx.emailSettings, String(args.id || ''))
      const parts = thread.messages.map((message, index) => {
        const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
        return [
          `#${index + 1} [${emailSourceId(message)}] ${message.id}${message.unread ? ' [unread]' : ''}`,
          `From: ${message.from || 'Unknown'}`,
          `To: ${message.to || '(not listed)'}`,
          `Date: ${when}`,
          `Subject: ${message.subject || '(no subject)'}`,
          '',
          message.body || message.snippet,
        ].join('\n')
      })
      const sources = thread.messages.map(emailSourceRef)
      return { ok: true, output: `${thread.summary}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}\n\n${parts.join('\n\n---\n\n')}`, title: thread.root.subject || 'Email thread', path: thread.root.id, meta: `${thread.messages.length} messages`, sources }
    }

    case 'email_summary': {
      if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
      const result = await summarizeEmail(ctx.emailSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        unreadOnly: Boolean(args.unreadOnly),
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 50,
      })
      return { ok: true, output: result.summary, title: 'Email summary', sources: result.sources, meta: `${result.sources.length} sources` }
    }

    case 'email_compose': {
      const input: EmailDraftInput = {
        accountId: typeof args.accountId === 'string' ? args.accountId : undefined,
        mode: args.mode === 'send' ? 'send' : 'draft',
        to: String(args.to || '').trim(),
        subject: String(args.subject || '').trim(),
        body: String(args.body || '').trim(),
        threadKey: typeof args.threadKey === 'string' ? args.threadKey : undefined,
        inReplyTo: typeof args.inReplyTo === 'string' ? args.inReplyTo : undefined,
      }
      if (!input.to || !input.subject || !input.body) return { ok: false, output: 'Email compose requires to, subject, and body.' }
      let sources: ConnectorSourceRef[] = []
      if (ctx.emailSettings && (input.threadKey || input.inReplyTo)) {
        const thread = await readEmailThread(ctx.emailSettings, input.threadKey || input.inReplyTo || '')
        sources = thread.messages.map(emailSourceRef)
        input.threadKey = `email:${thread.root.id}`
        input.inReplyTo = thread.root.id
      }
      const event = ctx.recordEmailDraft ? await ctx.recordEmailDraft(input, sources) : emailDraftEvent(input, Date.now(), sources)
      const sourceLines = event.sources?.length ? `\n\nSources:\n${event.sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}` : ''
      const status = event.requiresApproval ? 'waiting for approval' : 'logged locally'
      return {
        ok: true,
        output: `${event.action} ${status} for ${input.to}.${sourceLines}`,
        title: event.action === 'reply_requested' ? 'Email reply approval' : input.mode === 'send' ? 'Email send request' : event.action === 'reply_draft_created' ? 'Email reply draft' : 'Email draft',
        path: event.threadKey,
        meta: event.approvalId ? `approval ${event.approvalId}` : event.action,
        sources: event.sources ?? sources,
      }
    }

    case 'email_resolve': {
      if (!ctx.emailSettings) return { ok: false, output: 'Email connector is not configured.' }
      const thread = await readEmailThread(ctx.emailSettings, String(args.id || ''))
      const event = emailThreadResolvedEvent({ id: String(args.id || ''), note: typeof args.note === 'string' ? args.note : undefined }, thread)
      const recorded = ctx.recordEmailEvent ? await ctx.recordEmailEvent(event) : event
      const sources = recorded.sources ?? thread.messages.map(emailSourceRef)
      return {
        ok: true,
        output: `Email thread marked resolved locally: ${thread.summary}\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
        title: 'Email thread resolved',
        path: recorded.threadKey,
        meta: `${thread.messages.length} messages`,
        sources,
      }
    }

    case 'message_list': {
      if (!ctx.slackSettings) return { ok: false, output: 'Slack connector is not configured.' }
      const { messages } = await scanSlackMessages(ctx.slackSettings, {
        channel: typeof args.channel === 'string' ? args.channel : undefined,
        query: typeof args.query === 'string' ? args.query : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
      })
      const lines = messages.map((message) => {
        const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
        return `- [${slackSourceId(message)}] #${message.channelName} ${when} | ${message.senderName || 'Unknown'} | ${message.snippet}`
      })
      return { ok: true, output: lines.length ? lines.join('\n') : 'No Slack messages matched.\n\nSources: none.', title: 'Message list', meta: `${messages.length} messages`, sources: messages.map(slackSourceRef) }
    }

    case 'message_read': {
      if (!ctx.slackSettings) return { ok: false, output: 'Slack connector is not configured.' }
      const message = await readSlackMessage(ctx.slackSettings, String(args.id || ''))
      const when = message.date ? new Date(message.date).toLocaleString() : 'no date'
      const source = slackSourceRef(message)
      return {
        ok: true,
        output: [
          `ID: ${message.id}`,
          `Source ID: ${source.id}`,
          `Source: ${message.sourceLabel}`,
          `Channel: #${message.channelName}`,
          `Sender: ${message.senderName || 'Unknown'}`,
          `Date: ${when}`,
          message.threadTs ? `Thread: ${message.threadTs}` : '',
          '',
          message.text,
          '',
          'Sources:',
          `- ${source.id} | ${source.title}`,
        ].filter((line) => line !== '').join('\n'),
        title: `#${message.channelName}`,
        path: message.id,
        meta: message.senderName || 'Slack',
        sources: [source],
      }
    }

    case 'message_summary': {
      if (!ctx.slackSettings) return { ok: false, output: 'Slack connector is not configured.' }
      const result = await summarizeSlack(ctx.slackSettings, {
        channel: typeof args.channel === 'string' ? args.channel : undefined,
        query: typeof args.query === 'string' ? args.query : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 50,
      })
      return { ok: true, output: result.summary, title: 'Message summary', sources: result.sources, meta: `${result.sources.length} sources` }
    }

    case 'message_compose': {
      const connectorId = normalizePersonalMessageConnector(args.connectorId)
      const target = String(args.target || '').trim()
      const body = String(args.body || '').trim()
      const threadKey = typeof args.threadKey === 'string' ? args.threadKey.trim() : ''
      if (!connectorId) return { ok: false, output: 'message_compose requires connectorId to be telegram, whatsapp, or imessage.' }
      if (!target || !body) return { ok: false, output: 'message_compose requires target and body.' }
      const event = personalMessageDraftEvent({ connectorId, target, body, threadKey: threadKey || undefined, accountId: typeof args.accountId === 'string' ? args.accountId : undefined })
      const recorded = ctx.recordMessageEvent ? await ctx.recordMessageEvent(event) : event
      return {
        ok: true,
        output: `${recorded.action} waiting for approval for ${connectorId}:${target}. No external message was sent.`,
        title: `${connectorName(connectorId)} message approval`,
        path: recorded.threadKey,
        meta: recorded.approvalId ? `approval ${recorded.approvalId}` : recorded.action,
      }
    }

    case 'calendar_list': {
      if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
      const { events } = await scanCalendarEvents(ctx.calendarSettings, {
        day: typeof args.day === 'string' ? args.day : undefined,
        from: typeof args.from === 'string' || typeof args.from === 'number' ? args.from : undefined,
        to: typeof args.to === 'string' || typeof args.to === 'number' ? args.to : undefined,
        query: typeof args.query === 'string' ? args.query : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
      })
      const lines = events.map((event) => {
        const start = event.startAt ? new Date(event.startAt).toLocaleString() : 'no start'
        const end = event.endAt ? new Date(event.endAt).toLocaleString() : ''
        return `- [${calendarSourceId(event)}] ${event.id} | ${start}${end ? ` - ${end}` : ''} | ${event.title || '(untitled)'}${event.location ? ` | ${event.location}` : ''}`
      })
      return { ok: true, output: lines.length ? lines.join('\n') : 'No calendar events matched.\n\nSources: none.', title: 'Calendar list', meta: `${events.length} events`, sources: events.map(calendarSourceRef) }
    }

    case 'calendar_read': {
      if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
      const event = await readCalendarEvent(ctx.calendarSettings, String(args.id || ''))
      const start = event.startAt ? new Date(event.startAt).toLocaleString() : 'no start'
      const end = event.endAt ? new Date(event.endAt).toLocaleString() : '(not set)'
      const lines = [
        `ID: ${event.id}`,
        `Source ID: ${calendarSourceId(event)}`,
        `Source: ${event.sourceLabel}`,
        `When: ${event.allDay ? 'all day ' : ''}${start} - ${end}`,
        `Title: ${event.title || '(untitled)'}`,
        event.location ? `Location: ${event.location}` : '',
        event.organizer ? `Organizer: ${event.organizer}` : '',
        event.attendees.length ? `Attendees: ${event.attendees.join(', ')}` : '',
        event.status ? `Status: ${event.status}` : '',
        '',
        event.description || '(no description)',
      ].filter((line) => line !== '').join('\n')
      return { ok: true, output: lines, title: event.title || 'Calendar event', path: event.id, meta: event.allDay ? 'all day' : 'timed event', sources: [calendarSourceRef(event)] }
    }

    case 'calendar_summary': {
      if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
      const result = await summarizeCalendarDay(ctx.calendarSettings, {
        day: typeof args.day === 'string' ? args.day : undefined,
        query: typeof args.query === 'string' ? args.query : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 40,
      })
      return { ok: true, output: result.summary, title: 'Calendar day summary', meta: `${result.sources.length} sources`, sources: result.sources }
    }

    case 'calendar_create': {
      if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
      const enabledSources = ctx.calendarSettings.sources.filter((source) => source.enabled)
      const source = typeof args.sourceId === 'string'
        ? enabledSources.find((item) => item.id === args.sourceId)
        : enabledSources[0]
      if (!source) return { ok: false, output: 'No enabled calendar source is available for the create request.' }
      const input: CalendarMutationInput = {
        mode: 'create',
        sourceId: source.id,
        title: String(args.title || '').trim(),
        startAt: typeof args.startAt === 'string' || typeof args.startAt === 'number' ? args.startAt : null,
        endAt: typeof args.endAt === 'string' || typeof args.endAt === 'number' ? args.endAt : null,
        allDay: typeof args.allDay === 'boolean' ? args.allDay : undefined,
        location: typeof args.location === 'string' ? args.location : undefined,
        description: typeof args.description === 'string' ? args.description : undefined,
        organizer: typeof args.organizer === 'string' ? args.organizer : undefined,
        attendees: parseStringList(args.attendees),
      }
      if (!input.title || !input.startAt) return { ok: false, output: 'calendar_create requires title and startAt.' }
      const event = calendarMutationEvent(input)
      const recorded = ctx.recordCalendarEvent ? await ctx.recordCalendarEvent(event) : event
      return {
        ok: true,
        output: `${recorded.action} waiting for approval. Calendar file is not modified until approval.`,
        title: 'Calendar create approval',
        path: recorded.threadKey,
        meta: recorded.approvalId ? `approval ${recorded.approvalId}` : recorded.action,
        sources: recorded.sources,
      }
    }

    case 'calendar_update': {
      if (!ctx.calendarSettings) return { ok: false, output: 'Calendar connector is not configured.' }
      const id = String(args.id || '').trim()
      if (!id) return { ok: false, output: 'calendar_update requires id.' }
      const existing = await readCalendarEvent(ctx.calendarSettings, id)
      const input: CalendarMutationInput = {
        mode: 'update',
        id,
        sourceId: existing.sourceId,
        title: typeof args.title === 'string' && args.title.trim() ? args.title.trim() : existing.title,
        startAt: typeof args.startAt === 'string' || typeof args.startAt === 'number' ? args.startAt : existing.startAt,
        endAt: typeof args.endAt === 'string' || typeof args.endAt === 'number' ? args.endAt : existing.endAt,
        allDay: typeof args.allDay === 'boolean' ? args.allDay : existing.allDay,
        location: typeof args.location === 'string' ? args.location : existing.location,
        description: typeof args.description === 'string' ? args.description : existing.description,
        organizer: typeof args.organizer === 'string' ? args.organizer : existing.organizer,
        attendees: parseStringList(args.attendees) ?? existing.attendees,
      }
      const event = calendarMutationEvent(input, Date.now(), existing)
      const recorded = ctx.recordCalendarEvent ? await ctx.recordCalendarEvent(event) : event
      const sources = recorded.sources ?? [calendarSourceRef(existing)]
      return {
        ok: true,
        output: `${recorded.action} waiting for approval. Original event cited below; calendar file is not modified until approval.\n\nSources:\n${sources.map((source) => `- ${source.id} | ${source.title}`).join('\n')}`,
        title: 'Calendar update approval',
        path: recorded.threadKey,
        meta: recorded.approvalId ? `approval ${recorded.approvalId}` : recorded.action,
        sources,
      }
    }

    case 'document_list': {
      if (ctx.documentLibrary && typeof args.query === 'string' && args.query.trim()) return await documentEvidenceTool(ctx.documentLibrary, name, args)
      if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
      const { documents } = await scanDocumentItems(ctx.documentSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        path: typeof args.path === 'string' ? args.path : undefined,
        extension: typeof args.extension === 'string' ? args.extension : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
      })
      const lines = documents.map((doc) => `- [${documentSourceId(doc)}] ${doc.relativePath} | ${doc.title || '(untitled)'} | ${doc.excerpt}`)
      return { ok: true, output: lines.length ? lines.join('\n') : 'No documents matched.\n\nSources: none.', title: 'Document list', meta: `${documents.length} documents`, sources: documents.map(documentSourceRef) }
    }

    case 'document_read': {
      if (String(args.id ?? '').startsWith('evidence:')) {
        if (!ctx.documentLibrary) return { ok: false, output: 'Indexed evidence requires an eligible local model and enabled document sources.' }
        return await documentEvidenceTool(ctx.documentLibrary, name, args)
      }
      if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
      const doc = await readDocumentItem(ctx.documentSettings, String(args.id || ''))
      const source = documentSourceRef(doc)
      return {
        ok: true,
        output: [
          `ID: ${doc.id}`,
          `Source ID: ${source.id}`,
          `Source: ${doc.sourceLabel}`,
          `Path: ${doc.relativePath}`,
          `Modified: ${doc.modifiedAt ? new Date(doc.modifiedAt).toLocaleString() : 'unknown'}`,
          '',
          doc.content,
          '',
          'Sources:',
          `- ${source.id} | ${source.title}`,
        ].join('\n'),
        title: doc.title || doc.relativePath,
        path: doc.relativePath,
        meta: `${Math.ceil(doc.size / 1024)} KB`,
        sources: [source],
      }
    }

    case 'document_summary': {
      if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
      const result = await summarizeDocuments(ctx.documentSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        path: typeof args.path === 'string' ? args.path : undefined,
        extension: typeof args.extension === 'string' ? args.extension : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 12,
      })
      return { ok: true, output: result.summary, title: 'Document summary', meta: `${result.sources.length} sources`, sources: result.sources }
    }

    case 'table_list': {
      if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
      const { tables } = await scanTableItems(ctx.documentSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        path: typeof args.path === 'string' ? args.path : undefined,
        extension: typeof args.extension === 'string' ? args.extension : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
      })
      const lines = tables.map((table) => `- [${documentSourceId(table)}] ${table.relativePath} | ${table.rowCount} rows x ${table.columnCount} columns | ${table.columns.join(', ')}`)
      return { ok: true, output: lines.length ? lines.join('\n') : 'No tables matched.\n\nSources: none.', title: 'Table list', meta: `${tables.length} tables`, sources: tables.map(documentSourceRef) }
    }

    case 'table_read': {
      if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
      const table = await readTableItem(ctx.documentSettings, String(args.id || ''))
      const source = documentSourceRef(table)
      const limit = Math.max(1, Math.min(Number(args.limit) || 20, 100))
      return {
        ok: true,
        output: [
          `ID: ${table.id}`,
          `Source ID: ${source.id}`,
          `Source: ${table.sourceLabel}`,
          `Path: ${table.relativePath}`,
          `Shape: ${table.rowCount} rows x ${table.columnCount} columns`,
          table.numericColumns.length ? `Numeric columns: ${table.numericColumns.map((col) => `${col.name} avg ${formatToolNumber(col.average)}`).join('; ')}` : 'Numeric columns: none detected',
          '',
          formatMarkdownTable(table.columns, table.rows.slice(0, limit)),
          '',
          'Sources:',
          `- ${source.id} | ${source.title}`,
        ].join('\n'),
        title: table.title || table.relativePath,
        path: table.relativePath,
        meta: `${table.rowCount} rows x ${table.columnCount} cols`,
        sources: [source],
      }
    }

    case 'table_summary': {
      if (!ctx.documentSettings) return { ok: false, output: 'Document connector is not configured.' }
      const result = await summarizeTables(ctx.documentSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        path: typeof args.path === 'string' ? args.path : undefined,
        extension: typeof args.extension === 'string' ? args.extension : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 12,
      })
      return { ok: true, output: result.summary, title: 'Table summary', meta: `${result.sources.length} sources`, sources: result.sources }
    }

    case 'browser_list': {
      if (!ctx.browserSettings) return { ok: false, output: 'Browser connector is not configured.' }
      const { items } = await scanBrowserItems(ctx.browserSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 20,
      })
      const lines = items.map((item) => {
        const when = item.capturedAt ? new Date(item.capturedAt).toLocaleString() : 'unknown'
        return `- [${browserSourceId(item)}] ${item.title || item.url} | ${when} | ${item.url}${item.excerpt ? ` | ${item.excerpt}` : ''}`
      })
      return { ok: true, output: lines.length ? lines.join('\n') : 'No browser sources matched.\n\nSources: none.', title: 'Browser list', meta: `${items.length} sources`, sources: items.map(browserSourceRef) }
    }

    case 'browser_read': {
      if (!ctx.browserSettings) return { ok: false, output: 'Browser connector is not configured.' }
      const item = await readBrowserItem(ctx.browserSettings, String(args.id || ''))
      const source = browserSourceRef(item)
      return {
        ok: true,
        output: item.content,
        title: item.title || item.url,
        path: item.url,
        meta: item.kind === 'current_tab' ? 'current tab' : 'history export',
        sources: [source],
      }
    }

    case 'browser_summary': {
      if (!ctx.browserSettings) return { ok: false, output: 'Browser connector is not configured.' }
      const result = await summarizeBrowser(ctx.browserSettings, {
        query: typeof args.query === 'string' ? args.query : undefined,
        limit: Number.isFinite(Number(args.limit)) ? Number(args.limit) : 12,
      })
      return { ok: true, output: result.summary, title: 'Browser summary', meta: `${result.sources.length} sources`, sources: result.sources }
    }
    default: return { ok: false, output: `Unknown tool: ${name}` }
  }
}
