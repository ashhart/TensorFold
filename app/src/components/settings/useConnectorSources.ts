import { useRef, useState } from 'react'
import { api } from '../../lib/ipc'
import { formatSyncSummary, type ConnectorStatus } from '../../lib/connector-status'
import type { BrowserSourceConfig, DocumentSourceConfig, EmailSourceConfig, Settings as TSettings, SlackSourceConfig } from '../../lib/types'
import type { SettingsHookDeps } from './types'

// Email / Slack / Documents / Browser source add/update/remove/sync logic, plus
// the transient per-connector success flash and the browser-tab capture fields.
// Owns its busy flags + the flash timer ref; the commit* helpers close over the
// draft + commit primitive passed in from Settings.tsx.
export function useConnectorSources({ draft, commit }: SettingsHookDeps) {
  const [emailBusy, setEmailBusy] = useState('')
  const [slackBusy, setSlackBusy] = useState('')
  const [documentBusy, setDocumentBusy] = useState('')
  const [browserBusy, setBrowserBusy] = useState('')
  // Transient per-connector confirmation so a successful add/save/sync is visible
  // even when it doesn't change visible state (e.g. a sync that finds 0 new items).
  // Errors already surface via the global IPC toast — this only confirms success.
  const [connectorStatus, setConnectorStatus] = useState<Record<string, ConnectorStatus>>({})
  const connectorStatusTimers = useRef<Record<string, ReturnType<typeof setTimeout>>>({})
  const flashConnectorStatus = (key: string, msg: string, state: ConnectorStatus['state'] = 'ok') => {
    setConnectorStatus((s) => ({ ...s, [key]: { state, msg } }))
    clearTimeout(connectorStatusTimers.current[key])
    connectorStatusTimers.current[key] = setTimeout(() => {
      setConnectorStatus((s) => { const next = { ...s }; delete next[key]; return next })
    }, 4500)
  }
  const [browserTabUrl, setBrowserTabUrl] = useState('')
  const [browserTabTitle, setBrowserTabTitle] = useState('')
  const [browserTabExcerpt, setBrowserTabExcerpt] = useState('')
  const [browserHistoryPath, setBrowserHistoryPath] = useState('')

  const commitEmail = (email: TSettings['email']) => commit({ ...draft, email })
  const commitSlack = (slack: TSettings['slack']) => commit({ ...draft, slack })
  const commitDocuments = (documents: TSettings['documents']) => commit({ ...draft, documents })
  const commitBrowser = (browser: TSettings['browser']) => commit({ ...draft, browser })

  const addEmailSource = async () => {
    const dir = await api.pickDirectory()
    if (!dir) return
    setEmailBusy('add')
    try { commitEmail(await api.addEmailSource(dir)); flashConnectorStatus('email', 'Source added') } finally { setEmailBusy('') }
  }
  const updateEmailSource = async (id: string, sourcePatch: Partial<EmailSourceConfig>) => {
    commitEmail(await api.updateEmailSource(id, sourcePatch))
    flashConnectorStatus('email', 'Saved')
  }
  const removeEmailSource = async (id: string) => {
    commitEmail(await api.removeEmailSource(id))
    flashConnectorStatus('email', 'Source removed')
  }
  const syncEmail = async () => {
    setEmailBusy('sync')
    try {
      const result = await api.syncEmail()
      commitEmail({ ...draft.email, sources: result.sources, lastSyncAt: result.syncedAt })
      flashConnectorStatus('email', formatSyncSummary(result.sources.length, result.messages ?? 0, 'message'))
    } finally {
      setEmailBusy('')
    }
  }
  // IMAP accounts are email sources of kind 'imap'. We add a blank one and let the
  // user fill host/user/password in the UI; the password is secret-protected at rest.
  const addImapAccount = () => {
    const id = `email_${Math.random().toString(36).slice(2, 10)}`
    const source: EmailSourceConfig = {
      id, label: 'New IMAP account', path: '', kind: 'imap', enabled: true,
      host: '', port: 993, tls: true, user: '', password: '',
      createdAt: Date.now(), updatedAt: Date.now(), lastSyncAt: null, lastError: null,
    }
    commitEmail({ ...draft.email, sources: [source, ...draft.email.sources] })
  }
  const patchImapAccount = (id: string, patch: Partial<EmailSourceConfig>) => {
    commitEmail({ ...draft.email, sources: draft.email.sources.map((s) => s.id === id ? { ...s, ...patch, updatedAt: Date.now() } : s) })
  }
  const addSlackSource = async () => {
    const dir = await api.pickDirectory()
    if (!dir) return
    setSlackBusy('add')
    try { commitSlack(await api.addSlackSource(dir)); flashConnectorStatus('slack', 'Export added') } finally { setSlackBusy('') }
  }
  const updateSlackSource = async (id: string, sourcePatch: Partial<SlackSourceConfig>) => {
    commitSlack(await api.updateSlackSource(id, sourcePatch))
    flashConnectorStatus('slack', 'Saved')
  }
  const removeSlackSource = async (id: string) => {
    commitSlack(await api.removeSlackSource(id))
    flashConnectorStatus('slack', 'Source removed')
  }
  const syncSlack = async () => {
    setSlackBusy('sync')
    try {
      const result = await api.syncSlack()
      commitSlack({ ...draft.slack, sources: result.sources, lastSyncAt: result.syncedAt })
      flashConnectorStatus('slack', formatSyncSummary(result.sources.length, result.messages ?? 0, 'message'))
    } finally {
      setSlackBusy('')
    }
  }
  const addDocumentSource = async () => {
    const dir = await api.pickDirectory()
    if (!dir) return
    setDocumentBusy('add')
    try { commitDocuments(await api.addDocumentSource(dir)); flashConnectorStatus('documents', 'Source added') } finally { setDocumentBusy('') }
  }
  const updateDocumentSource = async (id: string, sourcePatch: Partial<DocumentSourceConfig>) => {
    commitDocuments(await api.updateDocumentSource(id, sourcePatch))
    flashConnectorStatus('documents', 'Saved')
  }
  const removeDocumentSource = async (id: string) => {
    commitDocuments(await api.removeDocumentSource(id))
    flashConnectorStatus('documents', 'Source removed')
  }
  const syncDocuments = async () => {
    setDocumentBusy('sync')
    try {
      const result = await api.syncDocuments()
      commitDocuments({ ...draft.documents, sources: result.sources, lastSyncAt: result.syncedAt })
      flashConnectorStatus('documents', formatSyncSummary(result.sources.length, result.documents.length, 'document'))
    } finally {
      setDocumentBusy('')
    }
  }
  const addBrowserCurrentTabSource = async () => {
    const url = browserTabUrl.trim()
    if (!url) return
    setBrowserBusy('add-tab')
    try {
      commitBrowser(await api.addBrowserCurrentTabSource({ url, title: browserTabTitle.trim(), excerpt: browserTabExcerpt.trim(), capturedAt: Date.now() }))
      setBrowserTabUrl('')
      setBrowserTabTitle('')
      setBrowserTabExcerpt('')
      flashConnectorStatus('browser', 'Current tab saved')
    } finally {
      setBrowserBusy('')
    }
  }
  const addBrowserHistorySource = async () => {
    const target = browserHistoryPath.trim()
    if (!target) return
    setBrowserBusy('add-history')
    try {
      commitBrowser(await api.addBrowserHistorySource(target))
      setBrowserHistoryPath('')
      flashConnectorStatus('browser', 'History export added')
    } finally {
      setBrowserBusy('')
    }
  }
  const updateBrowserSource = async (id: string, sourcePatch: Partial<BrowserSourceConfig>) => {
    commitBrowser(await api.updateBrowserSource(id, sourcePatch))
    flashConnectorStatus('browser', 'Saved')
  }
  const removeBrowserSource = async (id: string) => {
    commitBrowser(await api.removeBrowserSource(id))
    flashConnectorStatus('browser', 'Source removed')
  }
  const syncBrowser = async () => {
    setBrowserBusy('sync')
    try {
      const result = await api.syncBrowser()
      commitBrowser({ ...draft.browser, sources: result.sources, lastSyncAt: result.syncedAt })
      flashConnectorStatus('browser', formatSyncSummary(result.sources.length, result.items.length, 'item'))
    } finally {
      setBrowserBusy('')
    }
  }

  return {
    emailBusy,
    slackBusy,
    documentBusy,
    browserBusy,
    browserTabUrl,
    setBrowserTabUrl,
    browserTabTitle,
    setBrowserTabTitle,
    browserTabExcerpt,
    setBrowserTabExcerpt,
    browserHistoryPath,
    setBrowserHistoryPath,
    connectorStatus,
    flashConnectorStatus,
    addEmailSource,
    updateEmailSource,
    removeEmailSource,
    syncEmail,
    addImapAccount,
    patchImapAccount,
    addSlackSource,
    updateSlackSource,
    removeSlackSource,
    syncSlack,
    addDocumentSource,
    updateDocumentSource,
    removeDocumentSource,
    syncDocuments,
    addBrowserCurrentTabSource,
    addBrowserHistorySource,
    updateBrowserSource,
    removeBrowserSource,
    syncBrowser,
  }
}
