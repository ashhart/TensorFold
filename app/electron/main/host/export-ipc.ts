import { dialog } from 'electron'
import { promises as fs } from 'node:fs'
import { buildRunExport } from '../../agent/export'
import { auditPreview } from '../../agent/security'
import type { ExportOptions, ExportSaveResult, RunExportBundle } from '../../agent/types'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { handle } from './ipc'

function buildThreadExport(threadId: string, opts: ExportOptions): RunExportBundle {
  const thread = store.threads.find((t) => t.id === threadId)
  if (!thread) throw new Error('Thread not found.')
  return buildRunExport(thread, opts)
}

export function registerExportIpc(): void {
  handle('export:build', async (_e, threadId: string, opts: ExportOptions = {}) => {
    requireSecurityRole('operator', 'build run exports')
    const bundle = buildThreadExport(threadId, opts)
    return { bundle, json: JSON.stringify(bundle, null, 2) }
  })

  handle('export:save', async (_e, threadId: string, opts: ExportOptions = {}): Promise<ExportSaveResult> => {
    requireSecurityRole('operator', 'save run exports')
    const bundle = buildThreadExport(threadId, opts)
    const json = JSON.stringify(bundle, null, 2)
    const suffix = bundle.mode === 'share' ? 'share' : bundle.mode === 'eval' ? 'eval' : 'run'
    const base = (bundle.thread.title || 'run').replace(/[^a-z0-9-_]+/gi, '-').replace(/^-+|-+$/g, '').slice(0, 40) || 'run'
    const dialogOpts = {
      title: 'Export run',
      defaultPath: `vontra-${base}-${suffix}.json`,
      filters: [{ name: 'JSON', extensions: ['json'] }],
    }
    const res = store.win ? await dialog.showSaveDialog(store.win, dialogOpts) : await dialog.showSaveDialog(dialogOpts)
    if (res.canceled || !res.filePath) return { saved: false, canceled: true }
    await fs.writeFile(res.filePath, json, 'utf8')
    await auditEvent('export.save', { threadId, mode: bundle.mode, redacted: bundle.redacted, anonymized: bundle.anonymized, redactions: bundle.redactionReport.reduce((n, r) => n + r.count, 0), path: auditPreview(res.filePath) })
    return { saved: true, path: res.filePath, bytes: Buffer.byteLength(json, 'utf8') }
  })

  handle('export:importBundle', async () => {
    requireSecurityRole('operator', 'import run bundles')
    const dialogOpts = { title: 'Import run bundle', filters: [{ name: 'JSON', extensions: ['json'] }], properties: ['openFile' as const] }
    const res = store.win ? await dialog.showOpenDialog(store.win, dialogOpts) : await dialog.showOpenDialog(dialogOpts)
    if (res.canceled || !res.filePaths[0]) return { bundle: null }
    let parsed: any
    try { parsed = JSON.parse(await fs.readFile(res.filePaths[0], 'utf8')) } catch { return { bundle: null, error: 'That file is not valid JSON.' } }
    const ok = !!parsed && parsed.schema === 'vontra.run-export/1'
      && Array.isArray(parsed.messages) && parsed.messages.every((m: any) => m && typeof m.role === 'string' && typeof m.content === 'string')
      && !!parsed.thread && typeof parsed.thread.title === 'string'
      && !!parsed.stats && typeof parsed.stats.messages === 'number' && typeof parsed.stats.toolCalls === 'number'
    if (!ok) return { bundle: null, error: 'Not a valid TensorFold run-export bundle.' }
    return { bundle: parsed as RunExportBundle }
  })
}
