import { dialog } from 'electron'
import os from 'node:os'
import type { ReadableExportRequest, ReadableExportSaveRequest } from '../../../shared/readable-export'
import { createReadableExportService } from '../../readable-export/service'
import { atomicExport } from '../../store/atomic-export'
import { vontraHomeDir } from '../../store'
import { store } from '../state'
import { handle } from './ipc'
import { requireSecurityRole } from './audit'

export function registerReadableExportRuntime(options: { reserveOperation(): { release(): void } }): void {
  const service = createReadableExportService({
    readThread: id => store.threads.find(thread => thread.id === id && !thread.orchestratorHostId),
    scopeKey: vontraHomeDir,
    redactionContext: thread => ({ workspace: thread.cwd || null, homeDir: os.homedir(), username: os.userInfo().username, hostname: os.hostname(), terms: store.settings.confidentialTerms }),
  })
  handle('readableExport:preview', (event, request: ReadableExportRequest) => {
    requireSecurityRole('owner', 'preview a readable conversation export')
    return service.preview(request, String(event.sender.id))
  })
  handle('readableExport:save', async (event, request: ReadableExportSaveRequest) => {
    requireSecurityRole('owner', 'save a readable conversation export')
    const lease = options.reserveOperation(), owner = String(event.sender.id)
    let claim: ReturnType<typeof service.claim> | null = null
    try {
      claim = service.claim(request, owner)
      const extension = claim.format === 'markdown' ? 'md' : claim.format === 'text' ? 'txt' : 'html'
      const chosen = await dialog.showSaveDialog({ title: 'Save readable conversation', defaultPath: claim.suggestedName, filters: [{ name: 'Readable conversation', extensions: [extension] }] })
      if (chosen.canceled || !chosen.filePath) { service.cancel(claim.claimId); return { saved: false, canceled: true } }
      const packet = service.finalize(claim.claimId, owner)
      await atomicExport(chosen.filePath, Buffer.from(packet.body, 'utf8'), 8_000_000, () => {
        requireSecurityRole('owner', 'save a readable conversation export'); packet.assertCurrent()
      })
      return { saved: true, bytes: packet.bytes }
    } finally { if (claim) service.cancel(claim.claimId); lease.release() }
  })
}
