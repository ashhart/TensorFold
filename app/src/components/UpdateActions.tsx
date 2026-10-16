import { useStore } from '../lib/store'
import { useState } from 'react'
import { rendererUpdateBlockReason } from '../lib/update-safety'
import type { AppUpdateStatus } from '../lib/types'
import { Button } from './ui'

export function UpdateActions({ status }: { status: AppUpdateStatus | null }) {
  const download = useStore((s) => s.downloadAppUpdate)
  const install = useStore((s) => s.installAppUpdate)
  const cancel = useStore((s) => s.cancelAppUpdateDownload)
  const check = useStore((s) => s.checkForUpdates)
  const openInstaller = useStore((s) => s.openUpdateDownload)
  const installPending = useStore((s) => s.updateInstallPending)
  const [blocked, setBlocked] = useState<string | null>(null)
  const prepareInstall = async () => {
    if (useStore.getState().updateInstallPending || useStore.getState().updateStatus?.state === 'installing') return
    const reason = rendererUpdateBlockReason()
    setBlocked(reason)
    if (!reason) {
      try { await install() }
      catch (cause) { setBlocked(cause instanceof Error ? cause.message : 'The update could not be prepared. Your app remains open.') }
    }
  }
  const progress = status?.downloadProgress
  const total = progress?.totalBytes
  const percent = total && progress ? Math.min(100, Math.floor(progress.receivedBytes / total * 100)) : null
  const downloading = status?.state === 'downloading'
  const ready = status?.state === 'ready'
  const installing = installPending || status?.state === 'installing'
  return (
    <div className="min-w-0 space-y-2">
      {downloading && <div role="status" aria-live="polite" className="text-[13px] text-ink-2">
        <div className="mb-1 flex justify-between gap-2"><span>Downloading and verifying</span><span>{percent == null ? `${Math.floor((progress?.receivedBytes ?? 0) / 1024 ** 2)} MB` : `${percent}%`}</span></div>
        <progress aria-label="Update download" max={100} value={percent ?? undefined} className="h-1.5 w-full accent-[var(--ink)]" />
      </div>}
      {ready && <p className="text-[13px] leading-relaxed text-ink-2">Your update is verified; finish active work, then install and restart. Restart disconnects local API clients; copy the new API connection after TensorFold reopens.</p>}
      {blocked && <p role="alert" className="text-[13px] leading-relaxed text-danger">{blocked}</p>}
      {status?.state === 'available' && !status.automaticInstall?.supported && <p className="text-[12px] leading-relaxed text-ink-2">{status.automaticInstall?.reason ?? 'Use the verified installer for this release.'} After it opens, replace the app in Applications.</p>}
      <div className="flex flex-wrap justify-end gap-2">
        {installing ? <Button variant="default" size="sm" disabled>Saving and preparing restart…</Button>
          : downloading ? <Button variant="ghost" size="sm" onClick={() => void cancel()}>Cancel download</Button>
          : ready ? status?.automaticInstall?.supported ? <Button variant="default" size="sm" onClick={() => void prepareInstall()}>Install and restart</Button> : <Button variant="default" size="sm" onClick={() => void openInstaller()}>Open verified installer</Button>
          : status?.state === 'available' ? <Button variant="default" size="sm" icon="arrowDown" onClick={() => void download()}>{status.automaticInstall?.supported ? 'Download update' : 'Download, verify and open'}</Button>
          : <Button variant="default" size="sm" icon="refresh" onClick={() => void check()} disabled={status?.state === 'checking'}>{status?.state === 'checking' ? 'Checking…' : 'Check for updates'}</Button>}
      </div>
    </div>
  )
}
