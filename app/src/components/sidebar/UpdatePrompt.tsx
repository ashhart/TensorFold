import { useRef, useState } from 'react'
import { useStore } from '../../lib/store'
import { rendererUpdateBlockReason } from '../../lib/update-safety'
import type { AppUpdateStatus } from '../../lib/types'
import { Button } from '../ui'

// The host owns semantic-version ordering and manifest verification. Do not turn
// unchecked, unsigned or error-only statuses into a new-version announcement.
export function actionableSidebarUpdate(status: AppUpdateStatus | null): AppUpdateStatus | null {
  if (!status?.signatureVerified || !['available', 'downloading', 'ready', 'installing'].includes(status.state)) return null
  const latest = status.latestVersion?.trim().replace(/^v/, '')
  const current = status.currentVersion.trim().replace(/^v/, '')
  if (!latest || !current || latest === current) return null
  if (['ready', 'installing'].includes(status.state) && status.integrity !== 'artifact_verified') return null
  return status
}

export function sidebarUpdateProgress(status: AppUpdateStatus): number | null {
  const progress = status.downloadProgress
  if (!progress || !Number.isFinite(progress.totalBytes) || !progress.totalBytes || progress.totalBytes < 0 || !Number.isFinite(progress.receivedBytes)) return null
  return Math.max(0, Math.min(100, Math.floor(progress.receivedBytes / progress.totalBytes * 100)))
}

type Action = 'download' | 'install' | 'open'

export function UpdatePrompt() {
  const incoming = useStore((s) => s.updateStatus)
  const status = actionableSidebarUpdate(incoming)
  const actionRef = useRef<Action | null>(null)
  const cancelRef = useRef(false)
  const [pending, setPending] = useState<Action | null>(null)
  const [cancelling, setCancelling] = useState(false)
  const [localError, setLocalError] = useState<{ version: string; message: string } | null>(null)
  if (!status) return null

  const version = status.latestVersion!.trim()
  const versionLabel = version.startsWith('v') ? version : `v${version}`
  const downloading = status.state === 'downloading'
  const ready = status.state === 'ready'
  const installing = status.state === 'installing'
  const manual = !status.automaticInstall?.supported
  const percent = sidebarUpdateProgress(status)
  const error = (localError?.version === version ? localError.message : null) ?? status.error
  const action = async (kind: Action) => {
    if (actionRef.current) return
    const state = useStore.getState()
    const current = actionableSidebarUpdate(state.updateStatus)
    if (!current || current.latestVersion?.trim() !== version) return
    if (kind === 'download' ? current.state !== 'available' : current.state !== 'ready') return
    if (kind === 'install' && !current.automaticInstall?.supported) return
    if (kind === 'install') {
      const reason = rendererUpdateBlockReason()
      if (reason) { setLocalError({ version, message: reason }); return }
    }
    actionRef.current = kind
    setPending(kind)
    setLocalError(null)
    try {
      if (kind === 'download') await state.downloadAppUpdate()
      else if (kind === 'install') await state.installAppUpdate()
      else await state.openUpdateDownload()
    } catch (cause) {
      setLocalError({ version, message: cause instanceof Error ? cause.message : 'The update could not be completed. Open Details for help.' })
    } finally { actionRef.current = null; setPending(null) }
  }
  const cancel = async () => {
    if (cancelRef.current || useStore.getState().updateStatus?.state !== 'downloading') return
    cancelRef.current = true
    setCancelling(true)
    setLocalError(null)
    try { await useStore.getState().cancelAppUpdateDownload() }
    catch (cause) { setLocalError({ version, message: cause instanceof Error ? cause.message : 'The download could not be cancelled. Open Details for help.' }) }
    finally { cancelRef.current = false; setCancelling(false) }
  }
  const title = downloading ? percent === 100 ? 'Verifying update' : 'Downloading update'
    : ready ? 'Update ready' : installing ? 'Preparing restart' : 'New version available'

  return (
    <section aria-label="TensorFold update" className="no-drag mb-2 min-w-0 rounded-xl border border-line bg-surface p-2.5">
      <div className="min-w-0">
        <p role="status" aria-live="polite" className="text-[13px] font-medium leading-5 text-ink">{title}</p>
        <p className="break-words text-[12px] leading-5 text-ink-2">{versionLabel}</p>
      </div>
      {downloading && <div className="mt-2">
        <progress aria-label="Update download" max={100} value={percent ?? undefined} className="block h-1.5 w-full accent-[var(--accent)]" />
        <p className="mt-1 text-[12px] leading-4 text-ink-2">{percent == null ? 'Downloading and verifying…' : `${percent}%${percent === 100 ? ' · Checking files' : ''}`}</p>
      </div>}
      {ready && <p className="mt-1 text-[12px] leading-[18px] text-ink-2">{manual ? 'Open the verified installer to replace the app in Applications.' : 'Saves your workspace, then restarts the app. Restart disconnects local API clients.'}</p>}
      {status.state === 'available' && manual && <p className="mt-1 text-[12px] leading-[18px] text-ink-2">Downloads, verifies and opens the installer.</p>}
      {error && <p role="alert" className="mt-2 break-words text-[12px] leading-[18px] text-danger">{error}</p>}
      <div className="mt-2 flex flex-wrap items-center justify-between gap-1.5">
        <button type="button" onClick={() => useStore.getState().openUpdateModal()} className="rounded-md px-1 py-1.5 text-[12px] text-ink-2 underline-offset-4 hover:text-ink hover:underline focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-info">Details</button>
        {downloading ? <Button variant="ghost" size="xs" disabled={cancelling} onClick={() => void cancel()}>{cancelling ? 'Cancelling…' : 'Cancel'}</Button>
          : installing ? <Button variant="default" size="xs" disabled>Restarting…</Button>
          : ready ? <Button variant="default" size="xs" disabled={pending !== null} onClick={() => void action(manual ? 'open' : 'install')}>{pending ? 'Preparing…' : manual ? 'Open installer' : 'Restart to update'}</Button>
          : <Button variant="default" size="xs" disabled={pending !== null} onClick={() => void action('download')}>{pending ? 'Starting…' : 'Update'}</Button>}
      </div>
    </section>
  )
}
