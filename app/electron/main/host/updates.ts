import { app, autoUpdater, shell } from 'electron'
import { execFile } from 'node:child_process'
import { constants, promises as fs } from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'
import releaseTrust from '../../../build/release-trust.json'
import { checkUpdateFeed, checkingUpdateStatus, emptyUpdateStatus } from '../../agent/update-checker'
import { resolveUpdateFeedUrl } from '../../agent/update-feed'
import { assertUpdateFeedUrl, auditPreview } from '../../agent/security'
import type { AppUpdateStatus } from '../../agent/types'
import { vontraPath } from '../../store'
import { store } from '../state'
import { auditEvent, requireSecurityRole } from './audit'
import { downloadVerifiedUpdate, reverifyDownloadedUpdate } from '../../agent/update-download'
import { updateTrustRootPublicKey, VONTRA_UPDATE_TRUST_ROOT_ID } from '../../agent/update-trust'
import { assertPinnedMacSignature, installVerifiedUpdate, startVerifiedUpdateFeed, type NativeUpdateDriver, type UpdateInstallSafety } from '../../agent/update-install'
import type { VerifiedUpdateDownloadResult } from '../../agent/update-download'
import { writeJson } from '../../store'
import { handle } from './ipc'
import { privateUpdateRelaunchProfile } from './update-profile'

// App update chain: verify a manifest against the embedded Ed25519 root, then download the
// signed URL in-process through a DNS-pinned redirect loop, hash the actual bytes, and only hand
// the verified LOCAL artifact to the OS. No browser download or environment trust override.

function emitUpdateStatus(s: AppUpdateStatus) { try { store.win?.webContents.send('app:update', s) } catch { /* main-process update state survives renderer teardown */ } }

// Canonical TensorFold release feed. Hardcoded (task #50) so the feed cannot
// be repointed at an arbitrary or hostile manifest via settings or env vars.
export function updateFeedUrl(): string {
  return resolveUpdateFeedUrl()
}

export function updateSignaturePublicKey(): string {
  return updateTrustRootPublicKey()
}

export function signedUpdatesRequired(): boolean {
  return true
}

let updateTimer: ReturnType<typeof setInterval> | null = null
let updateCheckInFlight: Promise<AppUpdateStatus> | null = null
let updateDownloadInFlight: Promise<{ ok: boolean }> | null = null
let downloadController: AbortController | null = null
let installSafety: UpdateInstallSafety | null = null
let readyDownload: VerifiedUpdateDownloadResult | null = null
const exec = promisify(execFile)

export function configureUpdateInstallSafety(safety: UpdateInstallSafety): void { installSafety = safety }

function busy(): boolean { return ['downloading', 'ready', 'installing'].includes(store.updateStatus?.state) }

async function automaticInstallSupport(status: AppUpdateStatus): Promise<NonNullable<AppUpdateStatus['automaticInstall']>> {
  const unavailable = (reason: string) => ({ supported: false, reason })
  if (process.platform !== 'darwin' || !app.isPackaged) return unavailable('Automatic installation is available in the signed macOS app; this build uses the verified installer.')
  if (!status.automaticUpdate) return unavailable('This release provides a verified installer; automatic installation requires its signed ZIP.')
  if (status.automaticUpdate.arch !== 'universal' && status.automaticUpdate.arch !== process.arch) return unavailable('This update ZIP does not match this Mac’s architecture.')
  if (!installSafety) return unavailable('Automatic installation is unavailable until workspace preservation is configured.')
  const bundle = path.resolve(process.execPath, '..', '..', '..')
  if (process.env.VONTRA_HOME && privateUpdateRelaunchProfile(process.execPath)?.root !== process.env.VONTRA_HOME) return unavailable('This app uses a custom data folder; use the verified installer, then reopen TensorFold with the same data folder.')
  if (!bundle.endsWith('.app') || bundle.includes('/AppTranslocation/') || bundle.startsWith('/Volumes/')) return unavailable('Move TensorFold to Applications before installing updates automatically.')
  try {
    await fs.access(bundle, constants.W_OK)
    await fs.access(path.dirname(bundle), constants.W_OK)
    await exec('/usr/bin/codesign', ['--verify', '--deep', '--strict', bundle], { timeout: 30_000 })
    const result = await exec('/usr/bin/codesign', ['--display', '--verbose=4', bundle], { timeout: 30_000 })
    assertPinnedMacSignature(`${result.stdout}\n${result.stderr}`, releaseTrust.macos)
    return { supported: true, reason: null }
  } catch { return unavailable('This app needs the production Developer ID signature and a writable installation folder; use the verified installer.') }
}

async function recoverReadyDownload(status: AppUpdateStatus): Promise<AppUpdateStatus> {
  if (status.state !== 'available' || !status.automaticInstall?.supported || !status.automaticUpdate) return status
  try {
    const raw = await fs.readFile(vontraPath('updates', 'ready.json'), 'utf8')
    if (Buffer.byteLength(raw) > 16_384) return status
    const receipt = JSON.parse(raw)
    // A disk receipt conveys no trust: a fresh root-verified manifest must bind these exact bytes.
    if (receipt.version !== status.latestVersion || receipt.zipUrl !== status.automaticUpdate.zipUrl || receipt.sha256 !== status.automaticUpdate.checksum.value || receipt.download?.sha256 !== receipt.sha256) return status
    readyDownload = await reverifyDownloadedUpdate(receipt.download, vontraPath('updates'))
    return { ...status, state: 'ready', integrity: 'artifact_verified', downloadProgress: { receivedBytes: readyDownload.bytes, totalBytes: readyDownload.bytes } }
  } catch { return status }
}

export function setUpdateStatus(status: AppUpdateStatus): AppUpdateStatus {
  store.updateStatus = status
  emitUpdateStatus(status)
  return status
}

export async function checkForAppUpdate(): Promise<AppUpdateStatus> {
  if (busy()) return store.updateStatus
  const currentVersion = app.getVersion()
  const feedUrl = updateFeedUrl()
  if (!feedUrl) return setUpdateStatus(emptyUpdateStatus(currentVersion, null))
  try { assertUpdateFeedUrl(feedUrl) } catch (e: any) {
    return setUpdateStatus({ ...emptyUpdateStatus(currentVersion, feedUrl), state: 'error', checkedAt: Date.now(), error: String(e?.message ?? e) })
  }
  if (updateCheckInFlight) return updateCheckInFlight
  setUpdateStatus(checkingUpdateStatus(currentVersion, feedUrl))
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 12_000)
  updateCheckInFlight = checkUpdateFeed(feedUrl, currentVersion, controller.signal)
    .then(async (status) => {
      const next = { ...status, automaticInstall: await automaticInstallSupport(status) }
      return setUpdateStatus(await recoverReadyDownload(next))
    })
    .catch((e: any) => setUpdateStatus({
      ...emptyUpdateStatus(currentVersion, feedUrl),
      state: 'error',
      checkedAt: Date.now(),
      error: e?.name === 'AbortError' ? 'Update check timed out' : String(e?.message ?? e),
    }))
    .finally(() => { clearTimeout(timeout); updateCheckInFlight = null }) as Promise<AppUpdateStatus>
  return updateCheckInFlight
}

export function startUpdateChecks() {
  if (updateTimer) { clearInterval(updateTimer); updateTimer = null }
  const feedUrl = updateFeedUrl()
  if (!busy()) setUpdateStatus(emptyUpdateStatus(app.getVersion(), feedUrl || null))
  if (!store.settings?.autoCheckUpdates || !feedUrl) return
  setTimeout(() => { void checkForAppUpdate() }, 1500)
  updateTimer = setInterval(() => { void checkForAppUpdate() }, 6 * 60 * 60_000)
}

export async function openUpdateDownload(): Promise<{ ok: boolean }> {
  requireSecurityRole('operator', 'open update downloads')
  const status = store.updateStatus
  if (!['available', 'ready'].includes(status.state) || !status.downloadUrl) return { ok: false }
  if (!status.checksum) {
    await auditEvent('updates.open_blocked', { reason: 'missing_sha256_checksum', url: auditPreview(status.downloadUrl) })
    return { ok: false }
  }
  if (!status.signatureVerified) {
    await auditEvent('updates.open_blocked', { reason: 'signature_not_verified', url: auditPreview(status.downloadUrl), integrity: status.integrity, trustRoot: VONTRA_UPDATE_TRUST_ROOT_ID })
    return { ok: false }
  }
  if (updateDownloadInFlight) return updateDownloadInFlight

  const controller = new AbortController()
  downloadController = controller
  setUpdateStatus({ ...status, state: 'downloading', downloadProgress: { receivedBytes: 0, totalBytes: null }, error: null })
  const timeout = setTimeout(() => controller.abort(), 15 * 60_000)
  updateDownloadInFlight = (async () => {
    try {
      const downloadRoot = vontraPath('updates')
      const verified = await downloadVerifiedUpdate({
        downloadUrl: status.downloadUrl!,
        checksum: status.checksum!,
        manifestSignatureVerified: status.signatureVerified,
        downloadRoot,
        signal: controller.signal,
        onProgress: (downloadProgress) => setUpdateStatus({ ...store.updateStatus, downloadProgress }),
      })
      setUpdateStatus({ ...status, state: 'available', integrity: 'artifact_verified', downloadProgress: { receivedBytes: verified.bytes, totalBytes: verified.bytes }, error: null })
      await auditEvent('updates.download_verified', {
        url: auditPreview(verified.finalUrl),
        sha256: verified.sha256,
        bytes: verified.bytes,
        redirects: verified.redirects,
        trustRoot: VONTRA_UPDATE_TRUST_ROOT_ID,
      })
      // Re-open, identity-check, and hash the managed private file as the final operation before
      // the OS consumes its path. No awaited work is inserted between this check and openPath.
      const finalArtifact = await reverifyDownloadedUpdate(verified, downloadRoot)
      const openError = await shell.openPath(finalArtifact.path)
      if (openError) {
        setUpdateStatus({ ...store.updateStatus, error: `The verified update could not be opened: ${openError}` })
        await auditEvent('updates.open_failed', { reason: auditPreview(openError), sha256: finalArtifact.sha256 })
        return { ok: false }
      }
      await auditEvent('updates.open_verified', { sha256: finalArtifact.sha256, bytes: finalArtifact.bytes })
      return { ok: true }
    } catch (error: any) {
      const message = error?.name === 'AbortError' ? 'Update download cancelled or timed out.' : String(error?.message ?? error)
      setUpdateStatus({ ...status, state: 'available', integrity: 'artifact_verification_failed', downloadProgress: null, error: message })
      await auditEvent('updates.download_blocked', { reason: auditPreview(message), url: auditPreview(status.downloadUrl), trustRoot: VONTRA_UPDATE_TRUST_ROOT_ID })
      return { ok: false }
    } finally {
      clearTimeout(timeout)
      updateDownloadInFlight = null
      downloadController = null
    }
  })()
  return updateDownloadInFlight
}

export async function downloadAppUpdate(): Promise<{ ok: boolean }> {
  requireSecurityRole('operator', 'download an app update')
  const status = store.updateStatus
  if (status.state !== 'available' || !status.signatureVerified) return { ok: false }
  const support = await automaticInstallSupport(status)
  if (!support.supported || !status.automaticUpdate) return openUpdateDownload()
  if (updateDownloadInFlight) return updateDownloadInFlight
  const controller = new AbortController()
  downloadController = controller
  const timeout = setTimeout(() => controller.abort(), 15 * 60_000)
  setUpdateStatus({ ...status, automaticInstall: support, state: 'downloading', downloadProgress: { receivedBytes: 0, totalBytes: null }, error: null })
  updateDownloadInFlight = (async () => {
    try {
      const download = await downloadVerifiedUpdate({
        downloadUrl: status.automaticUpdate!.zipUrl, checksum: status.automaticUpdate!.checksum,
        manifestSignatureVerified: status.signatureVerified, downloadRoot: vontraPath('updates'), signal: controller.signal,
        onProgress: (downloadProgress) => setUpdateStatus({ ...store.updateStatus, downloadProgress }),
      })
      readyDownload = await reverifyDownloadedUpdate(download, vontraPath('updates'))
      await writeJson(vontraPath('updates', 'ready.json'), { version: status.latestVersion, zipUrl: status.automaticUpdate!.zipUrl, sha256: download.sha256, download })
      setUpdateStatus({ ...status, automaticInstall: support, state: 'ready', integrity: 'artifact_verified', downloadProgress: { receivedBytes: download.bytes, totalBytes: download.bytes }, error: null })
      await auditEvent('updates.ready', { version: status.latestVersion, sha256: download.sha256, bytes: download.bytes })
      return { ok: true }
    } catch (error: any) {
      readyDownload = null
      setUpdateStatus({ ...status, automaticInstall: support, state: 'available', downloadProgress: null, error: controller.signal.aborted ? 'Download cancelled; your current app is unchanged.' : String(error?.message ?? error) })
      return { ok: false }
    } finally { clearTimeout(timeout); updateDownloadInFlight = null; downloadController = null }
  })()
  return updateDownloadInFlight
}

export function cancelAppUpdateDownload(): { cancelled: boolean } {
  requireSecurityRole('operator', 'cancel an app update download')
  if (!downloadController) return { cancelled: false }
  downloadController.abort()
  return { cancelled: true }
}

export async function installAppUpdate(): Promise<{ ok: boolean }> {
  requireSecurityRole('operator', 'install an app update and restart')
  const status = store.updateStatus
  if (status.state !== 'ready' || !status.signatureVerified || !readyDownload || !installSafety) return { ok: false }
  const support = await automaticInstallSupport(status)
  if (!support.supported) { setUpdateStatus({ ...status, automaticInstall: support, error: support.reason }); return { ok: false } }
  if (store.updateStatus.state !== 'ready') return { ok: false }
  try {
    installSafety.assertIdle()
    setUpdateStatus({ ...status, state: 'installing', error: null })
    await auditEvent('updates.install_requested', { version: status.latestVersion, sha256: readyDownload.sha256 })
    await installVerifiedUpdate({
      driver: autoUpdater as NativeUpdateDriver, safety: installSafety,
      openFeed: () => startVerifiedUpdateFeed({ download: readyDownload!, downloadRoot: vontraPath('updates'), version: status.latestVersion!, releaseName: status.releaseName, releaseNotes: status.releaseNotes }),
    })
    // quitAndInstall is a handoff, never proof that a replacement or relaunch succeeded.
    return { ok: true }
  } catch (error: any) {
    setUpdateStatus({ ...status, state: 'ready', error: String(error?.message ?? error) })
    return { ok: false }
  }
}

export function registerUpdateInstallIpc(): void {
  // Native work may finish after a stage timeout; absorb late errors without logging feed tokens.
  autoUpdater.on('error', () => {
    if (store.updateStatus?.state === 'installing') setUpdateStatus({ ...store.updateStatus, state: 'ready', error: 'macOS could not finish preparing the update; your app remains open.' })
  })
  handle('updates:download', () => downloadAppUpdate())
  handle('updates:cancel', () => cancelAppUpdateDownload())
  handle('updates:install', () => installAppUpdate())
}
