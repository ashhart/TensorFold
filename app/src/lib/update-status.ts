import type { AppUpdateStatus } from './types'

export function updateStatusLabel(status: AppUpdateStatus | null | undefined): string {
  if (!status || status.state === 'not_checked') return 'Updates have not been checked yet'
  if (status.state === 'available') return `Update ${status.latestVersion ?? ''} available`
  if (status.state === 'downloading') return 'Downloading and verifying update'
  if (status.state === 'ready') return `Update ${status.latestVersion ?? ''} ready to install`
  if (status.state === 'installing') return 'Preparing update and restart'
  if (status.state === 'checking') return 'Checking for updates'
  if (status.state === 'error') return 'Update check failed'
  if (status.state === 'not_configured') return 'Update feed not configured'
  return status.checkedAt != null && status.signatureVerified ? 'TensorFold is current' : 'Updates have not been checked yet'
}

export function updateEmptyMessage(status: AppUpdateStatus): string {
  if (status.error) return status.error
  if (status.state === 'available') return 'Release notes for this version were not published; you can download the verified update below.'
  if (status.state === 'downloading') return 'Your current app stays open while the update downloads and verifies.'
  if (status.state === 'ready') return 'The update is verified; finish active work before installing and restarting.'
  if (status.state === 'installing') return 'Saving your workspace and preparing the verified update for restart.'
  if (status.state === 'checking') return 'Checking the release feed for a newer version.'
  if (status.state === 'not_configured') return 'The update feed is not configured for this build.'
  if (status.state === 'current' && status.checkedAt != null && status.signatureVerified) return 'You are on the latest version of TensorFold.'
  return 'No update check has completed yet; check now to find available releases.'
}
