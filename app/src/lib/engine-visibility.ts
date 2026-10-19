import type { EngineStatus } from './types'

/**
 * Whether this build can manage a local engine at all. Linux builds ship no engine, so the Models
 * workspace would only advertise a feature that cannot work; macOS always keeps it.
 */
export function engineManagementVisible(platform: string, status: EngineStatus | null | undefined): boolean {
  if (platform === 'darwin') return true
  return status?.available === true
}
