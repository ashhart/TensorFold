import { shell } from 'electron'
import { isSafeExternalUrl } from '../../agent/security'

// Single choke point for opening a URL in the OS browser. Only https/mailto (and,
// opt-in, http loopback) pass isSafeExternalUrl — blocking file:, javascript:, and
// arbitrary http so a model- or feed-supplied URL cannot be weaponised.
export async function openExternalSafely(raw: string, opts: { allowHttpLocalhost?: boolean } = {}) {
  if (!isSafeExternalUrl(raw, opts)) throw new Error('Blocked unsafe external URL.')
  await shell.openExternal(raw)
}
