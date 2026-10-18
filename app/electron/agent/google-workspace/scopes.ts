import type { GoogleWorkspaceCapability } from '../../../shared/google-workspace'

const ROOT = 'https://www.googleapis.com/auth/'
export const GOOGLE_IDENTITY_SCOPES = ['openid', `${ROOT}userinfo.email`] as const
export const GOOGLE_CAPABILITY_SCOPES: Record<GoogleWorkspaceCapability, readonly string[]> = {
  'gmail.read': [`${ROOT}gmail.readonly`],
  'gmail.send': [`${ROOT}gmail.send`],
  'calendar.read': [`${ROOT}calendar.calendarlist.readonly`, `${ROOT}calendar.events.readonly`, `${ROOT}calendar.events.freebusy`],
  'calendar.write': [`${ROOT}calendar.events`],
}
export function validateCapabilities(input: unknown): GoogleWorkspaceCapability[] {
  if (!Array.isArray(input) || !input.length || input.length > 4 || input.some(value => typeof value !== 'string' || !Object.hasOwn(GOOGLE_CAPABILITY_SCOPES, value))) throw new Error('Choose Gmail or Calendar permissions.')
  return [...new Set(input)] as GoogleWorkspaceCapability[]
}
export function requestedScopes(capabilities: GoogleWorkspaceCapability[]): string[] {
  return [...new Set([...GOOGLE_IDENTITY_SCOPES, ...validateCapabilities(capabilities).flatMap(capability => GOOGLE_CAPABILITY_SCOPES[capability])])]
}
export function grantedCapabilities(scopes: string[]): GoogleWorkspaceCapability[] {
  const grants = new Set(scopes)
  return (Object.keys(GOOGLE_CAPABILITY_SCOPES) as GoogleWorkspaceCapability[]).filter(capability => GOOGLE_CAPABILITY_SCOPES[capability].every(scope => grants.has(scope)))
}
export function assertCapability(capabilities: GoogleWorkspaceCapability[], required: GoogleWorkspaceCapability): void {
  if (!capabilities.includes(required)) throw new Error(`Google permission ${required} is not granted. Reconnect with that permission selected.`)
}
