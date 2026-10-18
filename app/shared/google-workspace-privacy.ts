import { isExternalProvider, type ProviderTrustInput } from './provider-trust'

export interface GoogleContextThread {
  googleLocalOnly?: boolean
  orchestratorHostId?: string | null
  engine?: string
  workspace?: { kind: string } | null
}
/** A host-owned, monotonic marker follows Google-derived history into every retained branch. */
export function assertGoogleContextTransport(thread: GoogleContextThread | null | undefined, provider: ProviderTrustInput | null | undefined): void {
  if (!thread?.googleLocalOnly) return
  if (!provider || isExternalProvider(provider) || thread.orchestratorHostId || thread.engine && thread.engine !== 'native' || thread.workspace?.kind === 'ssh') throw new Error('This conversation contains private Google account data and can only use an API model running on this Mac. Start a separate chat for external, LAN, CLI or remote models.')
}
export function googleContextLocalOnly(thread: GoogleContextThread | null | undefined): boolean { return thread?.googleLocalOnly === true }
