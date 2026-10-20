// Host-neutral wording for bot surfaces: the local target is a Mac only on macOS.
export function localExecutionSource(platform: string): 'This Mac' | 'Local' {
  return platform === 'darwin' ? 'This Mac' : 'Local'
}

export function botVmSupported(platform: string): boolean {
  return platform === 'darwin'
}

/** Trust badge: a network engine is not "on this Mac", even though it is trusted like one. */
export function threadTrustSource(platform: string, trustedNetwork: boolean): 'On this Mac' | 'Network' | 'Local' {
  if (trustedNetwork) return 'Network'
  return platform === 'darwin' ? 'On this Mac' : 'Local'
}
