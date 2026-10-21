// Host-neutral wording for bot surfaces: the local target is a Mac only on macOS.
export function localExecutionSource(platform: string): 'This Mac' | 'Local' {
  return platform === 'darwin' ? 'This Mac' : 'Local'
}

export function botVmSupported(platform: string): boolean {
  return platform === 'darwin'
}

/** Trust badge wording for a thread running on a trusted local engine. */
export function threadTrustSource(platform: string): 'On this Mac' | 'Local' {
  return platform === 'darwin' ? 'On this Mac' : 'Local'
}
