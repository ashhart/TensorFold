// Host-neutral wording for bot surfaces: the local execution target is a Mac only on macOS,
// and bot VM isolation exists only there.
export function localExecutionSource(platform: string): 'This Mac' | 'Local' {
  return platform === 'darwin' ? 'This Mac' : 'Local'
}

export function botVmSupported(platform: string): boolean {
  return platform === 'darwin'
}
