export function applicationBootstrapState(ready: boolean, settings: unknown): 'loading' | 'ready' | 'settings-unavailable' {
  if (!ready) return 'loading'
  return settings == null ? 'settings-unavailable' : 'ready'
}
