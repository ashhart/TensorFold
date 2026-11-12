export type Route = 'chat' | 'models' | 'teach' | 'connect' | 'bench' | 'monitor' | 'fleet' | 'settings'
export const ROUTES: Route[] = ['chat', 'models', 'teach', 'connect', 'bench', 'monitor', 'fleet', 'settings']
export const SHORTCUTS: Route[] = ['chat', 'models', 'chat', 'teach', 'connect', 'bench', 'monitor', 'fleet', 'settings']
export const TITLES: Record<Route, string> = { chat: 'New chat', models: 'Models', teach: 'Learning', connect: 'API', bench: 'Benchmarks', monitor: 'Engine details', fleet: 'Compute and clusters', settings: 'Settings' }
export function routeOf(hash: string): Route {
  const screen = hash.replace(/^#/, '').split('/')[0] as Route
  return ROUTES.includes(screen) ? screen : 'chat'
}
export function modelTab(hash: string): 'mine' | 'discover' | 'learning' | 'api' {
  if (hash.startsWith('#teach')) return 'learning'
  if (hash.startsWith('#connect')) return 'api'
  const tab = hash.split('/')[1]
  return tab === 'discover' || tab === 'learning' || tab === 'api' ? tab : 'mine'
}

export function loopbackEngine(address: string | undefined): boolean {
  try { const host = new URL(address ?? '').hostname; return host === 'localhost' || host === '[::1]' || /^127\./.test(host) } catch { return false }
}
