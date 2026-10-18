import { HostError, type MobileHost } from './client'
import { createWatchRelay as createSharedWatchRelay } from '../../../shared/watch-relay'
export { WATCH_PROTOCOL, WATCH_MAX_AGE_MS, WATCH_REPLY_LIMIT, parseWatchCommand, watchSnapshot } from '../../../shared/watch-relay'
export type { WatchJob, WatchSnapshot, WatchCommand, WatchResult } from '../../../shared/watch-relay'

/** Foreground UI compatibility facade; background watch traffic uses the authenticated Mac command route. */
export function createWatchRelay(deps: Parameters<typeof createSharedWatchRelay<MobileHost>>[0]) {
  return createSharedWatchRelay({...deps, safeError: error => error instanceof HostError ? error.message : null})
}
