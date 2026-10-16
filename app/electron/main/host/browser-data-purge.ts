// Electron-free lifecycle helper for clearing the persistent browser-automation partition.
// Kept structural/injected so the destructive sequence is testable without an Electron runtime.

export interface PurgeableBrowserSession {
  flushStorageData?(): void | Promise<void>
  clearStorageData(): Promise<void>
  clearCache(): Promise<void>
  clearAuthCache(): Promise<void>
  clearCodeCaches?(options: Record<string, never>): Promise<void>
  clearHostResolverCache?(): Promise<void>
}

export async function clearPersistentBrowserData(session: PurgeableBrowserSession): Promise<void> {
  // Commit pending cookie/storage mutations before clearing, then flush the deletion. This keeps
  // a renderer process that was closing at purge time from replaying a late storage write.
  await session.flushStorageData?.()
  await Promise.all([
    session.clearStorageData(),
    session.clearCache(),
    session.clearAuthCache(),
    session.clearCodeCaches?.({}) ?? Promise.resolve(),
    session.clearHostResolverCache?.() ?? Promise.resolve(),
  ])
  await session.flushStorageData?.()
}
