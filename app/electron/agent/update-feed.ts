// Canonical TensorFold release feed.
//
// Hardcoded (task #50) so the auto-update feed cannot be repointed at an
// arbitrary or hostile manifest via user settings (`settings.updateFeedUrl`)
// or environment variables. `resolveUpdateFeedUrl()` is the single source of
// truth — it deliberately ignores any caller-supplied overrides.

export const VONTRA_UPDATE_FEED_URL = 'https://tensorfold.dev/studio/update.json'

export function resolveUpdateFeedUrl(): string {
  return VONTRA_UPDATE_FEED_URL
}
