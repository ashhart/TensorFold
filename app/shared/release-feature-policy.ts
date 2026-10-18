// Release decisions are compiled into the desktop app, never read from preferences or env.
// The bundled engine/standalone CLI still has its own capabilities.
export const CORTHEON_RELEASE_POLICY = Object.freeze({
  enabled: false,
  reason: 'Cortheon is temporarily disabled in this beta.',
})

export function cortheonEnabledForRelease(requested: unknown): boolean {
  return CORTHEON_RELEASE_POLICY.enabled && requested === true
}

export function assertCortheonReleaseAllowed(enabled: unknown): asserts enabled is boolean {
  if (typeof enabled !== 'boolean') throw new Error('Cortheon mode must be on or off.')
  if (enabled && !CORTHEON_RELEASE_POLICY.enabled) throw new Error(CORTHEON_RELEASE_POLICY.reason)
}
