export interface PhonePreviewRect { left: number; top: number; width: number; height: number }

// object-contain adds letterboxing. Ignore clicks there rather than turning them
// into edge taps on an unrelated control on the phone.
export function phonePreviewPoint(clientX: number, clientY: number, rect: PhonePreviewRect, width: number, height: number): { x: number; y: number } | null {
  if (![clientX, clientY, rect.left, rect.top, rect.width, rect.height, width, height].every(Number.isFinite) || rect.width <= 0 || rect.height <= 0 || width <= 0 || height <= 0) return null
  const scale = Math.min(rect.width / width, rect.height / height)
  const left = rect.left + (rect.width - width * scale) / 2
  const top = rect.top + (rect.height - height * scale) / 2
  const x = (clientX - left) / scale
  const y = (clientY - top) / scale
  if (x < 0 || y < 0 || x >= width || y >= height) return null
  return { x: Math.floor(x), y: Math.floor(y) }
}

export function phoneObservationAge(capturedAt: number | null | undefined, now = Date.now()): string {
  if (!capturedAt || !Number.isFinite(capturedAt)) return 'No screen captured'
  const seconds = Math.max(0, Math.floor((now - capturedAt) / 1000))
  if (seconds < 5) return 'Captured just now'
  if (seconds < 60) return `Captured ${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  return `Captured ${minutes}m ago`
}

export function phoneError(error: unknown): string {
  return String((error as Error)?.message ?? error ?? 'Phone control is unavailable.').replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, '')
}

export function phoneChatGranted(session: { threadId: string | null; grantExpiresAt?: number | null } | null | undefined, threadId: string | null | undefined, now = Date.now()): boolean {
  return !!threadId && session?.threadId === threadId && (session.grantExpiresAt == null || Number.isFinite(session.grantExpiresAt) && session.grantExpiresAt > now)
}
