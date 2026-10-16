/**
 * Deny a pending connector draft, then refresh only after the denial is known to
 * have succeeded. Keeping this ordering in one testable helper prevents the UI
 * from hiding a still-pending/sendable draft after a failed backend denial.
 */
export async function discardPendingDraft(
  resolve: (id: string, decision: 'deny') => Promise<unknown>,
  refresh: () => Promise<unknown>,
  eventId: string,
): Promise<void> {
  await resolve(eventId, 'deny')
  await refresh()
}
