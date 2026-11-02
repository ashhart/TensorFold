export const GOOGLE_TOKEN_FAILURES = {
  invalid_client: 'Google rejected the Desktop OAuth client. Check the Google Cloud client and import a valid Desktop app client JSON before reconnecting.',
  invalid_grant: 'Google sign-in expired or was revoked. Reconnect this account. Google projects in Testing can require renewed consent.',
  invalid_scope: 'Google rejected the requested permissions. Check the OAuth consent configuration and enabled APIs, then reconnect with the required access.',
  unauthorized_client: 'This Desktop OAuth client is not authorized. Check its Google Cloud configuration before reconnecting.',
} as const
export type GoogleTokenFailure = keyof typeof GOOGLE_TOKEN_FAILURES

const SAFE_FAILURES = new Set<string>([
  ...Object.values(GOOGLE_TOKEN_FAILURES),
  'Google sign-in expired. Reconnect this account.',
  'Google denied this operation. Check the granted permissions and enabled API.',
  'Google is rate limiting this connection. Try again later.',
  'Google could not complete this request.',
  'Google rejected this request.',
  'Google request was cancelled or timed out.',
  'Could not reach Google. Check your connection.',
  'Google account is disconnected. Reconnect it in Connections.',
  'Google account changed. Review this operation again.',
  'Google sign-in must be renewed. Reconnect this account.',
])

// IPC may wrap a host error; arbitrary exception text and server descriptions stay private.
export function googleRequestError(reason: unknown): string {
  const message = (reason instanceof Error ? reason.message : '')
    .replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, '')
  if (SAFE_FAILURES.has(message)) return message
  if (/^Google permission (gmail\.(read|send)|calendar\.(read|write)) is not granted\. Reconnect with that permission selected\.$/.test(message)) return message
  return 'Google could not complete this request; check the account permissions and connection, then retry.'
}
