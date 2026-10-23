/// What a Google sign-in failure tells the user to check, and how much of Google's own
/// wording is carried into the message. Only static text and bounded ASCII are shown.

/// The longest Google text carried into a message; anything longer is dropped whole.
const MAX_GOOGLE_TEXT = 200

/// What to check for each error Google's authorization endpoint returns.
const ERROR_HELP: Record<string, string> = {
  access_denied: 'Google sign-in was declined. Approve the Gmail and Calendar permissions and try again; if your OAuth project is still in Testing, add this Google account as a test user.',
  invalid_scope: 'This Google project does not allow the permissions the app asked for. Check the OAuth consent screen and the enabled APIs, then try again.',
  unauthorized_client: 'This Google OAuth client is not allowed to sign in. Use a Desktop app client from your own Google Cloud project.',
  invalid_client: 'Google rejected this OAuth client. Download the Desktop app client JSON again and import it.',
  invalid_request: 'Google rejected the sign-in request. Check the Desktop client configuration and the requested permissions, then try again.',
  server_error: "Google's sign-in service failed. Try again in a moment.",
  temporarily_unavailable: "Google's sign-in service is temporarily unavailable. Try again in a moment.",
  interaction_required: 'Google needs another interaction to finish signing in. Try again and complete every prompt.',
  consent_required: 'Google needs your consent again. Sign in again and approve the requested permissions.',
  login_required: 'Google needs you to sign in again. Choose your account in the browser and try again.',
  account_selection_required: 'Google needs you to choose the account to connect. Try again and pick the account.',
}

/// Google's own wording, kept only when it is short printable ASCII with no control characters.
export function boundedGoogleText(value: string | null | undefined): string | undefined {
  if (typeof value !== 'string') return undefined
  const trimmed = value.trim()
  if (!trimmed || trimmed.length > MAX_GOOGLE_TEXT || !/^[\x20-\x7e]+$/.test(trimmed)) return undefined
  return trimmed
}

/// The message the app shows when Google refuses a sign-in. Never carries a code, a
/// verifier, a client secret or a token: only Google's error code and its bounded description.
export function googleOAuthErrorMessage(code: string, description?: string | null): string {
  const known = ERROR_HELP[code]
  const help = known ?? 'Google did not complete the sign-in. Try again; if it keeps failing, check the Desktop client configuration, the enabled APIs and the test-user list.'
  const named = known ? undefined : boundedGoogleText(code)
  const said = boundedGoogleText(description)
  const parts = [help]
  if (named) parts.push(`Google reported: ${named}.`)
  if (said) parts.push(`Google said: ${said}`)
  return parts.join(' ')
}
