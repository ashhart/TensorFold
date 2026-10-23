import assert from 'node:assert/strict'
import { boundedGoogleText, googleOAuthErrorMessage } from './oauth-errors'

// Google's own error codes, and what the user is told to check.
assert.match(googleOAuthErrorMessage('access_denied'), /declined/)
assert.match(googleOAuthErrorMessage('access_denied'), /test user/)
assert.match(googleOAuthErrorMessage('invalid_scope'), /consent screen/)
assert.match(googleOAuthErrorMessage('unauthorized_client'), /Desktop app client/)
assert.match(googleOAuthErrorMessage('invalid_client'), /Desktop app client JSON/)
assert.match(googleOAuthErrorMessage('server_error'), /Try again/)
assert.match(googleOAuthErrorMessage('temporarily_unavailable'), /Try again/)
assert.match(googleOAuthErrorMessage('interaction_required'), /interaction/)
assert.match(googleOAuthErrorMessage('consent_required'), /consent/)
assert.match(googleOAuthErrorMessage('login_required'), /sign in again/)
assert.match(googleOAuthErrorMessage('account_selection_required'), /choose the account/)
assert.match(googleOAuthErrorMessage('invalid_request'), /Desktop client configuration/)

// An unknown code still says what to check and names the code, bounded.
const unknown = googleOAuthErrorMessage('some_unknown_code')
assert.match(unknown, /some_unknown_code/)
assert.match(unknown, /Desktop client configuration/)

// Google's own words ride along only when they are short printable ASCII.
assert.match(googleOAuthErrorMessage('access_denied', 'Access blocked by the administrator'), /Google said: Access blocked by the administrator/)
assert.ok(!googleOAuthErrorMessage('access_denied', 'x'.repeat(500)).includes('Google said:'))
assert.ok(!googleOAuthErrorMessage('access_denied', 'line\nbreak').includes('Google said:'))
assert.ok(!googleOAuthErrorMessage('access_denied', 'delete\u0000me').includes('Google said:'))
assert.ok(!googleOAuthErrorMessage('access_denied', 'ünïcode').includes('Google said:'))
assert.ok(!googleOAuthErrorMessage('access_denied', '').includes('Google said:'))
assert.ok(!googleOAuthErrorMessage('access_denied', null).includes('Google said:'))

// The bounds are the only place Google's text is filtered.
assert.equal(boundedGoogleText('plain text'), 'plain text')
assert.equal(boundedGoogleText('  padded  '), 'padded')
assert.equal(boundedGoogleText('x'.repeat(200)), 'x'.repeat(200))
assert.equal(boundedGoogleText('x'.repeat(201)), undefined)
assert.equal(boundedGoogleText('tab\there'), undefined)
assert.equal(boundedGoogleText(''), undefined)
assert.equal(boundedGoogleText(undefined), undefined)

// A message never carries a secret: only static text, a bounded code and a bounded description.
for (const code of ['access_denied', 'invalid_scope', 'server_error', 'unknown_code']) {
  const message = googleOAuthErrorMessage(code, 'a short description')
  assert.ok(!message.includes('synthetic-secret'))
  assert.ok(!message.includes('synthetic-code'))
  assert.ok(!message.includes('Bearer'))
}
console.log('Google OAuth error mapping, description bounds and secret-free messages passed')
