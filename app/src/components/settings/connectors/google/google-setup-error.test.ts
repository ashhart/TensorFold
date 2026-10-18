import assert from 'node:assert/strict'
import { googleClientImportError } from './google-setup-error'
assert.match(googleClientImportError(new Error('Google OAuth client must be a Desktop app, not a web client or service account.')), /Desktop app/)
assert.match(googleClientImportError(new Error('Error invoking remote method \'googleWorkspace:importClient\': Error: Google desktop client JSON must include its client secret.')), /complete Desktop app client JSON/)
assert.match(googleClientImportError(new Error('Disconnect Google accounts before changing the desktop client.')), /no accounts were disconnected automatically/)
assert.match(googleClientImportError(new Error('ELOOP: fixture-private-file')), /original JSON/)
for (const value of ['fixture-private-token', 'Unknown exception with credential: fixture-private-token', 'Error: client_id=fixture-private-token']) {
  assert.doesNotMatch(googleClientImportError(new Error(value)), /fixture-private|client_id=/)
}
assert.doesNotMatch(googleClientImportError({ message: 'fixture-private-token' }), /fixture-private/)
console.log('Google Desktop client import field feedback, IPC wrapper handling and arbitrary secret/path omission passed')
