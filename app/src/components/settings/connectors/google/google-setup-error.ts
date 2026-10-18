const IMPORT_ERRORS: Record<string, string> = {
  'Google desktop client configuration is too large.': 'The Desktop client JSON is too large; choose the original Google Cloud download smaller than 32 KB.',
  'Select the installed desktop OAuth client JSON from Google Cloud.': 'This file is not valid Desktop OAuth client JSON; download the Desktop app client from Google Cloud.',
  'Google OAuth client must be a Desktop app, not a web client or service account.': 'Choose a Desktop app OAuth client; web clients and service-account keys cannot sign in to this app.',
  'Unexpected Google authorization endpoint.': 'The client file contains an unexpected authorization endpoint; use an unmodified Desktop client download from Google Cloud.',
  'Unexpected Google token endpoint.': 'The client file contains an unexpected token endpoint; use an unmodified Desktop client download from Google Cloud.',
  'Google desktop client does not allow the local callback.': 'The client file does not allow the local callback; create and download a Desktop app OAuth client.',
  'Enter a valid Google desktop OAuth client ID.': 'The client ID in this file is missing or invalid; download the Desktop app client JSON again.',
  'Google desktop client JSON must include its client secret.': 'This file is missing a valid client secret; import the complete Desktop app client JSON downloaded from Google Cloud.',
  'Disconnect Google accounts before changing the desktop client.': 'Disconnect every Google account before replacing the Desktop client; no accounts were disconnected automatically.',
  'Choose a Google Desktop OAuth client JSON smaller than 64 KB.': 'Choose a nonempty, regular Desktop client JSON file; the configuration must be smaller than 32 KB.',
  'The selected client file changed while reading it.': 'The selected file changed during import; choose the original Desktop client download again.',
}

// Only known, static host validation messages become specific UI copy; filenames and
// arbitrary exception text never leave this classifier for display.
export function googleClientImportError(reason: unknown): string {
  const raw = reason instanceof Error ? reason.message : ''
  const message = raw.replace(/^Error invoking remote method '[^']+':\s*/, '').replace(/^Error:\s*/, '')
  if (IMPORT_ERRORS[message]) return IMPORT_ERRORS[message]
  if (/^(ELOOP|EMLINK)(?:\b|:)/.test(message)) return 'Choose the original JSON file rather than a symbolic link or alias.'
  if (/^(ENOENT|EACCES|EPERM)(?:\b|:)/.test(message)) return 'The selected file is missing or cannot be read; choose an accessible original Desktop client JSON file.'
  return 'The Desktop client could not be imported; check the file, local credential store and Google Cloud configuration, then retry.'
}
