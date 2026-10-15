import assert from 'node:assert/strict'
import os from 'node:os'
import { blockedCommandReason } from './command-policy'

const HOME = os.homedir()

// Catastrophic-deletion bypasses that previously slipped through blockedCommandReason.
// Full Auto is now sandboxed too, but this remains a hard fail-closed boundary in every mode.
const blocked = [
  // `${HOME}` brace syntax (the literal `$HOME` form was already caught, the brace form was not).
  'rm -rf ${HOME}',
  // The bare resolved home root, with no trailing subpath.
  `rm -rf ${HOME}`,
  `rm -rf "${HOME}"`,
  `rm -rf ${HOME}/`,
  // Quoted-then-slashed home expansion: the closing quote sits between `$HOME` and the `/`.
  'rm -rf "$HOME"/Documents',
  "rm -rf '$HOME'/Documents",
  `rm -rf "${HOME}"/Documents`,
  // git invoked by absolute / relative path must still hit the commit/push hygiene gate.
  '/usr/bin/git push origin main',
  '/usr/bin/git commit -m "x"',
  '/usr/local/bin/git push',
  './git push origin main',
  'echo ok && /usr/bin/git -C . commit -m "x"',
]

for (const command of blocked) {
  assert.ok(blockedCommandReason(command), `expected blocked: ${command}`)
}

// Regression guard: legitimate non-home/non-git commands and look-alike user dirs stay unblocked.
const allowed = [
  'rm -rf node_modules',
  'rm -rf ./build',
  'rm -rf /tmp/scratch',
  // sibling dirs that merely share the home prefix are NOT the home directory.
  `rm -rf ${HOME}-backup`,
  `rm -rf ${HOME}xyz`,
  // not the `git` binary.
  'gitfoo push',
  '/usr/bin/digit push',
  'git status',
]

for (const command of allowed) {
  assert.equal(blockedCommandReason(command), null, `expected allowed: ${command}`)
}

console.log('command-policy home/git bypass tests passed')
