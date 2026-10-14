import assert from 'node:assert/strict'
import {
  tokenizeSshConfigLine,
  parseSshConfig,
  sshHostMatches,
  listSshConfigHosts,
  resolveSshHostSettings,
  expandTilde,
  expandSshConfigIncludes,
  type SshIncludeResolver,
} from './ssh-config'

// --- tokenizer ---
assert.deepEqual(tokenizeSshConfigLine('HostName example.com'), ['HostName', 'example.com'])
assert.deepEqual(tokenizeSshConfigLine('Port=2222'), ['Port', '2222'])
assert.deepEqual(tokenizeSshConfigLine('Port = 2222'), ['Port', '2222'])
assert.deepEqual(tokenizeSshConfigLine('  User    deploy  '), ['User', 'deploy'])
assert.deepEqual(
  tokenizeSshConfigLine('IdentityFile "/path/with spaces/id_ed25519"'),
  ['IdentityFile', '/path/with spaces/id_ed25519'],
)
assert.deepEqual(tokenizeSshConfigLine('Host a b c'), ['Host', 'a', 'b', 'c'])

const CONFIG = `
# top of file
Host bastion
  HostName bastion.example.com
  User ops
  Port 2222
  IdentityFile ~/.ssh/bastion_ed25519

Host web-* !web-staging
  User deploy
  ForwardAgent yes

Host web-prod
  HostName 10.0.0.5
  IdentityFile ~/.ssh/prod_rsa

Host *
  User fallback
  IdentityFile ~/.ssh/id_ed25519
`

// --- model parsing ---
const model = parseSshConfig(CONFIG)
assert.equal(model.hosts.length, 4)
assert.deepEqual(model.hosts[0].patterns, ['bastion'])
assert.deepEqual(model.hosts[1].patterns, ['web-*', '!web-staging'])

// --- host listing (concrete aliases only) ---
assert.deepEqual(listSshConfigHosts(CONFIG), ['bastion', 'web-prod'])

// --- pattern matching ---
assert.equal(sshHostMatches(['web-*', '!web-staging'], 'web-prod'), true)
assert.equal(sshHostMatches(['web-*', '!web-staging'], 'web-staging'), false)
assert.equal(sshHostMatches(['*'], 'anything'), true)
assert.equal(sshHostMatches(['db?'], 'db1'), true)
assert.equal(sshHostMatches(['db?'], 'db12'), false)

// --- resolution: first-match-wins, IdentityFile accumulates ---
const bastion = resolveSshHostSettings(CONFIG, 'bastion')
assert.equal(bastion.hostName, 'bastion.example.com')
assert.equal(bastion.user, 'ops')
assert.equal(bastion.port, 2222)
// bastion's own key first, then the catch-all `Host *` key accumulates
assert.deepEqual(bastion.identityFiles, ['~/.ssh/bastion_ed25519', '~/.ssh/id_ed25519'])

const prod = resolveSshHostSettings(CONFIG, 'web-prod')
assert.equal(prod.hostName, '10.0.0.5')
// web-* sets User deploy first; catch-all `Host *` User fallback is ignored
assert.equal(prod.user, 'deploy')
assert.equal(prod.forwardAgent, true)
assert.equal(prod.port, 22) // default
assert.deepEqual(prod.identityFiles, ['~/.ssh/prod_rsa', '~/.ssh/id_ed25519'])

// unknown host falls through to catch-all only
const unknown = resolveSshHostSettings(CONFIG, 'random-box')
assert.equal(unknown.hostName, 'random-box')
assert.equal(unknown.user, 'fallback')
assert.deepEqual(unknown.identityFiles, ['~/.ssh/id_ed25519'])

// --- Match blocks are parsed but never applied ---
const withMatch = `
Match host *.internal
  User insider
Host known
  HostName known.example.com
`
assert.equal(resolveSshHostSettings(withMatch, 'anything.internal').user, undefined)
assert.deepEqual(listSshConfigHosts(withMatch), ['known'])

// --- tilde expansion ---
assert.equal(expandTilde('~/.ssh/id_ed25519', '/Users/dev'), '/Users/dev/.ssh/id_ed25519')
assert.equal(expandTilde('~', '/Users/dev'), '/Users/dev')
assert.equal(expandTilde('/abs/path', '/Users/dev'), '/abs/path')

// --- Include expansion (textual inlining, injected fs) ---
const files: Record<string, string> = {
  '/Users/dev/.ssh/config.d/work': 'Host work\n  HostName work.example.com\n  User worker\n',
}
const resolver: SshIncludeResolver = {
  homeDir: '/Users/dev',
  baseDir: '/Users/dev/.ssh',
  resolve: (pattern) => (pattern in files ? [pattern] : []),
  read: (path) => files[path] ?? null,
}
const root = 'Include ~/.ssh/config.d/work\nHost main\n  HostName main.example.com\n'
const expanded = expandSshConfigIncludes(root, resolver)
assert.deepEqual(listSshConfigHosts(expanded), ['work', 'main'])
assert.equal(resolveSshHostSettings(expanded, 'work').hostName, 'work.example.com')
assert.equal(resolveSshHostSettings(expanded, 'work').user, 'worker')

// include cycles terminate
const cyclic: SshIncludeResolver = {
  homeDir: '/h',
  baseDir: '/h/.ssh',
  resolve: () => ['/h/.ssh/loop'],
  read: () => 'Include /h/.ssh/loop\nHost deep\n  HostName deep.example.com\n',
}
const cyclicExpanded = expandSshConfigIncludes('Include /h/.ssh/loop\n', cyclic)
assert.ok(cyclicExpanded.length >= 0) // did not stack-overflow

console.log('ssh-config tests passed')
