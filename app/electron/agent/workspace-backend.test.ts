import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import {
  LocalBackend,
  SshBackend,
  type WorkspaceExecResult,
  type SshBackendSession,
} from './workspace-backend'
import type { SshSftpLike } from './ssh-connection'

async function testLocal() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-wb-'))
  const execCalls: Array<{ command: string; cwd?: string }> = []
  const backend = new LocalBackend(root, async (command, opts): Promise<WorkspaceExecResult> => {
    execCalls.push({ command, cwd: opts.cwd })
    return { code: 0, stdout: 'ok', stderr: '' }
  })

  const file = path.join(root, 'sub', 'a.txt')
  await backend.writeFile(file, 'hello')
  assert.equal(await backend.readFile(file), 'hello')
  assert.equal(await backend.exists(file), true)
  assert.equal(await backend.exists(path.join(root, 'nope')), false)

  const st = await backend.stat(file)
  assert.equal(st.isFile, true)
  assert.equal(st.size, 5)

  const entries = await backend.readdir(path.join(root, 'sub'))
  assert.deepEqual(entries.map((e) => e.name).sort(), ['a.txt'])
  assert.equal(entries[0].isFile, true)

  await backend.rename(file, path.join(root, 'sub', 'b.txt'))
  assert.equal(await backend.exists(file), false)
  assert.equal(await backend.exists(path.join(root, 'sub', 'b.txt')), true)

  await backend.remove(path.join(root, 'sub'))
  assert.equal(await backend.exists(path.join(root, 'sub')), false)

  const res = await backend.exec('echo hi')
  assert.equal(res.stdout, 'ok')
  assert.equal(execCalls[0].cwd, root) // defaults to rootCwd

  await fs.rm(root, { recursive: true, force: true })
}

function makeFakeSshSession() {
  const files = new Map<string, Buffer>()
  const execCalls: string[] = []
  const sftp: SshSftpLike = {
    async readFile(p) {
      const f = files.get(p)
      if (!f) throw new Error('ENOENT')
      return f
    },
    async writeFile(p, data) { files.set(p, Buffer.isBuffer(data) ? data : Buffer.from(data)) },
    async readdir(p) {
      const prefix = p.endsWith('/') ? p : p + '/'
      const names = new Set<string>()
      for (const key of files.keys()) {
        if (key.startsWith(prefix)) names.add(key.slice(prefix.length).split('/')[0])
      }
      return [...names].map((name) => ({
        filename: name,
        attrs: { mode: 0o100644, size: files.get(prefix + name)?.length ?? 0, mtime: 1000 },
      }))
    },
    async stat(p) {
      if (files.has(p)) return { isDirectory: false, isFile: true, mode: 0o100644, size: files.get(p)!.length, mtime: 1000 }
      throw new Error('ENOENT')
    },
    async mkdir() {},
    async unlink(p) { files.delete(p) },
    async rename(from, to) { const v = files.get(from); if (v) { files.set(to, v); files.delete(from) } },
  }
  // Map a logical path to where it really points (simulates remote symlinks). realpath -m
  // resolves the LONGEST matching symlink prefix, like the real tool.
  const canonical = new Map<string, string>()
  // Paths whose canonicalization yields EMPTY output (simulates the no-realpath-m fallback's
  // `cd` failing on a symlinked dir the SSH user can't enter — must FAIL CLOSED, not allow).
  const empty = new Set<string>()
  const fakeRealpath = (p: string): string => {
    if (empty.has(p)) return ''
    let logical = ''
    let real = ''
    for (const seg of p.split('/').filter(Boolean)) {
      logical += `/${seg}`
      real += `/${seg}`
      if (canonical.has(logical)) real = canonical.get(logical)!
    }
    return (real || '/').replace(/\/+$/, '') || '/'
  }
  const session: SshBackendSession = {
    async exec(command) {
      execCalls.push(command)
      const m = /^realpath -m -- '(.*?)' 2>\/dev\/null/.exec(command)
      if (m) {
        const p = m[1].replace(/'\\''/g, "'")
        return { code: 0, stdout: `${fakeRealpath(p)}\n`, stderr: '' }
      }
      return { code: 0, stdout: '', stderr: '' }
    },
    async sftp() { return sftp },
  }
  return { session, files, execCalls, canonical, empty }
}

async function testSsh() {
  const { session, files, execCalls } = makeFakeSshSession()
  const backend = new SshBackend('/srv/app', session)

  await backend.writeFile('/srv/app/src/index.ts', 'export const x = 1')
  assert.equal(await backend.readFile('/srv/app/src/index.ts'), 'export const x = 1')
  // writeFile created the parent via `mkdir -p`
  assert.ok(execCalls.some((c) => c.includes('mkdir -p') && c.includes('/srv/app/src')))

  assert.equal(await backend.exists('/srv/app/src/index.ts'), true)
  assert.equal(await backend.exists('/srv/app/missing'), false)

  const entries = await backend.readdir('/srv/app/src')
  assert.deepEqual(entries.map((e) => e.name), ['index.ts'])
  assert.equal(entries[0].isFile, true)

  const st = await backend.stat('/srv/app/src/index.ts')
  assert.equal(st.isFile, true)
  assert.equal(st.size, 'export const x = 1'.length)
  assert.equal(st.mtimeMs, 1000 * 1000)

  await backend.remove('/srv/app/src')
  assert.ok(execCalls.some((c) => c.startsWith('rm -rf') && c.includes('/srv/app/src')))

  const res = await backend.exec('npm test')
  assert.equal(res.code, 0)
  assert.ok(execCalls.includes('npm test'))

  // sftp is cached (single open)
  assert.ok(files instanceof Map)
}

async function testSshSymlinkContainment() {
  // A symlinked directory on the remote host must not let a write escape the workspace:
  // the parent dir canonicalizes (pwd -P) to a path outside the root, so the op is rejected.
  const { session, canonical, execCalls, empty } = makeFakeSshSession()
  const backend = new SshBackend('/srv/app', session)
  // /srv/app/evil is a symlink to /etc (a directory) on the host
  canonical.set('/srv/app/evil', '/etc')
  await assert.rejects(backend.writeFile('/srv/app/evil/passwd', 'x'), /escapes the workspace/, 'symlinked-dir write rejected')
  await assert.rejects(backend.readFile('/srv/app/evil/passwd'), /escapes the workspace/, 'symlinked-dir read rejected')
  await assert.rejects(backend.readdir('/srv/app/evil'), /escapes the workspace/, 'symlinked-dir listing rejected')
  await assert.rejects(backend.stat('/srv/app/evil/passwd'), /escapes the workspace/, 'symlinked-dir stat rejected')
  await assert.rejects(backend.mkdirp('/srv/app/evil/newdir'), /escapes the workspace/, 'symlinked-dir mkdir rejected')
  await assert.rejects(backend.remove('/srv/app/evil/passwd'), /escapes the workspace/, 'symlinked-dir remove rejected')
  // A symlinked LEAF (ln -s /etc/passwd leak) must also be caught — the parent is a real dir
  // but the leaf resolves outside (verifier bypass: only the parent was canonicalized before).
  canonical.set('/srv/app/leak', '/etc/passwd')
  await assert.rejects(backend.readFile('/srv/app/leak'), /escapes the workspace/, 'symlinked-leaf read rejected')
  await assert.rejects(backend.writeFile('/srv/app/leak', 'pwn'), /escapes the workspace/, 'symlinked-leaf write rejected')
  canonical.set('/srv/app/metadata', '/srv/app/.git')
  canonical.set('/srv/app/settings', '/srv/app/.env')
  await assert.rejects(backend.writeFile('/srv/app/metadata/config', 'fixture'), /safety policy/)
  await assert.rejects(backend.remove('/srv/app/metadata/config'), /safety policy/)
  await assert.rejects(backend.readFile('/srv/app/settings'), /safety policy/)
  // A nested write under a symlinked parent must not run mkdir first. `mkdir -p
  // /srv/app/evil/newdir` follows the symlink and creates /etc/newdir before the
  // later file containment check can reject the write.
  const beforeNested = execCalls.length
  await assert.rejects(backend.writeFile('/srv/app/evil/newdir/passwd', 'x'), /escapes the workspace/, 'nested symlinked-dir write rejected')
  const nestedCalls = execCalls.slice(beforeNested)
  assert.ok(!nestedCalls.some((c) => c.includes('mkdir -p') && c.includes('/srv/app/evil/newdir')), 'rejected nested symlink write must not mkdir outside the workspace before containment')
  // Fail-closed when canonicalization yields EMPTY (the fallback `cd` failed on a symlinked
  // dir the user can't enter): the root resolves but the target can't, so the op is REJECTED,
  // not allowed (verifier residual — empty canon previously fell through to allow).
  empty.add('/srv/app/blocked/secret')
  await assert.rejects(backend.readFile('/srv/app/blocked/secret'), /escapes the workspace/, 'unresolvable (empty-canon) path rejected, not allowed')
  await assert.rejects(backend.writeFile('/srv/app/blocked/secret', 'x'), /escapes the workspace/, 'unresolvable write rejected')
  // an ordinary in-workspace path still works
  await backend.writeFile('/srv/app/src/ok.ts', 'export const y = 2')
  assert.equal(await backend.readFile('/srv/app/src/ok.ts'), 'export const y = 2')
}

async function testSshSftpRetryAfterFailure() {
  // A single SFTP channel failure on a still-live transport must NOT poison the
  // backend forever: getSftp() must retry opening a fresh channel on the next op.
  const files = new Map<string, Buffer>()
  files.set('/srv/app/a.txt', Buffer.from('hi'))
  const sftp: SshSftpLike = {
    async readFile(p) { const f = files.get(p); if (!f) throw new Error('ENOENT'); return f },
    async writeFile(p, data) { files.set(p, Buffer.isBuffer(data) ? data : Buffer.from(data)) },
    async readdir() { return [] },
    async stat(p) {
      if (files.has(p)) return { isDirectory: false, isFile: true, mode: 0o100644, size: files.get(p)!.length, mtime: 1000 }
      throw new Error('ENOENT')
    },
    async mkdir() {},
    async unlink() {},
    async rename() {},
  }
  let sftpCalls = 0
  const session: SshBackendSession = {
    async exec(command) {
      // Resolve realpath probes to the path itself so the containment check passes and the test
      // actually reaches the SFTP path it is exercising.
      const m = /^realpath -m -- '(.*?)' 2>\/dev\/null/.exec(command)
      if (m) return { code: 0, stdout: `${m[1].replace(/'\\''/g, "'")}\n`, stderr: '' }
      return { code: 0, stdout: '', stderr: '' }
    },
    async sftp() {
      sftpCalls += 1
      // Fail only the first channel-open attempt; transport stays alive.
      if (sftpCalls === 1) throw new Error('SFTP channel open failed')
      return sftp
    },
  }
  const backend = new SshBackend('/srv/app', session)

  // First op fails (channel could not open)...
  await assert.rejects(backend.readFile('/srv/app/a.txt'), /SFTP channel open failed/)
  // ...but a subsequent op must recover by opening a fresh channel, not reuse
  // the cached rejection.
  assert.equal(await backend.readFile('/srv/app/a.txt'), 'hi')
  assert.equal(sftpCalls, 2)
}

async function testSshTransientEmptyRootDoesNotDowngrade() {
  // REGRESSION: a single transient empty stdout when canonicalizing the workspace ROOT (exec
  // timeout / premature channel close / flaky realpath) must NOT be cached forever and permanently
  // downgrade symlink containment to lexical-only. The op during the empty result must FAIL CLOSED,
  // and a later op (once the root resolves) must restore full containment, still catching escapes.
  const files = new Map<string, Buffer>()
  files.set('/srv/app/ok.txt', Buffer.from('hi'))
  const sftp: SshSftpLike = {
    async readFile(p) { const f = files.get(p); if (!f) throw new Error('ENOENT'); return f },
    async writeFile(p, data) { files.set(p, Buffer.isBuffer(data) ? data : Buffer.from(data)) },
    async readdir() { return [] },
    async stat(p) { if (files.has(p)) return { isDirectory: false, isFile: true, mode: 0o100644, size: files.get(p)!.length, mtime: 1000 }; throw new Error('ENOENT') },
    async mkdir() {}, async unlink() {}, async rename() {},
  }
  let rootResolveCalls = 0
  const session: SshBackendSession = {
    async exec(command) {
      const m = /^realpath -m -- '(.*?)' 2>\/dev\/null/.exec(command)
      if (m) {
        const p = m[1].replace(/'\\''/g, "'")
        if (p === '/srv/app') {
          rootResolveCalls += 1
          // First root canonicalization is transiently empty (e.g. exec timeout); then it works.
          return { code: 0, stdout: rootResolveCalls === 1 ? '\n' : '/srv/app\n', stderr: '' }
        }
        // /srv/app/evil is a symlink to /etc (escape); everything else maps to itself.
        if (p === '/srv/app/evil/passwd') return { code: 0, stdout: '/etc/passwd\n', stderr: '' }
        return { code: 0, stdout: `${p}\n`, stderr: '' }
      }
      return { code: 0, stdout: '', stderr: '' }
    },
    async sftp() { return sftp },
  }
  const backend = new SshBackend('/srv/app', session)

  // First op: root canonicalization is transiently empty -> must FAIL CLOSED (not silently allow).
  await assert.rejects(backend.readFile('/srv/app/ok.txt'), /could not be canonicalized|escapes the workspace/, 'transient empty root must fail closed')

  // Subsequent op: the root now resolves, so containment is fully restored — an in-workspace read
  // succeeds and a real symlink escape is still caught (proving we did NOT cache the empty root).
  assert.equal(await backend.readFile('/srv/app/ok.txt'), 'hi')
  await assert.rejects(backend.readFile('/srv/app/evil/passwd'), /escapes the workspace/, 'escape still caught after root recovers')
}

Promise.all([testLocal(), testSsh(), testSshSymlinkContainment(), testSshSftpRetryAfterFailure(), testSshTransientEmptyRootDoesNotDowngrade()])
  .then(() => console.log('workspace-backend tests passed'))
  .catch((e) => { console.error(e); process.exit(1) })
