import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import os from 'node:os'
import { BOT_VM_GUEST } from './guest'

const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-vm-guest-test-'))
try {
  const workspace = path.join(root, 'workspace')
  await fs.mkdir(workspace)
  // Exercise the exact guest program against a private fixture directory, with
  // only the guest root remapped so tests never create a host /workspace.
  const program = BOT_VM_GUEST.replaceAll('/workspace', workspace).replace('value=value[11:]', `value=value[${workspace.length + 1}:]`)
  const run = (tool: string, args: Record<string, unknown>) => {
    const result = spawnSync('/usr/bin/python3', ['-I', '-c', program], { input: JSON.stringify({ tool, args }), encoding: 'utf8', timeout: 10000 })
    assert.equal(result.status, 0, result.stderr)
    return JSON.parse(result.stdout)
  }
  assert.equal(run('write_file', { path: 'draft/a.txt', content: 'blue\nsecond' }).ok, true)
  assert.match(run('read_file', { path: 'draft/a.txt', offset: 2, limit: 1 }).output, /2\tsecond/)
  assert.equal(run('edit_file', { path: 'draft/a.txt', old_string: 'blue', new_string: 'amber' }).ok, true)
  assert.match(run('search', { query: 'amber' }).output, /draft\/a.txt:1:amber/)
  assert.match(run('glob', { pattern: '**/*.txt' }).output, /draft\/a.txt/)
  assert.equal(run('write_file', { path: '../outside.txt', content: 'bad' }).ok, false)
  assert.equal(run('read_file', { path: '/etc/passwd' }).ok, false)
  assert.equal(run('read_file', { path: 'a\0b' }).ok, false)
  await fs.writeFile(path.join(root, 'outside.txt'), 'fixture secret')
  await fs.symlink(path.join(root, 'outside.txt'), path.join(workspace, 'link.txt'))
  assert.equal(run('read_file', { path: 'link.txt' }).ok, false)
  assert.equal(run('write_file', { path: 'link.txt', content: 'bad' }).ok, false)
  assert.equal(await fs.readFile(path.join(root, 'outside.txt'), 'utf8'), 'fixture secret')
  await fs.symlink(root, path.join(workspace, 'linked-dir'))
  assert.equal(run('write_file', { path: 'linked-dir/outside.txt', content: 'bad' }).ok, false)
  await fs.link(path.join(root, 'outside.txt'), path.join(workspace, 'hardlink.txt'))
  assert.equal(run('read_file', { path: 'hardlink.txt' }).ok, false)
  assert.equal(run('write_file', { path: 'hardlink.txt', content: 'bad' }).ok, false)
  assert.equal(run('list_dir', {}).output.split('\n').includes('link.txt'), false)
  assert.equal(run('edit_file', { path: 'draft/a.txt', old_string: '', new_string: 'bad' }).ok, false)
  assert.equal(run('computer_use', {}).ok, false)
  const command = run('run_command', { command: 'printf hello' })
  assert.deepEqual({ ok: command.ok, output: command.output, code: command.code }, { ok: true, output: 'hello', code: 0 })
  const timeout = run('run_command', { command: 'sleep 8', timeoutMs: 1000 })
  assert.equal(timeout.ok, false); assert.match(timeout.output, /time limit/)
  const bounded = run('run_command', { command: 'yes bounded-output' })
  assert.equal(bounded.ok, false); assert.equal(bounded.truncated, true); assert.ok(bounded.output.length < 66000)
  console.log('bot-vm guest file safety, bounded commands and literal search checks passed')
} finally { await fs.rm(root, { recursive: true, force: true }) }
