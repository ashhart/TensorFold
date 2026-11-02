import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import net from 'node:net'
import { runSandboxedCommand } from './sandbox'
import { canUseLinuxSandbox } from './linux-sandbox'
import { canUseWindowsSandbox } from './windows-sandbox'

const supported = canUseLinuxSandbox() || canUseWindowsSandbox()
if (!supported) {
  assert.notEqual(process.env.TENSORFOLD_REQUIRE_PORTABLE_SANDBOX, '1', 'Native qualification requires its actual sandbox backend.')
  console.log('Portable kernel sandbox requires a qualified Linux or Windows host.')
} else {
  const parent = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tf-native-confinement-')))
  const root = path.join(parent, 'project'), outside = path.join(parent, 'outside.txt')
  await fs.mkdir(root); await fs.writeFile(path.join(root, 'input.txt'), 'original'); await fs.writeFile(outside, 'private fixture')
  await fs.mkdir(path.join(root, '.git')); await fs.writeFile(path.join(root, '.git/config'), 'protected fixture')
  const secretBefore = process.env.TENSORFOLD_ISOLATION_TEST_SECRET
  process.env.TENSORFOLD_ISOLATION_TEST_SECRET = 'synthetic secret outside the child environment'
  const command = (javascript: string) => process.platform === 'win32'
    ? `node -e "${javascript.replace(/"/g, '\\"')}"` : `node -e '${javascript.replace(/'/g, `'"'"'`)}'`
  const run = (text: string, readOnly = false) => runSandboxedCommand(command(text), { cwd: root, profile: readOnly ? 'read-only' : 'build-test', timeoutMs: 15000 })
  try {
    let result = await run('require("fs").writeFileSync("input.txt","changed");require("fs").writeFileSync("output.txt","built")')
    assert.equal(result.code, 0, result.stderr); assert.equal(result.sandboxed, true)
    assert.equal(await fs.readFile(path.join(root, 'input.txt'), 'utf8'), 'changed')
    assert.equal(await fs.readFile(path.join(root, 'output.txt'), 'utf8'), 'built')
    result = await run('require("fs").writeFileSync("input.txt","forbidden")', true)
    assert.notEqual(result.code, 0); assert.equal(await fs.readFile(path.join(root, 'input.txt'), 'utf8'), 'changed')
    result = await run(`require("fs").readFileSync(${JSON.stringify(outside)})`)
    assert.notEqual(result.code, 0, 'Unrelated private files must be denied by the kernel')
    result = await run('if(process.env.TENSORFOLD_ISOLATION_TEST_SECRET)process.exit(42)')
    assert.equal(result.code, 0, result.stderr)
    result = await run('require("fs").writeFileSync(".git/config","unsafe")')
    assert.notEqual(result.code, 0); assert.equal(await fs.readFile(path.join(root, '.git/config'), 'utf8'), 'protected fixture')
    result = await run('const f=require("fs");f.writeFileSync("input.txt","unpublished");f.mkdirSync("nested/.git/hooks",{recursive:true});f.writeFileSync("nested/.git/hooks/pre-commit","unsafe")')
    assert.equal(result.code, 126, result.stderr)
    assert.equal(await fs.readFile(path.join(root, 'input.txt'), 'utf8'), 'changed')
    await assert.rejects(fs.stat(path.join(root, 'nested')), /ENOENT/)
    const controller = new AbortController()
    const pending = runSandboxedCommand(command('require("fs").writeFileSync("cancelled.txt","private");setTimeout(()=>{},30000)'), {
      cwd: root, profile: 'build-test', timeoutMs: 30000, signal: controller.signal,
    })
    setTimeout(() => controller.abort(), 500)
    const cancelled = await pending
    assert.notEqual(cancelled.code, 0)
    await assert.rejects(fs.stat(path.join(root, 'cancelled.txt')), /ENOENT/)
    const timed = await runSandboxedCommand(command('process.on("SIGTERM",()=>process.exit(0));require("fs").writeFileSync("timed.txt","private");setInterval(()=>{},1000)'), {
      cwd: root, profile: 'build-test', timeoutMs: 1500,
    })
    assert.equal(timed.code, 124, timed.stderr)
    await assert.rejects(fs.stat(path.join(root, 'timed.txt')), /ENOENT/)
    const overflow = await runSandboxedCommand(command('process.on("SIGTERM",()=>process.exit(0));require("fs").writeFileSync("overflow.txt","private");process.stdout.write("x".repeat(65536));setInterval(()=>{},1000)'), {
      cwd: root, profile: 'build-test', timeoutMs: 10000, maxBuffer: 1024,
    })
    assert.equal(overflow.code, 126, overflow.stderr)
    await assert.rejects(fs.stat(path.join(root, 'overflow.txt')), /ENOENT/)
    const listener = net.createServer(socket => socket.end())
    await new Promise<void>(resolve => listener.listen(0, '127.0.0.1', resolve))
    try {
      const port = (listener.address() as net.AddressInfo).port
      const probe = command(`const s=require("net").connect({host:"127.0.0.1",port:${port}});s.setTimeout(1000);s.on("connect",()=>{s.destroy();process.exit(0)});s.on("error",()=>process.exit(5));s.on("timeout",()=>process.exit(6))`)
      const denied = await runSandboxedCommand(probe, { cwd: root, profile: 'read-only', timeoutMs: 10000 })
      assert.notEqual(denied.code, 0, 'Default command network access is denied')
      if (process.platform === 'linux') {
        const admitted = await runSandboxedCommand(probe, { cwd: root, profile: 'network-fetch', allowNetwork: true, timeoutMs: 10000 })
        assert.equal(admitted.code, 0, admitted.stderr)
      }
    } finally { await new Promise<void>(resolve => listener.close(() => resolve())) }
    result = await run('const f=require("fs");const d=f.openSync("large-output.bin","w");f.ftruncateSync(d,129*1024*1024);f.closeSync(d)')
    assert.equal(result.code, 0, result.stderr)
    assert.equal((await fs.stat(path.join(root, 'large-output.bin'))).size, 129 * 1024 * 1024)
    console.log('Native portable sandbox: ordinary writes, read-only denial, private-file denial, environment filtering, Git protection and cancellation passed.')
  } finally {
    if (secretBefore === undefined) delete process.env.TENSORFOLD_ISOLATION_TEST_SECRET
    else process.env.TENSORFOLD_ISOLATION_TEST_SECRET = secretBefore
    await fs.rm(parent, { recursive: true, force: true })
  }
}
