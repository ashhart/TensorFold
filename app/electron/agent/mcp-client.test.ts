// Run: npx tsx electron/agent/mcp-client.test.ts
import assert from 'node:assert/strict'
import { createHash, generateKeyPairSync, sign } from 'node:crypto'
import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { canonicalMcpManifest, createMcpServer, mcpSpawnSpec, normalizeMcpPrompt, normalizeMcpResource, verifyConfiguredManifest } from './mcp-client'

async function main() {
  const dir = await fs.mkdtemp(path.join(os.tmpdir(), 'vontra-mcp-manifest-'))
  const manifest = { schemaVersion: 1, name: 'Echo MCP', command: '/bin/echo', args: ['hello'] }
  const manifestPath = path.join(dir, 'mcp-manifest.json')
  await fs.writeFile(manifestPath, JSON.stringify(manifest, null, 2), 'utf8')

  const { publicKey, privateKey } = generateKeyPairSync('ed25519')
  const signature = sign(null, Buffer.from(canonicalMcpManifest(manifest)), privateKey).toString('base64')
  const config = createMcpServer({
    name: 'Echo MCP',
    command: '/bin/echo',
    args: ['hello'],
    manifestPath,
    manifestSignature: signature,
    manifestPublicKey: publicKey.export({ type: 'spki', format: 'pem' }).toString(),
  })

  const verified = await verifyConfiguredManifest(config)
  assert.equal(verified.ok, true, verified.error ?? '')
  assert.match(verified.digest ?? '', /^[a-f0-9]{64}$/)

  const unpinnedAutoConnect = await verifyConfiguredManifest({ ...config, autoConnect: true })
  assert.equal(unpinnedAutoConnect.ok, false)
  assert.match(unpinnedAutoConnect.error ?? '', /must pin commandSha256/i)

  const tampered = await verifyConfiguredManifest({ ...config, args: ['goodbye'] })
  assert.equal(tampered.ok, false)
  assert.match(tampered.error ?? '', /args do not match/)

  const badSignature = await verifyConfiguredManifest({ ...config, manifestSignature: Buffer.from('bad').toString('base64') })
  assert.equal(badSignature.ok, false)
  assert.match(badSignature.error ?? '', /signature verification failed/)

  // commandSha256 pin: a signed manifest can pin the on-disk binary's hash. A correct hash
  // verifies; a wrong hash (binary swapped, same name) is rejected. (gap-audit: manifest
  // previously pinned only the command string, never the executable.)
  {
    const echoSha = createHash('sha256').update(await fs.readFile('/bin/echo')).digest('hex')
    const pinned = { ...manifest, commandSha256: echoSha }
    const pinnedPath = path.join(dir, 'pinned-manifest.json')
    await fs.writeFile(pinnedPath, JSON.stringify(pinned, null, 2), 'utf8')
    const pinnedSig = sign(null, Buffer.from(canonicalMcpManifest(pinned)), privateKey).toString('base64')
    const pinnedConfig = createMcpServer({
      name: 'Echo MCP', command: '/bin/echo', args: ['hello'],
      manifestPath: pinnedPath, manifestSignature: pinnedSig,
      manifestPublicKey: publicKey.export({ type: 'spki', format: 'pem' }).toString(),
    })
    const okHash = await verifyConfiguredManifest(pinnedConfig)
    assert.equal(okHash.ok, true, okHash.error ?? 'expected pinned-hash manifest to verify')

    const wrong = { ...manifest, commandSha256: 'deadbeef'.repeat(8) }
    const wrongPath = path.join(dir, 'wrong-manifest.json')
    await fs.writeFile(wrongPath, JSON.stringify(wrong, null, 2), 'utf8')
    const wrongSig = sign(null, Buffer.from(canonicalMcpManifest(wrong)), privateKey).toString('base64')
    const wrongConfig = createMcpServer({
      name: 'Echo MCP', command: '/bin/echo', args: ['hello'],
      manifestPath: wrongPath, manifestSignature: wrongSig,
      manifestPublicKey: publicKey.export({ type: 'spki', format: 'pem' }).toString(),
    })
    const badHash = await verifyConfiguredManifest(wrongConfig)
    assert.equal(badHash.ok, false)
    assert.match(badHash.error ?? '', /hash does not match/)
  }

  // A trusted interpreter launch binds both the interpreter and the mutable script bytes.
  // Re-signing a manifest without a launch-file pin is insufficient; swapping the script
  // after signing is detected before the child process starts.
  {
    const scriptPath = path.join(dir, 'server.js')
    await fs.writeFile(scriptPath, 'console.log("safe")\n', 'utf8')
    const commandSha256 = createHash('sha256').update(await fs.readFile(process.execPath)).digest('hex')
    const scriptSha256 = createHash('sha256').update(await fs.readFile(scriptPath)).digest('hex')
    const scriptManifest = {
      schemaVersion: 1,
      name: 'Script MCP',
      command: process.execPath,
      args: ['server.js'],
      commandSha256,
      launchFiles: [{ path: 'server.js', sha256: scriptSha256 }],
    }
    const scriptManifestPath = path.join(dir, 'script-manifest.json')
    await fs.writeFile(scriptManifestPath, JSON.stringify(scriptManifest), 'utf8')
    const scriptSig = sign(null, Buffer.from(canonicalMcpManifest(scriptManifest)), privateKey).toString('base64')
    const scriptConfig = createMcpServer({
      name: 'Script MCP', command: process.execPath, args: ['server.js'], cwd: dir, autoConnect: true,
      manifestPath: scriptManifestPath, manifestSignature: scriptSig,
      manifestPublicKey: publicKey.export({ type: 'spki', format: 'pem' }).toString(),
    })
    assert.equal((await verifyConfiguredManifest(scriptConfig)).ok, true)

    const missingFilePin = { ...scriptManifest, launchFiles: [] }
    const missingPath = path.join(dir, 'missing-script-pin.json')
    await fs.writeFile(missingPath, JSON.stringify(missingFilePin), 'utf8')
    const missingSig = sign(null, Buffer.from(canonicalMcpManifest(missingFilePin)), privateKey).toString('base64')
    const missing = await verifyConfiguredManifest({ ...scriptConfig, manifestPath: missingPath, manifestSignature: missingSig })
    assert.equal(missing.ok, false)
    assert.match(missing.error ?? '', /must pin mutable launch file/i)

    await fs.writeFile(scriptPath, 'console.log("swapped")\n', 'utf8')
    const swapped = await verifyConfiguredManifest(scriptConfig)
    assert.equal(swapped.ok, false)
    assert.match(swapped.error ?? '', /launch file hash does not match/i)
  }

  const pinnedDockerImage = `vontra/mcp-test@sha256:${'a'.repeat(64)}`
  const dockerConfig = createMcpServer({
    name: 'Gateway MCP',
    launchMode: 'docker_gateway',
    command: 'node',
    args: ['server.js'],
    cwd: dir,
    env: { API_TOKEN: 'secret', TENANT_ID: 'tenant' },
    dockerGateway: {
      image: pinnedDockerImage,
      workdir: '/workspace',
      network: 'none',
      readOnlyRootfs: true,
      envAllowlist: ['API_TOKEN'],
      mounts: [{ source: dir, target: '/workspace', readOnly: true }],
    },
  })
  const spawnSpec = mcpSpawnSpec(dockerConfig)
  assert.equal(spawnSpec.command, 'docker')
  assert.ok(spawnSpec.args.includes('--read-only'))
  assert.ok(spawnSpec.args.includes('type=bind,src=' + path.resolve(dir) + ',dst=/workspace,readonly'))
  assert.ok(spawnSpec.args.includes('API_TOKEN'))
  assert.ok(!spawnSpec.args.includes('API_TOKEN=secret'))
  assert.ok(!spawnSpec.args.includes('TENANT_ID'))
  assert.deepEqual(spawnSpec.args.slice(-3), [pinnedDockerImage, 'node', 'server.js'])
  assert.equal(spawnSpec.gateway?.kind, 'docker')
  assert.equal(spawnSpec.gateway?.network, 'none')
  assert.equal(spawnSpec.gateway?.readOnlyRootfs, true)
  assert.equal(spawnSpec.gateway?.mountCount, 1)
  assert.equal(spawnSpec.env.API_TOKEN, 'secret')
  assert.equal(spawnSpec.env.TENANT_ID, undefined)
  assert.throws(() => mcpSpawnSpec(createMcpServer({
    ...dockerConfig,
    dockerGateway: { ...dockerConfig.dockerGateway!, network: 'host' },
  })), /cannot use Docker host networking/i)
  assert.throws(() => mcpSpawnSpec(createMcpServer({
    ...dockerConfig,
    dockerGateway: { ...dockerConfig.dockerGateway!, image: 'vontra/mcp-test:latest' },
  })), /must be pinned by immutable sha256 digest/i)
  // The cwd becomes a sandbox WRITE root; a filesystem root must never qualify.
  assert.throws(() => mcpSpawnSpec({ ...createMcpServer({ name: 'root-cwd', command: '/bin/ls' }), cwd: '/' }), /cannot be a filesystem root/i)

  const directConfig = createMcpServer({
    name: 'Direct MCP',
    command: '/bin/echo',
    args: ['hello'],
    cwd: dir,
  })
  delete process.env.VONTRA_ALLOW_UNSANDBOXED_MCP
  assert.throws(
    () => mcpSpawnSpec(directConfig, { macSandboxAvailable: false }),
    /No supported MCP process sandbox is available/,
  )
  process.env.VONTRA_ALLOW_UNSANDBOXED_MCP = '1'
  const unsafeDirect = mcpSpawnSpec(directConfig, { macSandboxAvailable: false })
  assert.equal(unsafeDirect.command, '/bin/echo')
  assert.equal(unsafeDirect.sandboxed, false)
  delete process.env.VONTRA_ALLOW_UNSANDBOXED_MCP

  const resource = normalizeMcpResource(config, {
    uri: 'file:///repo/docs/guide.md',
    title: 'Project guide',
    description: 'Shared context',
    mimeType: 'text/markdown',
  }, 0)
  assert.deepEqual(resource, {
    serverId: config.id,
    serverName: config.name,
    uri: 'file:///repo/docs/guide.md',
    name: 'Project guide',
    title: 'Project guide',
    description: 'Shared context',
    mimeType: 'text/markdown',
  })
  assert.equal(normalizeMcpResource(config, { name: 'missing uri' }, 0), null)

  const prompt = normalizeMcpPrompt(config, {
    name: 'triage',
    title: 'Triage',
    description: 'Summarise a ticket',
    arguments: [
      { name: 'ticket', description: 'Ticket body', required: true },
      { name: 'optionalNotes', required: false },
      { description: 'ignored' },
    ],
  }, 0)
  assert.deepEqual(prompt, {
    serverId: config.id,
    serverName: config.name,
    name: 'triage',
    title: 'Triage',
    description: 'Summarise a ticket',
    arguments: [
      { name: 'ticket', description: 'Ticket body', required: true },
      { name: 'optionalNotes', description: undefined, required: false },
    ],
  })
  assert.equal(normalizeMcpPrompt(config, { title: 'missing name' }, 0), null)

  await fs.rm(dir, { recursive: true, force: true })
  console.log('mcp-client tests passed')
}

void main()
