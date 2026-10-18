import assert from 'node:assert/strict'
import { promises as fs } from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import { request as httpsRequest } from 'node:https'
import { createClusterNodeServer } from './server'
import { validateNodeConfig } from './config'
import { FakeEngine } from './testing/fixture'
import { ephemeralTls } from './testing/tls'

async function main() {
  const base = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'tensorfold-node-bounds-')))
  const nodes: Array<Awaited<ReturnType<typeof createClusterNodeServer>>> = []
  try {
    for (const patch of [{ host: '0.0.0.0' }, { advertise: 'yes' }, { managementToken: 'inline secret not permitted' }, { dataDir: 'relative' }, { port: 65536 }]) assert.throws(() => validateNodeConfig({ schema: 1, dataDir: path.join(base, 'invalid'), host: '127.0.0.1', port: 0, ...patch }))
    let now = Date.now()
    const root = path.join(base, 'http'), engine = new FakeEngine(), node = await createClusterNodeServer({ config: { schema: 1, dataDir: root, host: '127.0.0.1', port: 0 }, engine, maxBodyBytes: 128, now: () => now }); nodes.push(node)
    const endpoint = await node.listen(), token = (await fs.readFile(path.join(root, 'management.token'), 'utf8')).trim(), status = endpoint.url + '/api/cluster/v1/status'
    const submit = (body: string | Buffer, headers: Record<string, string> = {}) => fetch(endpoint.url + '/api/cluster/v1/models/ensure', { method: 'POST', headers: { authorization: `Bearer ${token}`, 'content-type': 'application/json', ...headers }, body: typeof body === 'string' ? body : Uint8Array.from(body) })
    assert.equal((await submit(JSON.stringify({ input: 'x'.repeat(256) }))).status, 413)
    assert.equal((await submit('{"__proto__":{"unsafe":true}}')).status, 400)
    assert.equal((await submit(Buffer.from([0xff, 0xff]))).status, 400)
    assert.equal((await submit('{}', { 'content-encoding': 'gzip' })).status, 415)
    assert.equal((await submit('{}', { 'content-type': 'text/plain' })).status, 415)
    assert.equal(engine.ensureCalls + engine.startCalls + engine.requestCalls, 0)
    for (let i = 0; i < 12; i++) assert.equal((await fetch(status, { headers: { authorization: 'Bearer ' + '0'.repeat(64) } })).status, 401)
    assert.equal((await fetch(status, { headers: { authorization: `Bearer ${token}` } })).status, 429)
    now += 60_001
    assert.equal((await fetch(status, { headers: { authorization: `Bearer ${token}` } })).status, 200)
    await node.close()
    const linked = path.join(base, 'linked'); await fs.symlink(root, linked)
    await assert.rejects(createClusterNodeServer({ config: { schema: 1, dataDir: linked, host: '127.0.0.1', port: 0 }, engine }), /real directories/)
    const tls = ephemeralTls(), tlsRoot = path.join(base, 'tls'); await fs.mkdir(tlsRoot, { mode: 0o700 })
    const certFile = path.join(tlsRoot, 'cert.pem'), keyFile = path.join(tlsRoot, 'key.pem')
    await fs.writeFile(certFile, tls.cert, { mode: 0o600 }); await fs.writeFile(keyFile, tls.key, { mode: 0o600 })
    const secure = await createClusterNodeServer({ config: { schema: 1, dataDir: tlsRoot, host: '0.0.0.0', port: 0, tls: { certFile, keyFile } }, engine }); nodes.push(secure)
    const secureEndpoint = await secure.listen(), secureToken = (await fs.readFile(path.join(tlsRoot, 'management.token'), 'utf8')).trim()
    const response = await new Promise<{ status: number; body: any }>((resolve, reject) => {
      const req = httpsRequest({ host: '127.0.0.1', port: secureEndpoint.port, path: '/api/cluster/v1/identity?nonce=synthetic_tls_nonce', ca: tls.cert, servername: 'localhost', headers: { authorization: `Bearer ${secureToken}` } }, res => { let body = ''; res.on('data', part => { body += part }); res.on('end', () => resolve({ status: res.statusCode!, body: JSON.parse(body) })) }); req.on('error', reject); req.end()
    })
    assert.equal(response.status, 200); assert.equal(response.body.authenticated, true)
    await assert.rejects(new Promise<void>((resolve, reject) => { const req = httpsRequest({ host: '127.0.0.1', port: secureEndpoint.port, path: '/' }, () => resolve()); req.on('error', reject); req.end() }), /self-signed|certificate/i, 'TLS trust is checked rather than disabled for a remote endpoint')
    console.log('PASS node limits: bounded/deep-safe JSON, encoding refusal, role-auth rate limit and recovery, private path/symlink refusal, no inline credentials, nonloopback TLS with a per-test generated certificate and default client trust enforcement')
  } finally { for (const node of nodes.reverse()) await node.close(); await fs.rm(base, { recursive: true, force: true }) }
}
main().catch(error => { console.error(error); process.exitCode = 1 })
