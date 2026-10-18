import assert from 'node:assert/strict'
import { execFile as execFileCallback, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { createServer } from 'node:https'
import { tmpdir } from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'
const execFile = promisify(execFileCallback)

if (process.platform !== 'darwin') {
  console.log('Actual native watch HTTPS transport regression requires macOS Security/Swift.')
} else {
  const directory = mkdtempSync(path.join(tmpdir(), 'tensorfold-watch-tls-'))
  const key = path.join(directory, 'fixture.key'), cert = path.join(directory, 'fixture.pem'), der = path.join(directory, 'fixture.der')
  const run = (file: string, args: string[]) => {
    const result = spawnSync(file, args, {encoding: 'utf8', timeout: 60_000})
    assert.equal(result.status, 0, result.stderr || result.error?.message)
  }
  let server: ReturnType<typeof createServer> | undefined
  try {
    run('openssl', ['req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '2', '-keyout', key, '-out', cert,
      '-subj', '/CN=127.0.0.1', '-addext', 'subjectAltName=IP:127.0.0.1', '-addext', 'extendedKeyUsage=serverAuth'])
    run('openssl', ['x509', '-in', cert, '-outform', 'DER', '-out', der])
    const pin = createHash('sha256').update(readFileSync(der)).digest('hex')
    const executable = path.join(directory, 'native-request')
    run('xcrun', ['swiftc', 'mobile/modules/tensorfold-pinned-network/ios/PinnedCertificate.swift',
      'mobile/modules/tensorfold-watch-connectivity/ios/WatchBackgroundPolicy.swift',
      'mobile/modules/tensorfold-watch-connectivity/ios/WatchBackgroundRelay.swift',
      'mobile/modules/tensorfold-watch-connectivity/ios/WatchBackgroundTransport.swift',
      'mobile/modules/tensorfold-watch-connectivity/tests/RequestMain.swift', '-o', executable])
    const visited: string[] = []
    server = createServer({key: readFileSync(key), cert: readFileSync(cert)}, (request, response) => {
      visited.push(request.url || '')
      assert.equal(request.method, 'POST')
      assert.equal(request.headers.authorization, 'Bearer fixture-phone-only-token')
      assert.equal(request.headers.cookie, undefined)
      assert.equal(request.headers.accept, 'application/json')
      const chunks: Buffer[] = []
      request.on('data', chunk => chunks.push(chunk))
      request.on('end', () => {
        const body = JSON.parse(Buffer.concat(chunks).toString())
        assert.equal(body.session, 'session-test-1')
        assert.equal(body.token, undefined)
        if (request.url?.startsWith('/redirect/')) { response.writeHead(302, {location: '/destination'}); response.end(); return }
        if (request.url?.startsWith('/large/')) { response.writeHead(200, {'content-type': 'application/json', 'content-length': '60001'}); response.end('x'.repeat(60_001)); return }
        if (request.url?.startsWith('/chunks/')) {
          response.writeHead(200, {'content-type': 'application/json'}); response.write('x'.repeat(50_000)); response.end('x'.repeat(10_001)); return
        }
        if (request.url?.startsWith('/wrong-mime/')) { response.writeHead(200, {'content-type': 'text/plain'}); response.end('untrusted'); return }
        if (request.url?.startsWith('/revoked/')) { response.writeHead(401, {'content-type': 'application/json'}); response.end('{"error":"revoked"}'); return }
        response.writeHead(200, {'content-type': 'application/json', 'set-cookie': 'fixtureCookie=must-not-return; Secure'})
        response.end(JSON.stringify({id: body.id, ok: true, message: 'Updated'}))
      })
    })
    await new Promise<void>(resolve => server!.listen(0, '127.0.0.1', resolve))
    const address = server.address(); assert.ok(address && typeof address !== 'string')
    const base = `https://127.0.0.1:${address.port}`
    const request = async (suffix: string, certificatePin = pin) => {
      const result = await execFile(executable, [base + suffix, certificatePin], {timeout: 20_000, maxBuffer: 20_000})
      return JSON.parse(result.stdout.trim()) as {completed: boolean; accepted: boolean; bytes: number}
    }
    const accepted = await request('')
    assert.equal(accepted.completed, true); assert.equal(accepted.accepted, true); assert.ok(accepted.bytes > 0)
    const count = visited.length
    assert.equal((await request('', '0'.repeat(64))).accepted, false, 'Wrong pinned certificate must fail before HTTP credentials are delivered')
    assert.equal(visited.length, count, 'Wrong pin cannot send an authenticated request')
    for (const suffix of ['/redirect', '/large', '/chunks', '/wrong-mime', '/revoked']) {
      const denied = await request(suffix)
      assert.equal(denied.completed, true, `${suffix} completes within finite budget`)
      assert.equal(denied.accepted, false, `${suffix} is rejected`)
    }
    assert.ok(!visited.includes('/destination'), 'Redirect target never receives the bearer credential')
    assert.equal((await request('')).accepted, true, 'Subsequent request uses fresh cookie-free ephemeral session')
    console.log('Actual Swift URLSession transport: exact pin and fixed POST succeed; wrong pin, redirect, declared/chunked oversize, MIME and revoked credentials fail; cookies are omitted.')
  } finally {
    if (server) await new Promise<void>(resolve => server!.close(() => resolve()))
    rmSync(directory, {recursive: true, force: true})
  }
}
