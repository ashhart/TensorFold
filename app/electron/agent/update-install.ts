import { randomBytes } from 'node:crypto'
import { constants, promises as fs } from 'node:fs'
import http from 'node:http'
import { reverifyDownloadedUpdate, type VerifiedUpdateDownloadResult } from './update-download'

export interface NativeUpdateDriver {
  on(event: string, listener: (...args: any[]) => void): unknown
  removeListener(event: string, listener: (...args: any[]) => void): unknown
  setFeedURL(options: { url: string; serverType: 'default' }): void
  checkForUpdates(): void
  quitAndInstall(): void
}

export interface UpdateInstallSafety {
  acquireLease(): Promise<{ release(): void }>
  assertIdle(): void
  preserve(): Promise<void>
}

export function assertPinnedMacSignature(details: string, trust: { identity: string; teamId: string; productionReady: boolean }): void {
  const lines = details.split(/\r?\n/).map((line) => line.trim())
  if (!trust.productionReady || lines.includes('Signature=adhoc') || !lines.includes(`Authority=${trust.identity}`) || !lines.includes(`TeamIdentifier=${trust.teamId}`)) {
    throw new Error('Automatic installation requires the pinned production Developer ID signature.')
  }
}

// Squirrel sees only this private loopback feed after the public manifest and ZIP were verified.
// The random path authenticates both requests; no renderer gets the server URL or artifact path.
export async function startVerifiedUpdateFeed(input: {
  download: VerifiedUpdateDownloadResult
  downloadRoot: string
  version: string
  releaseName: string | null
  releaseNotes: string | null
}): Promise<{ url: string; close(): Promise<void> }> {
  const verified = await reverifyDownloadedUpdate(input.download, input.downloadRoot)
  const token = randomBytes(32).toString('hex')
  const manifestPath = `/${token}/manifest`
  const artifactPath = `/${token}/update.zip`
  let port = 0
  const server = http.createServer((request, response) => {
    void (async () => {
      if (request.headers.host !== `127.0.0.1:${port}` || !['GET', 'HEAD'].includes(request.method ?? '')) {
        response.writeHead(403).end()
        return
      }
      if (request.url === manifestPath) {
        const body = JSON.stringify({ url: `http://127.0.0.1:${port}${artifactPath}`, name: input.releaseName ?? input.version, notes: input.releaseNotes ?? '', sha256: verified.sha256, size: verified.bytes })
        response.writeHead(200, { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body), 'Cache-Control': 'no-store' })
        response.end(request.method === 'HEAD' ? undefined : body)
        return
      }
      if (request.url !== artifactPath) { response.writeHead(404).end(); return }
      // Recheck managed path, hash and inode immediately before serving Squirrel's request.
      const artifact = await reverifyDownloadedUpdate(verified, input.downloadRoot)
      const handle = await fs.open(artifact.path, constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0))
      const stat = await handle.stat({ bigint: true })
      if (String(stat.dev) !== artifact.fileIdentity.device || String(stat.ino) !== artifact.fileIdentity.inode || stat.size !== BigInt(artifact.bytes)) {
        await handle.close()
        throw new Error('Verified update file changed before staging.')
      }
      response.writeHead(200, { 'Content-Type': 'application/zip', 'Content-Length': artifact.bytes, 'Cache-Control': 'no-store' })
      if (request.method === 'HEAD') { await handle.close(); response.end(); return }
      const stream = handle.createReadStream()
      stream.on('error', () => response.destroy())
      response.on('close', () => stream.destroy())
      stream.pipe(response)
    })().catch(() => { if (response.headersSent) response.destroy(); else response.writeHead(500).end() })
  })
  server.requestTimeout = 30_000
  await new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen({ host: '127.0.0.1', port: 0, exclusive: true }, () => resolve())
  })
  const address = server.address()
  if (!address || typeof address === 'string') { server.close(); throw new Error('Could not create the verified update feed.') }
  port = address.port
  server.unref()
  return {
    url: `http://127.0.0.1:${port}${manifestPath}`,
    close: () => new Promise<void>((resolve) => { server.close(() => resolve()); server.closeAllConnections() }),
  }
}

export function stageNativeUpdate(driver: NativeUpdateDriver, feedUrl: string, timeoutMs = 15 * 60_000): Promise<void> {
  return new Promise((resolve, reject) => {
    const finish = (error?: Error) => {
      clearTimeout(timer)
      driver.removeListener('error', failed)
      driver.removeListener('update-downloaded', downloaded)
      driver.removeListener('update-not-available', missing)
      error ? reject(error) : resolve()
    }
    const failed = () => finish(new Error('macOS could not stage the verified update; the app remains open.'))
    const downloaded = () => finish()
    const missing = () => finish(new Error('macOS did not stage an update; the app remains open.'))
    const timer = setTimeout(() => finish(new Error('Preparing the update timed out; the app remains open.')), timeoutMs)
    driver.on('error', failed)
    driver.on('update-downloaded', downloaded)
    driver.on('update-not-available', missing)
    try { driver.setFeedURL({ url: feedUrl, serverType: 'default' }); driver.checkForUpdates() } catch { failed() }
  })
}

export async function installVerifiedUpdate(input: {
  driver: NativeUpdateDriver
  safety: UpdateInstallSafety
  openFeed(): Promise<{ url: string; close(): Promise<void> }>
  stage?: typeof stageNativeUpdate
}): Promise<void> {
  input.safety.assertIdle()
  const lease = await input.safety.acquireLease()
  let feed: Awaited<ReturnType<typeof input.openFeed>> | null = null
  let handedOff = false
  let handoffFailed: (() => void) | null = null
  try {
    input.safety.assertIdle()
    await input.safety.preserve()
    input.safety.assertIdle()
    feed = await input.openFeed()
    await (input.stage ?? stageNativeUpdate)(input.driver, feed.url)
    // Squirrel only receives the ZIP after admission is closed and durable state is preserved.
    input.safety.assertIdle()
    await input.safety.preserve()
    input.safety.assertIdle()
    await feed.close()
    feed = null
    // quitAndInstall can fail asynchronously; release admission if macOS keeps this app open.
    handoffFailed = () => { lease.release(); if (handoffFailed) input.driver.removeListener('error', handoffFailed) }
    input.driver.on('error', handoffFailed)
    input.driver.quitAndInstall()
    handedOff = true
  } finally {
    if (feed) await feed.close()
    if (!handedOff) {
      if (handoffFailed) input.driver.removeListener('error', handoffFailed)
      lease.release()
    }
  }
}
