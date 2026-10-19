import { constants } from 'node:fs'
import { open } from 'node:fs/promises'

export const MAX_ARTIFACT_IPC_TEXT_BYTES = 16 * 1024 * 1024

/** Read only the requested prefix; never materialize the rest of a large artifact. */
export async function readArtifactTextPrefix(file: string, maxBytes: number): Promise<string> {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new Error('Artifact preview limit must be a positive integer.')
  const handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    const info = await handle.stat()
    if (!info.isFile()) throw new Error('Artifact path is not a regular file.')
    const buffer = Buffer.allocUnsafe(Math.min(maxBytes, Math.max(1, info.size)))
    const { bytesRead } = await handle.read(buffer, 0, buffer.byteLength, 0)
    return buffer.subarray(0, bytesRead).toString('utf8')
  } finally {
    await handle.close()
  }
}

/**
 * Read textual content crossing IPC with an absolute byte cap. The handle is opened before the
 * size check and the loop enforces the cap again, so replacement/growth cannot turn the check
 * into an unbounded allocation.
 */
export async function readArtifactTextBounded(file: string, maxBytes = MAX_ARTIFACT_IPC_TEXT_BYTES): Promise<string> {
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) throw new Error('Artifact text limit must be a positive integer.')
  const handle = await open(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
  try {
    const info = await handle.stat()
    if (!info.isFile()) throw new Error('Artifact path is not a regular file.')
    if (info.size > maxBytes) throw new Error(`Artifact exceeds the ${maxBytes}-byte IPC text limit.`)

    const chunks: Buffer[] = []
    let total = 0
    while (true) {
      const chunk = Buffer.allocUnsafe(Math.min(64 * 1024, maxBytes - total + 1))
      const { bytesRead } = await handle.read(chunk, 0, chunk.byteLength, null)
      if (!bytesRead) break
      total += bytesRead
      if (total > maxBytes) throw new Error(`Artifact exceeds the ${maxBytes}-byte IPC text limit.`)
      chunks.push(chunk.subarray(0, bytesRead))
    }
    return Buffer.concat(chunks, total).toString('utf8')
  } finally {
    await handle.close()
  }
}
