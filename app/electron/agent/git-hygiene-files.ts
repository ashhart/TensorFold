import { constants, promises as fs } from 'node:fs'
import { canonicalWorkspacePath } from './workspace-path-policy'
import { readOpenedTextFile, withFileReadBudget } from './bounded-file-read'

export async function readHygieneFile(root: string, file: string): Promise<Buffer> {
  return withFileReadBudget({}, async (budget) => {
    const resolved = await budget.wait(canonicalWorkspacePath(root, file, 'write'))
    const opening = fs.open(resolved, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK)
    opening.then((handle) => { if (budget.signal.aborted) void handle.close().catch(() => {}) }, () => {})
    const handle = await budget.wait(opening)
    return readOpenedTextFile({
      stat: async () => {
        const opened = await handle.stat()
        const current = await canonicalWorkspacePath(root, file, 'write')
        const now = await fs.stat(current)
        if (current !== resolved || opened.dev !== now.dev || opened.ino !== now.ino) {
          throw new Error('Hygiene file changed during open.')
        }
        return { isFile: opened.isFile(), size: opened.size }
      },
      read: (buffer, length, position) => handle.read(buffer, 0, length, position).then((value) => value.bytesRead),
      close: () => handle.close(),
    }, budget) as Promise<Buffer>
  })
}
