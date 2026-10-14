import { mkdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { MANAGED_OUTPUT_REF_PREFIX } from '../../agent/context-governance'
import { overwriteAndUnlink, readProtectedJson, writeProtectedJson } from '../../store'

const SAFE_OUTPUT_NAME = /^[a-zA-Z0-9_-]+\.txt$/

export function createManagedOutputStore(root: string) {
  const outputRoot = path.resolve(root)

  async function write(suggestedRef: string, content: string): Promise<string> {
    const name = managedOutputName(suggestedRef)
    if (!name) throw new Error('invalid managed output reference')
    await mkdir(outputRoot, { recursive: true, mode: 0o700 })
    // Managed spills can contain full file contents, command output, or tool errors. Store the
    // value in the same app-key protected envelope as threads instead of a misleading plaintext
    // .txt duplicate under ~/.vontra/outputs.
    await writeProtectedJson(path.join(outputRoot, name), content)
    return `${MANAGED_OUTPUT_REF_PREFIX}${name}`
  }

  async function read(ref: string): Promise<string | null> {
    const name = managedOutputName(ref)
    if (!name) return null
    const file = path.join(outputRoot, name)
    let raw: string
    try { raw = await readFile(file, 'utf8') } catch { return null }
    let protectedEnvelope = false
    try {
      const parsed = JSON.parse(raw)
      protectedEnvelope = !!parsed && typeof parsed === 'object' && typeof parsed.encrypted === 'string' && parsed.encrypted.startsWith('safe-json:v1:')
    } catch { /* legacy plaintext spill */ }
    if (protectedEnvelope) return readProtectedJson<string | null>(file, null, false)

    // One-time migration for spills written by older builds. Securely remove the plaintext inode
    // before creating the encrypted replacement so the live file never remains a raw secret copy.
    await overwriteAndUnlink(file)
    await writeProtectedJson(file, raw)
    return raw
  }

  return { write, read }
}

function managedOutputName(ref: string): string | null {
  if (!ref.startsWith(MANAGED_OUTPUT_REF_PREFIX)) return null
  const name = ref.slice(MANAGED_OUTPUT_REF_PREFIX.length)
  return SAFE_OUTPUT_NAME.test(name) ? name : null
}
