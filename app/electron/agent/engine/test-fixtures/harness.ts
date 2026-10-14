import { promises as fs } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import type { EngineBinary } from '../paths'

// Shared setup for the engine tests: a private Hugging Face cache and the fake engine as the binary.

export const FAKE_ENGINE = path.join(__dirname, 'fake-engine.mjs')

export interface Harness {
  root: string
  hub: string
  binary: EngineBinary
  env: NodeJS.ProcessEnv
  recordFile: string
  records(): Promise<any[]>
  cleanup(): Promise<void>
}

export async function makeHarness(extraEnv: NodeJS.ProcessEnv = {}): Promise<Harness> {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'tf-engine-test-'))
  const hub = path.join(root, 'hub')
  await fs.mkdir(hub, { recursive: true })
  const recordFile = path.join(root, 'records.jsonl')
  return {
    root,
    hub,
    binary: { command: process.execPath, args: [FAKE_ENGINE] },
    env: { HOME: root, PATH: process.env.PATH, HF_HUB_CACHE: hub, FAKE_ENGINE_RECORD: recordFile, ...extraEnv },
    recordFile,
    async records() {
      try {
        return (await fs.readFile(recordFile, 'utf8')).split('\n').filter(Boolean).map((line) => JSON.parse(line))
      } catch { return [] }
    },
    cleanup: () => fs.rm(root, { recursive: true, force: true }),
  }
}

// A cached checkpoint as pull leaves it: blobs, a snapshot of links, refs/main.
export async function seedModel(hub: string, repo: string, opts: { modelType?: string; bytes?: number; dflash?: boolean } = {}): Promise<void> {
  const [owner, name] = repo.split('/')
  const dir = path.join(hub, `models--${owner}--${name}`)
  const sha = 'a'.repeat(40)
  const snapshot = path.join(dir, 'snapshots', sha)
  await fs.mkdir(path.join(dir, 'blobs'), { recursive: true })
  await fs.mkdir(snapshot, { recursive: true })
  await fs.mkdir(path.join(dir, 'refs'), { recursive: true })
  await fs.writeFile(path.join(dir, 'blobs', 'weights'), Buffer.alloc(opts.bytes ?? 8192, 1))
  await fs.symlink(path.join(dir, 'blobs', 'weights'), path.join(snapshot, 'model.safetensors'))
  const config = opts.dflash ? { model_type: 'qwen3', dflash_config: {} } : { model_type: opts.modelType ?? 'qwen3_5', quantization: { bits: 4, group_size: 64 }, max_position_embeddings: 32768 }
  await fs.writeFile(path.join(snapshot, 'config.json'), JSON.stringify(config))
  await fs.writeFile(path.join(dir, 'refs', 'main'), sha)
}

export async function waitFor<T>(check: () => T | Promise<T>, timeoutMs = 10_000, label = 'condition'): Promise<NonNullable<T>> {
  const deadline = Date.now() + timeoutMs
  for (;;) {
    const value = await check()
    if (value) return value as NonNullable<T>
    if (Date.now() > deadline) throw new Error(`Timed out waiting for ${label}.`)
    await new Promise((resolve) => setTimeout(resolve, 25))
  }
}

export function processAlive(pid: number): boolean {
  try { process.kill(pid, 0); return true } catch { return false }
}
