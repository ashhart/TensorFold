import { parentPort, workerData } from 'node:worker_threads'
import { createRequire } from 'node:module'
import type { KokoroSynthesisInput } from '../agent/types/kokoro'

// Keep ONNX and the phonemizer off the Electron main thread. This is a Node worker,
// not an Electron subprocess, so packaged builds do not need the runAsNode fuse.
const requireLocal = createRequire(__filename)
const { KokoroTTS } = requireLocal('kokoro-js') as typeof import('kokoro-js')
const { env } = createRequire(requireLocal.resolve('kokoro-js'))('@huggingface/transformers') as typeof import('@huggingface/transformers')
const { phonemize } = createRequire(requireLocal.resolve('kokoro-js'))('phonemizer') as { phonemize: (text: string, language: string) => Promise<string[]> }
env.allowRemoteModels = false
env.useFSCache = false
env.useBrowserCache = false
globalThis.fetch = async () => { throw new Error('Kokoro synthesis does not use the network.') }

let engine: Awaited<ReturnType<typeof KokoroTTS.from_pretrained>> | undefined

export function splitKokoroText(text: string): string[] {
  const words = text.trim().split(/\s+/)
  const chunks: string[] = []
  let chunk = ''
  for (const word of words) {
    if (word.length > 280) throw new Error('A word is too long to read. Shorten the text and try again.')
    if (chunk && chunk.length + word.length + 1 > 280) { chunks.push(chunk); chunk = '' }
    chunk += `${chunk ? ' ' : ''}${word}`
    if (chunk.length > 120 && /[.!?]$/.test(word)) { chunks.push(chunk); chunk = '' }
  }
  if (chunk) chunks.push(chunk)
  return chunks
}

export function encodeKokoroWav(chunks: Float32Array[]): Buffer {
  const samples = chunks.reduce((sum, chunk) => sum + chunk.length, 0)
  if (!samples || samples > 12_000_000) throw new Error('The generated speech is too long.')
  const wav = Buffer.alloc(44 + samples * 2)
  wav.write('RIFF', 0); wav.writeUInt32LE(wav.length - 8, 4); wav.write('WAVEfmt ', 8)
  wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22)
  wav.writeUInt32LE(24000, 24); wav.writeUInt32LE(48000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34)
  wav.write('data', 36); wav.writeUInt32LE(samples * 2, 40)
  let offset = 44
  for (const chunk of chunks) for (const sample of chunk) {
    if (!Number.isFinite(sample)) throw new Error('Kokoro returned invalid audio.')
    wav.writeInt16LE(Math.round(Math.max(-1, Math.min(1, sample)) * 32767), offset); offset += 2
  }
  return wav
}

const acknowledgements = new Map<string, () => void>()
parentPort?.on('message', async (input: KokoroSynthesisInput & { stream?: boolean; ack?: number }) => {
  if (typeof input.ack === 'number') { acknowledgements.get(`${input.requestId}:${input.ack}`)?.(); return }
  try {
    engine ??= await KokoroTTS.from_pretrained(workerData.modelDirectory, { dtype: 'q8', device: 'cpu' })
    const audio: Float32Array[] = []
    let index = 0
    for (const text of splitKokoroText(input.text)) {
      const phonemes = (await phonemize(text, 'en-us')).join(' ')
      const { input_ids } = engine.tokenizer(phonemes, { truncation: false })
      if (input_ids.dims.at(-1)! > 511) throw new Error('A passage is too long to read. Shorten the text and try again.')
      const output = await engine.generate_from_ids(input_ids, { voice: input.voice as 'af_heart', speed: input.rate })
      const samples = new Float32Array(output.audio)
      if (input.stream) {
        const wav = encodeKokoroWav([samples])
        await new Promise<void>((resolve) => {
          const key = `${input.requestId}:${index}`
          acknowledgements.set(key, () => { acknowledgements.delete(key); resolve() })
          parentPort?.postMessage({ requestId: input.requestId, chunk: true, index, ok: true, audioBase64: wav.toString('base64'), mimeType: 'audio/wav' })
        })
        index++
      } else audio.push(samples)
    }
    if (input.stream) { parentPort?.postMessage({ requestId: input.requestId, ok: true, mimeType: 'audio/wav' }); return }
    const wav = encodeKokoroWav(audio)
    parentPort?.postMessage({ requestId: input.requestId, ok: true, audioBase64: wav.toString('base64'), mimeType: 'audio/wav' })
  } catch (error) {
    // Never return model internals, absolute filesystem paths or speech text to logs.
    const message = error instanceof Error && /^(A word|A passage|The generated speech) is too long/.test(error.message) ? error.message : 'Kokoro could not generate speech. Retry, or reinstall its model in Voice settings.'
    parentPort?.postMessage({ requestId: input.requestId, ok: false, error: message })
  }
})
