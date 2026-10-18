import { streamChat, type ChatParams, type StreamChunk } from '../openai'
import { engineRoute } from '../engine/provider'

/** A check can borrow an already-loaded native endpoint, never start or switch one. */
export async function* qualificationStream(params: ChatParams): AsyncGenerator<StreamChunk> {
  const native = await engineRoute(params, { start: false })
  if (!native) { yield* streamChat(params); return }
  try {
    params.assertRequestAllowed?.()
    // The native lease already resolved the exact endpoint. Mark this private copy as a
    // custom local API route so streamChat cannot reacquire its automatic-start hook.
    const provider = { ...native.params.provider!, source: 'custom' as const }
    yield* streamChat({ ...native.params, provider })
  } finally { native.release?.() }
}
