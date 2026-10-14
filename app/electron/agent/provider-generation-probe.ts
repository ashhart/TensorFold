import { streamChat, type ChatParams, type StreamChunk } from './openai'
import type { Provider } from './types'

export interface ProviderGenerationProbe {
  model: string
  firstTokenMs: number
  elapsedMs: number
  transport: 'chat-completions'
}

// Remote error bodies can echo credentials; only return our fixed diagnostics
// and HTTP status, never untrusted provider text, through this setup IPC.
export function providerProbeFailure(error: unknown): string {
  const message = error instanceof Error ? error.message : ''
  const status = /\bHTTP\s+([1-5][0-9]{2})\b/.exec(message)?.[1]
  if (status) return `The model response test returned HTTP ${status}. Check the selected model, endpoint and account.`
  if (/within 20 seconds|timed?\s*out|timeout/i.test(message)) return 'The model did not respond within 20 seconds. Check the server or choose a smaller model.'
  if (/without a visible response/.test(message)) return 'The endpoint finished without a visible response; model listing alone does not prove chat is working.'
  if (/requested tools/.test(message)) return 'The endpoint requested tools during a tool-free response test.'
  if (/HTTP model endpoints/.test(message)) return 'Response testing is available for HTTP model endpoints; use CLI sign-in checks for this provider.'
  if (/valid model/.test(message)) return 'Select a valid model before testing a response.'
  return 'The model response test failed. Check the endpoint, selected model and account.'
}

// The test sends a fixed, tool-free prompt through the actual chat transport and
// stops on the first visible output; it never reads workspace or conversation data.
export async function probeProviderGeneration(provider: Provider, model: string, options: {
  stream?: (params: ChatParams) => AsyncGenerator<StreamChunk>
  timeoutMs?: number
  now?: () => number
} = {}): Promise<ProviderGenerationProbe> {
  if (provider.transport && provider.transport !== 'chat-completions') throw new Error('Response testing is available for HTTP model endpoints; use CLI sign-in checks for this provider.')
  if (typeof model !== 'string' || !model.trim() || model.length > 300 || /[\u0000-\u001f]/.test(model)) throw new Error('Select a valid model before testing a response.')
  const now = options.now ?? Date.now
  const startedAt = now()
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(new Error('The model did not respond within 20 seconds.')), options.timeoutMs ?? 20_000)
  try {
    for await (const chunk of (options.stream ?? streamChat)({
      provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model: model.trim(),
      messages: [{ role: 'user', content: 'Reply with OK.' }], tools: [],
      approvalMode: 'readonly', signal: controller.signal,
    })) {
      if (controller.signal.aborted) throw controller.signal.reason
      if (chunk.type === 'content' && chunk.text.trim()) {
        const firstTokenMs = Math.max(0, now() - startedAt)
        return { model: model.trim(), transport: 'chat-completions', firstTokenMs, elapsedMs: firstTokenMs }
      }
      if (chunk.type === 'tool_calls' && chunk.toolCalls.length) throw new Error('The endpoint requested tools during a tool-free response test.')
    }
    throw controller.signal.aborted ? controller.signal.reason : new Error('The endpoint finished without a visible response; model listing alone does not prove chat is working.')
  } finally {
    clearTimeout(timeout)
    controller.abort()
  }
}
