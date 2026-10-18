import { type ChatParams, type StreamChunk } from '../openai'
import type { Provider } from '../types'
import type { ModelCheck, ModelCheckResult } from '../../../shared/model-fit'
import { qualificationImage } from './fixture-image'
import { qualificationStream } from './transport'

export type QualificationStream = (params: ChatParams) => AsyncGenerator<StreamChunk>
const TOKEN = 'tensorfold_fixed_probe_1'
const TOOL = 'tensorfold_qualification_echo'
const FIXTURE_TOOL = { type: 'function', function: { name: TOOL, description: 'Return the given fixed value. This is a synthetic check and will never be executed.', parameters: { type: 'object', properties: { value: { type: 'string', enum: [TOKEN] } }, required: ['value'], additionalProperties: false } } }
export async function runQualificationProbe(provider: Provider, model: string, check: ModelCheck, options: { signal: AbortSignal; assertCurrent: () => void; stream?: QualificationStream; timeoutMs?: number; now?: () => number }): Promise<ModelCheckResult> {
  const now = options.now ?? Date.now
  const started = now()
  if (provider.transport && provider.transport !== 'chat-completions') return { state: 'unsupported', message: 'This fixed-fixture check requires the HTTP chat transport; delegated engine actions are never launched by a probe.' }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(new Error('Probe deadline reached')), options.timeoutMs ?? 8_000)
  const signal = AbortSignal.any([options.signal, controller.signal])
  const prompt = check === 'tools' ? `Call ${TOOL} once with value ${TOKEN}. Do not perform any other action.` : check === 'reasoning' ? 'What is 2 + 2? Reply briefly.' : check === 'vision' ? 'What color is this square? Reply only red, green, or blue.' : 'Reply with OK.'
  const messages: ChatParams['messages'] = [{ role: 'user', content: check === 'vision' ? [{ type: 'text', text: prompt }, { type: 'image_url', image_url: { url: qualificationImage() } }] : prompt }]
  let iterator: AsyncGenerator<StreamChunk> | null = null
  let content = '', reasoning = '', chunks = 0
  let calls: Extract<StreamChunk, { type: 'tool_calls' }>['toolCalls'] = []
  let cancelledForCheck = false
  let timeoutAbort: (() => void) | null = null
  const interrupted = new Promise<never>((_, reject) => { timeoutAbort = () => reject(new Error('Probe interrupted')); signal.addEventListener('abort', timeoutAbort, { once: true }); if (signal.aborted) timeoutAbort() })
  // The rejected cancellation branch remains observed even if iteration finishes first.
  void interrupted.catch(() => {})
  let drained = true
  try {
    options.assertCurrent()
    if (signal.aborted) return { state: 'cancelled', message: 'The check was cancelled before dispatch.' }
    iterator = (options.stream ?? qualificationStream)({ provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model, messages, tools: check === 'tools' ? [FIXTURE_TOOL] : [], temperature: 0, reasoningEffort: check === 'reasoning' ? 'low' : 'none', approvalMode: 'readonly', signal, assertRequestAllowed: options.assertCurrent })
    while (true) {
      const next = await Promise.race([iterator.next(), interrupted])
      if (next.done) break
      const chunk = next.value
      if (++chunks > 256) throw new Error('Probe response bound reached')
      if (chunk.type === 'content') content += chunk.text
      if (chunk.type === 'reasoning') reasoning += chunk.text
      if (chunk.type === 'tool_calls') calls = chunk.toolCalls
      if (content.length + reasoning.length > 8_192 || calls.length > 4) throw new Error('Probe response bound reached')
      if (check === 'cancellation' && content.trim()) { cancelledForCheck = true; controller.abort(); break }
    }
    options.assertCurrent()
    const elapsedMs = Math.max(0, now() - started)
    if (check === 'cancellation') {
      if (!cancelledForCheck) return { state: 'failed', message: 'No visible output arrived before the cancellation check could be performed.', elapsedMs }
      drained = await closeIterator(iterator)
      return { state: drained ? 'observed' : 'failed', message: drained ? 'The client stream closed after abort; provider-side compute termination is unverified.' : 'The client iterator did not confirm closure after abort; provider-side work is unverified.', elapsedMs, transportDrained: drained, providerComputeStopped: false }
    }
    if (check === 'tools') {
      if (calls.length !== 1 || calls[0].function.name !== TOOL) return { state: 'failed', message: 'The model did not return exactly the requested synthetic tool call; no tools were executed.', elapsedMs }
      try {
        const args = JSON.parse(calls[0].function.arguments)
        if (!args || args.value !== TOKEN || Object.keys(args).length !== 1) throw new Error('Wrong fixed arguments')
      } catch { return { state: 'failed', message: 'The synthetic tool call contained malformed or schema-invalid arguments; no tools were executed.', elapsedMs } }
      return { state: 'observed', message: 'One correctly named synthetic tool call with valid fixed arguments was observed; no tool was executed.', elapsedMs }
    }
    if (calls.length) return { state: 'failed', message: 'The model requested tools during a tool-free fixture; no tools were executed.', elapsedMs }
    if (check === 'vision') return { state: /^red[.!\s]*$/i.test(content.trim()) ? 'observed' : 'failed', message: /^red[.!\s]*$/i.test(content.trim()) ? 'The model correctly named the original red square in this one synthetic image.' : 'The response did not identify the fixed red square; image capability was not qualified.', elapsedMs }
    if (check === 'reasoning') return { state: reasoning.trim() ? 'observed' : 'unknown', message: reasoning.trim() ? 'A separate reasoning channel was observed; its correctness and quality were not assessed.' : 'No separate reasoning channel was observed; hidden reasoning cannot be inferred from an answer.', elapsedMs }
    return { state: content.trim() ? 'observed' : 'failed', message: content.trim() ? 'A visible text response was observed using the fixed prompt.' : 'The endpoint ended without visible text output.', elapsedMs }
  } catch {
    return { state: options.signal.aborted ? 'cancelled' : 'failed', message: options.signal.aborted ? 'The requested check was cancelled; any provider-side compute termination is unverified.' : controller.signal.aborted ? 'The fixed-fixture deadline was reached; provider-side work and acceptance are unverified.' : 'The fixed-fixture response failed or exceeded its bounds; no response body or conversation content was saved.', elapsedMs: Math.max(0, now() - started) }
  } finally {
    clearTimeout(timer)
    if (timeoutAbort) signal.removeEventListener('abort', timeoutAbort)
    controller.abort()
    if (iterator && !cancelledForCheck) await closeIterator(iterator)
  }
}
async function closeIterator(iterator: AsyncGenerator<StreamChunk>): Promise<boolean> {
  let timer: ReturnType<typeof setTimeout> | null = null
  try { return await Promise.race([iterator.return(undefined as never).then(() => true, () => false), new Promise<boolean>(resolve => { timer = setTimeout(() => resolve(false), 250) })]) }
  finally { if (timer) clearTimeout(timer) }
}
