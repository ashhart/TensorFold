import { completionEvaluationRequest, evaluateCompletion, type CompletionPolicy, type CompletionVerdict } from '../../agent/automation-scheduler'
import { streamChat } from '../../agent/openai'
import { ensureTensorFoldStudioStylePrompt } from '../../agent/prompt-style'
import type { Automation, AutomationRun, Provider, ReasoningEffortSetting } from '../../agent/types'

function completionPolicyOf(a: Automation): CompletionPolicy {
  const p = a.completionPolicy
  if (!p) return { type: 'none', version: 0 }
  return p
}

// Ask a fresh READ-ONLY model call to judge whether the run met its objective, using the
// run's own output as the transcript excerpt (VST-038). Only 'ai_evaluated' policies call
// a model; the rest decide deterministically. A model/parse failure degrades to
// needs_review — never a false "complete".
export async function evaluateRunCompletion(a: Automation, run: AutomationRun, provider: Provider | null, model: string | null, signal: AbortSignal, assertRequestAllowed?: () => void, reasoningEffort?: ReasoningEffortSetting): Promise<{ verdict: CompletionVerdict; reason: string } | undefined> {
  const policy = completionPolicyOf(a)
  if (policy.type === 'none') return undefined
  let aiVerdict: CompletionVerdict | undefined
  const request = completionEvaluationRequest(policy, (run.output || '').slice(-4000))
  if (request && provider && model) {
    try {
      assertRequestAllowed?.()
      let out = ''
      for await (const ch of streamChat({
        provider, baseUrl: provider.baseUrl, apiKey: provider.apiKey, model,
        messages: [
          { role: 'system', content: ensureTensorFoldStudioStylePrompt('You judge whether an automation met its objective. Reply with exactly one word.') },
          { role: 'user', content: request.prompt },
        ],
        temperature: 0, reasoningEffort, approvalMode: 'readonly', tools: [], signal, assertRequestAllowed,
      })) {
        if (ch.type === 'content') out += ch.text
      }
      const word = out.toLowerCase().match(/\b(complete|continue|needs_review|needs review)\b/)?.[1]
      aiVerdict = word === 'complete' ? 'complete' : word === 'continue' ? 'continue' : word ? 'needs_review' : undefined
    } catch {
      aiVerdict = undefined // fall through to the kernel, which yields needs_review below
    }
  }
  const result = evaluateCompletion(policy, {
    iterations: 1,
    signals: [],
    aiVerdict: policy.type === 'ai_evaluated' ? (aiVerdict ?? 'needs_review') : undefined,
    runPolicyVersion: run.permissions?.completionPolicyVersion,
  })
  return { verdict: result.verdict, reason: result.reason }
}

