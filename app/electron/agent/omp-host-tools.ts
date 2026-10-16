// OMP host-tools bridge (VST-050 slice 2): when a thread runs on the omp engine, the
// engine's own mutating tools are disabled and TensorFold's write/edit/patch/run tools are
// registered as omp HOST TOOLS instead. omp's model plans; every mutation executes
// through TensorFold's approval gate and tool executor — the same policy spine the native
// loop uses. Protocol facts (installed omp source, src/modes/rpc/rpc-types.ts):
// - host registers tools via {type:"set_host_tools", tools:[{name,description,parameters}]}
// - omp requests execution via a stdout frame {type:"host_tool_call", id, toolCallId, toolName, arguments}
// - the host answers with {type:"host_tool_result", id, result:{content:[{type:"text",text}]}, isError?}
// - {type:"host_tool_cancel", id, targetId} aborts a pending call.

import { randomUUID } from 'node:crypto'
import { approvalPolicyScope, requiresSavedApprovalPolicy } from './approval-policy'
import { riskBandForClassification, shouldAutoAllowInFullAuto } from './approval-risk'
import { toolApprovalReasonForMode } from './loop'
import { TOOL_SCHEMAS } from './tools'
import type { ApprovalMode } from './types'

/** TensorFold tools exposed to the engine. omp's own write/edit/bash stay disabled while bridged. */
export const OMP_HOST_TOOL_NAMES = ['write_file', 'edit_file', 'apply_patch', 'run_command'] as const

export interface OmpHostToolDefinition {
  name: string
  label?: string
  description: string
  parameters: Record<string, unknown>
}

export function ompHostToolDefinitions(): OmpHostToolDefinition[] {
  return OMP_HOST_TOOL_NAMES.flatMap((name) => {
    const entry = TOOL_SCHEMAS.find((candidate: any) => candidate.function?.name === name)
    const fn = entry?.function as { name?: string; label?: string; description?: string; parameters?: unknown } | undefined
    if (!fn?.name) return []
    return [{
      name,
      label: fn.label ?? name,
      description: String(fn.description ?? name).slice(0, 2000),
      parameters: (fn.parameters && typeof fn.parameters === 'object'
        ? fn.parameters
        : { type: 'object', properties: {} }) as Record<string, unknown>,
    }]
  })
}

export interface OmpHostToolCall {
  id: string
  toolCallId: string
  toolName: string
  args: Record<string, unknown>
}

export interface OmpHostToolResultFrame {
  type: 'host_tool_result'
  id: string
  result: { content: Array<{ type: 'text'; text: string }> }
  isError?: boolean
}

export interface OmpHostToolGateDeps {
  approvalMode: ApprovalMode
  /** The host approval resolver; absent (tests/embedders) means non-read calls deny. */
  requestApproval?: (callId: string) => Promise<'allow' | 'allow_always' | 'deny'>
  executeTool: (name: string, args: Record<string, unknown>) => Promise<{ ok: boolean; output: string }>
  emit: (events: Array<Record<string, unknown>>) => void
  signal: AbortSignal
  /** Timeline item id omp's tool_execution_start already created for this call, if any. */
  itemId?: string
  callIdPrefix?: string
}

/**
 * Runs one engine-requested tool call through TensorFold's approval gate, mirroring the
 * native loop's decision order: read-only auto-run, readonly-mode deny, saved-policy
 * requirement for package installs, Full Auto risk bands, interactive prompt otherwise.
 * Emits timeline patches through deps.emit and returns the omp result frame.
 */
export async function executeOmpHostTool(call: OmpHostToolCall, deps: OmpHostToolGateDeps): Promise<OmpHostToolResultFrame> {
  const name = call.toolName
  const args = (call.args && typeof call.args === 'object' ? call.args : {}) as Record<string, any>
  // The approval resolver reads the timeline item to scope saved policies. The RPC
  // host call is authoritative, even when tool_execution_start supplied no args.
  if (deps.itemId) deps.emit([{
    type: 'item_update',
    id: deps.itemId,
    patch: {
      tool: name,
      args,
      ...(name === 'run_command' && typeof args.command === 'string' ? { command: args.command } : {}),
      ...(typeof args.path === 'string' ? { path: args.path } : {}),
    },
  }])
  const patch = (state: string, output: string, approval?: string) => {
    if (!deps.itemId) return
    deps.emit([{ type: 'item_update', id: deps.itemId, patch: { state, output, ...(approval ? { approval } : {}) } }])
  }
  const deny = (output: string) => {
    patch('denied', output, output)
    return frameFor(call.id, true, `${output} Stop and ask how to proceed.`)
  }

  const approvalReason = toolApprovalReasonForMode(name, args, deps.approvalMode)
  const policy = approvalPolicyScope('tool', name, args)

  if (deps.approvalMode === 'readonly') return deny('Denied by read-only approval mode.')

  // Full Auto risk bands: low-risk classifications auto-run with an explanation,
  // exactly like the native loop's gate.
  if (deps.approvalMode === 'auto') {
    const kind: 'command' | 'connector' | 'tool' = name === 'run_command' ? 'command' : policy.scope === 'connector' ? 'connector' : 'tool'
    const risk = riskBandForClassification(policy.classification, kind)
    if (shouldAutoAllowInFullAuto(risk.band, 'low', false)) {
      return await run(call, deps, `Auto-approved by Full Auto (${risk.reason}). ${policy.explanation}`, patch)
    }
  }

  const needsPrompt = approvalReason !== null || deps.approvalMode === 'suggest' || requiresSavedApprovalPolicy(policy)
  if (!needsPrompt) return await run(call, deps, undefined, patch)
  if (!deps.requestApproval) return deny('Denied: this engine tool needs an approval decision, but no approval channel is connected.')

  const callId = `${deps.callIdPrefix ?? 'omp'}-${call.id}`
  if (deps.signal.aborted) return deny('Cancelled before execution.')
  if (requiresSavedApprovalPolicy(policy)) {
    patch('awaiting', 'Package install commands require an active saved package-install policy. Choose Always to allow this exact command signature.')
  } else {
    patch('awaiting', approvalReason ?? 'This engine tool needs approval.')
  }
  deps.emit([{ type: 'status', status: 'awaiting' }, { type: 'approval_request', callId, itemId: deps.itemId ?? '' }])
  const decision = await deps.requestApproval(callId)
  if (decision === 'deny') return deny('Denied by user.')
  if (requiresSavedApprovalPolicy(policy) && decision === 'allow') {
    return deny('Package install was not run. Choose Always to create an active saved package-install policy for this exact command signature.')
  }
  const approvalNote = decision === 'allow_always' ? `Approved and saved as a policy. ${policy.explanation}` : `Approved via saved policy. ${policy.explanation}`
  return await run(call, deps, approvalNote, patch)
}

async function run(call: OmpHostToolCall, deps: OmpHostToolGateDeps, approvalNote: string | undefined, patch: (state: string, output: string, approval?: string) => void): Promise<OmpHostToolResultFrame> {
  patch('running', '', approvalNote)
  if (deps.signal.aborted) {
    patch('error', 'Cancelled before execution.')
    return frameFor(call.id, true, 'Cancelled before execution.')
  }
  try {
    const result = await deps.executeTool(call.toolName, (call.args && typeof call.args === 'object' ? call.args : {}) as Record<string, unknown>)
    patch(result.ok ? 'ok' : 'error', result.output)
    return frameFor(call.id, !result.ok, result.output)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    patch('error', message)
    return frameFor(call.id, true, message)
  }
}

function frameFor(id: string, isError: boolean, text: string): OmpHostToolResultFrame {
  return { type: 'host_tool_result', id, result: { content: [{ type: 'text', text }] }, ...(isError ? { isError: true } : {}) }
}

/** Test/embedder helper: an id factory mirroring the engine's host-call ids. */
export function newHostCallId(): string {
  return randomUUID()
}
