// Approve and execute a tool turn, preserving cancellation evidence and source result order.

import { createHash, randomUUID } from 'node:crypto'
import { assertThreadChildSupported } from '../../shared/thread-child-policy'
import { botToolBlockReason } from '../../shared/bot-tool-policy'
import { outboundActionPreview } from './connector-action-safety'
import { toolLabel, isReadOnlyTool } from './tools'
import { webFetchNeedsApproval } from './web-fetch-vetting'
import { coerceToolArgs } from './tool-arg-coercion'
import { delegateTasks } from './delegate-task'
import { parseOracleArgs } from './oracle'
import { executeToolExecutionBatches } from './tool-execution-plan'
import { normalizePlanSteps, planProgress, renderPlanTree, TODO_MUTATING_TOOLS } from './todo-tracker'
import { contextEpoch, estimateTextTokens, toolOutputSpillPath, TOOL_OUTPUT_MAX_CHARS } from './context-governance'
import { compressToolOutput, rememberPreparedToolOutput } from './tool-output-compression'
import {
  approvalPolicyScope,
  commandClass,
  externalEffectApprovalReason,
  requiresSavedApprovalPolicy,
} from './approval-policy'
import { fullAccessAutoAllows, riskBandForClassification } from './approval-risk'
import { recordToolCalls } from './turn-stats'
import { type ApiMessage, type TimelineItem, type ToolCall } from './types'
import { reconcileOrphanedToolCalls, cancelledToolMutationPatch, wrapUntrustedToolOutput } from './run-loop-history'
import {
  THRASH_LIMIT,
  repeatGuardNote,
  toolCallSignature,
  searchLoopNote,
  toolCadenceNote,
  NO_PROGRESS_TOOLS,
  noProgressNote,
  isTruncatedToolCallArgs,
} from './run-loop-guards'
import {
  MAX_PARALLEL_READ_TOOLS,
  PARALLEL_READ_ONLY_TOOLS,
  executeRuntimeTool,
  toolApprovalReasonForMode,
} from './run-loop-tool-policy'
import {
  delegateChildToolSchemas,
  runDelegateChild,
  runExplore,
  runOracle,
  DELEGATE_TASK_MAX_CONCURRENT,
  parseDelegateTaskArgs,
  formatDelegateTaskResults,
} from './run-loop-auxiliary'
import { type RunSession } from './run-loop-session'

export async function runToolTurn(session: RunSession, step: number, toolCalls: ToolCall[], finishReason: string | null): Promise<void> {
  const {
    deps, state, thread, provider, model, signal, persist, capturedChildBoundary, assertRequestAllowed, runId,
    cwd, callbacksAllowed, pluginRuntime, send, recovery, failCounts, repeatCounts, toolAttemptCounts,
    refreshableActionSignatures, mutatingActionSignatures, toolNameCounts, resultHashCounts, guardPolicy,
    localActionGuardEnabled, localRepeatLimit, emitGuardWarning, deadlineAt, builtinToolSchemas,
    modelToolSchemas, turnStats,
  } = session
  recordToolCalls(turnStats, toolCalls.length)
  send({ type: 'status', status: 'tool' })

  const turnSchemas = modelToolSchemas()
  const turnBuiltinToolNames = new Set<string>(builtinToolSchemas().map((schema: any) => schema.function?.name).filter(Boolean))
  const toolResults = new Map<string, ApiMessage>()
  const recordToolResult = (call: ToolCall, name: string, content: string, prepared = false) => {
    const message: ApiMessage = { role: 'tool', tool_call_id: call.id, name, content }
    toolResults.set(call.id, prepared ? rememberPreparedToolOutput(message) : message)
  }
  const executeRequestedCall = async (call: ToolCall): Promise<void> => {
    if (signal.aborted) return
    recovery.begin(`tool:${call.id}`)
    let dispatchedMutation = false
    let toolConfirmed = false
    send({ type: 'lifecycle', phase: 'tool_start', ts: Date.now(), step, callId: call.id, tool: call.function.name })
    try {
    const matchedSchema = turnSchemas.find((schema: any) => schema.function?.name === call.function.name)
    if (!matchedSchema || deps.toolContext?.allowedToolNames && !deps.toolContext.allowedToolNames.includes(call.function.name)) {
      const output = `Tool ${call.function.name} is not available in this run's executable capability policy.`
      send({ type: 'item_add', item: { id: randomUUID(), kind: 'tool', tool: call.function.name, title: 'Tool blocked', state: 'denied', output, createdAt: Date.now() } })
      recordToolResult(call, call.function.name, output)
      return
    }
    let args: Record<string, any> = {}
    let argsParseFailed = false
    try { args = JSON.parse(call.function.arguments || '{}') } catch { args = {}; argsParseFailed = true }

    if (!argsParseFailed) {
      if (matchedSchema?.function?.parameters) args = coerceToolArgs(args, matchedSchema.function.parameters as Record<string, unknown>)
    }
    const botScopeBlock = botToolBlockReason(deps.toolContext?.currentThread?.bot ?? thread.bot, call.function.name, args, deps.toolContext?.allowedToolNames)
    if (botScopeBlock) {
      send({ type: 'item_add', item: { id: randomUUID(), kind: 'tool', tool: call.function.name, title: 'Tool blocked', state: 'denied', output: botScopeBlock, createdAt: Date.now() } })
      recordToolResult(call, call.function.name, botScopeBlock)
      return
    }
    if (['explore', 'delegate_task', 'oracle', 'launch_subagent', 'launch_subagents'].includes(call.function.name)) {
      try { assertThreadChildSupported(capturedChildBoundary); assertThreadChildSupported(thread) }
      catch (error) {
        const reason = error instanceof Error ? error.message : 'Secondary runtimes are unavailable for this conversation.'
        send({ type: 'item_add', item: { id: randomUUID(), kind: 'tool', tool: call.function.name, title: 'Tool blocked', state: 'denied', output: reason, createdAt: Date.now() } })
        recordToolResult(call, call.function.name, reason)
        return
      }
    }
    const { verb, target } = toolLabel(call.function.name, args)

    if (isTruncatedToolCallArgs(argsParseFailed, finishReason)) {
      const truncItem: TimelineItem = { id: randomUUID(), kind: 'tool', tool: call.function.name, title: verb, state: 'error', output: 'Response was truncated (hit the max output length) before this tool call’s arguments finished streaming, so they could not be parsed. The tool was not run.', createdAt: Date.now() }
      send({ type: 'item_add', item: truncItem })
      recordToolResult(call, call.function.name, 'Your previous response was truncated (hit the max output length) before this tool call’s arguments finished streaming, so they could not be parsed and the tool was NOT run. Retry with a shorter response, or break the task into smaller steps.')
      return
    }

    const isVisualization = call.function.name === 'visualize'
    const approvalId = randomUUID()
    const item: TimelineItem = {
      id: approvalId,
      kind: isVisualization ? 'visualization' : 'tool',
      callId: call.id,
      approvalId,
      tool: call.function.name,
      title: verb,
      path: isVisualization ? undefined : target,
      command: call.function.name === 'run_command' ? args.command : undefined,
      args,
      state: 'pending',
      createdAt: Date.now(),
    }
    send({ type: 'item_add', item })

    const sig = toolCallSignature(call.function.name, args)
    const priorAttempts = toolAttemptCounts.get(sig) ?? 0

    // Reserve attempts before approvals or hooks so parallel duplicate calls cannot bypass the cap.
    if (localActionGuardEnabled) {
      if (state.repeatedActionStop || priorAttempts >= localRepeatLimit || (failCounts.get(sig) ?? 0) >= THRASH_LIMIT) {
        state.repeatedActionStop = true
        state.budgetStop = 'repeated action limit'
        recovery.pause('repeated_action', 'The repeated-action limit was reached; completed work is preserved and no further tools will run.')
        const output = 'Not run: the model kept requesting the same action after reaching the repeat limit. The run is stopping with a final summary.'
        send({ type: 'item_update', id: item.id, patch: { state: 'error', output } })
        emitGuardWarning('repeated-action-stop', '[guardrail] Stopped the model from repeating the same action again. Wrapping up without more tools.')
        recordToolResult(call, call.function.name, output)
        return
      }
      toolAttemptCounts.set(sig, priorAttempts + 1)
      const isVerification = call.function.name === 'run_command'
        && ['read-only', 'git-read', 'test', 'build'].includes(commandClass(String(args.command ?? '')))
      if (!mutatingActionSignatures.has(sig) && (isReadOnlyTool(call.function.name, deps.toolContext?.mcpTools, deps.approvalMode) || isVerification)) {
        refreshableActionSignatures.add(sig)
      }
    }

    if (call.function.name === 'explore') {
      send({ type: 'item_update', id: item.id, patch: { state: 'running', title: 'Explore', path: undefined } })
      send({ type: 'status', status: 'tool' })
      let summary = ''
      try { summary = await runExplore(provider, model, cwd, String(args.query ?? args.task ?? ''), signal, deps.toolContext, assertRequestAllowed) }
      catch (e: any) { summary = `Explore failed: ${e?.message ?? e}` }

      if (signal.aborted) return
      send({ type: 'item_update', id: item.id, patch: { state: 'ok', title: 'Explore', output: summary } })
      recordToolResult(call, 'explore', wrapUntrustedToolOutput('explore', compressToolOutput({ tool: 'explore', output: summary, maxChars: 8_000, maxBytes: 24_000 }).content), true)
      return
    }

    if (call.function.name === 'delegate_task') {
      send({ type: 'item_update', id: item.id, patch: { state: 'running', title: 'Delegate task', path: undefined } })
      send({ type: 'status', status: 'tool' })
      const inputs = parseDelegateTaskArgs(args)
      let summary = ''
      if (!inputs.length) {
        summary = 'delegate_task requires at least one task.'
      } else {
        try {
          const results = await delegateTasks(inputs, {
            provider, model, cwd, signal,
            maxConcurrent: DELEGATE_TASK_MAX_CONCURRENT,
            runOneChild: async (input) => {
              try {
                const tools = delegateChildToolSchemas(input.toolNames)
                const result = await runDelegateChild(provider, model, cwd, input.task, signal, tools, input.maxSteps, deps.toolContext, assertRequestAllowed)
                return { result }
              } catch (e: any) {
                return { result: '', error: e?.message ?? String(e) }
              }
            },
          })
          summary = formatDelegateTaskResults(results)
        } catch (e: any) {
          summary = `delegate_task failed: ${e?.message ?? e}`
        }
      }

      if (signal.aborted) return
      send({ type: 'item_update', id: item.id, patch: { state: 'ok', title: 'Delegate task', output: summary } })
      recordToolResult(call, 'delegate_task', wrapUntrustedToolOutput('delegate_task', compressToolOutput({ tool: 'delegate_task', output: summary, maxChars: 8_000, maxBytes: 24_000 }).content), true)
      return
    }

    if (call.function.name === 'oracle') {
      send({ type: 'item_update', id: item.id, patch: { state: 'running', title: 'Oracle', path: undefined } })
      send({ type: 'status', status: 'tool' })
      const q = parseOracleArgs(args)
      let summary = ''
      if (!q) summary = 'oracle requires a question.'
      else { try { summary = await runOracle(provider, model, cwd, q, signal, deps.toolContext, assertRequestAllowed) } catch (e: any) { summary = `Oracle failed: ${e?.message ?? e}` } }
      if (signal.aborted) return
      send({ type: 'item_update', id: item.id, patch: { state: 'ok', title: 'Oracle', output: summary } })
      recordToolResult(call, 'oracle', wrapUntrustedToolOutput('oracle', compressToolOutput({ tool: 'oracle', output: summary, maxChars: 8_000, maxBytes: 24_000 }).content), true)
      return
    }

    if (call.function.name === 'update_plan') {
      const steps = normalizePlanSteps(args.steps)
      thread.plan = steps
      state.todoMutationsSincePlanTouch = 0
      if (thread.goal && !thread.goalStatus) thread.goalStatus = 'active'
      send({ type: 'plan', plan: steps, goalStatus: thread.goalStatus })
      const progress = planProgress(steps)
      send({ type: 'item_update', id: item.id, patch: { state: 'ok', title: 'Plan', path: undefined, output: renderPlanTree(steps), meta: `${progress.done}/${progress.total} done${progress.blocked ? ` · ${progress.blocked} blocked` : ''}` } })
      recordToolResult(call, 'update_plan', `Plan updated (${progress.done}/${progress.total} steps done${progress.blocked ? `, ${progress.blocked} blocked` : ''}).`)
      return
    }

    const unvettedFetch = call.function.name === 'web_fetch'
      && webFetchNeedsApproval(args, thread.timeline.filter((entry) => entry.kind === 'user').map((entry) => entry.text ?? ''))
    const isRead = isReadOnlyTool(call.function.name, deps.toolContext?.mcpTools, deps.approvalMode) && !unvettedFetch
    const approvalPolicy = approvalPolicyScope('tool', call.function.name, args)

    const approvalReason = toolApprovalReasonForMode(call.function.name, args, deps.approvalMode)
    const requiresSavedPolicy = requiresSavedApprovalPolicy(approvalPolicy)
    let decision: 'allow' | 'allow_always' | 'deny' = 'allow'
    let denialOutput = 'Denied by user.'
    let approvalOutput = ''
    if (!isRead) {
      if (deps.approvalMode === 'readonly') {
        decision = 'deny'
        denialOutput = unvettedFetch
          ? 'Denied by read-only approval mode: web_fetch only reaches hosts the user typed. Ask the user to share the link.'
          : 'Denied by read-only approval mode.'
        approvalOutput = denialOutput
      } else if (requiresSavedPolicy && deps.approvalMode !== 'auto') {
        send({ type: 'item_update', id: item.id, patch: { state: 'awaiting', output: 'Package install commands require an active saved package-install policy. Choose Always to allow this exact command signature.' } })
        send({ type: 'status', status: 'awaiting' })
        send({ type: 'approval_request', callId: approvalId, itemId: item.id })
        decision = await deps.requestApproval(approvalId)
        if (decision === 'allow') {
          decision = 'deny'
          denialOutput = 'Package install was not run. Choose Always to create an active saved package-install policy for this exact command signature.'
          approvalOutput = denialOutput
        } else if (decision === 'allow_always') {
          approvalOutput = `Approved and saved as a policy. ${approvalPolicy.explanation}`
        }
      } else if (deps.approvalMode === 'auto') {
        const kind: 'command' | 'connector' | 'tool' = call.function.name === 'run_command' ? 'command'
          : approvalPolicy.scope === 'connector' ? 'connector' : 'tool'
        const risk = riskBandForClassification(approvalPolicy.classification, kind)
        if (!approvalReason && fullAccessAutoAllows(kind, call.function.name, risk.band)) {
          decision = 'allow'
          approvalOutput = `Ran under Full access (${risk.reason}).`
        } else {
          const outboundPreview = outboundActionPreview(call.function.name, args)
          const awaitingOutput = [outboundPreview?.text, approvalReason || risk.reason].filter(Boolean).join('\n') || undefined
          send({ type: 'item_update', id: item.id, patch: { state: 'awaiting', output: awaitingOutput } })
          send({ type: 'status', status: 'awaiting' })
          send({ type: 'approval_request', callId: approvalId, itemId: item.id })
          decision = await deps.requestApproval(approvalId)
          if (decision === 'allow') approvalOutput = `Approved via saved policy (${risk.reason}). ${approvalPolicy.explanation}`
          else if (decision === 'allow_always') approvalOutput = `Approved and saved as a policy. ${approvalPolicy.explanation}`
        }
      } else {
        const outboundPreview = outboundActionPreview(call.function.name, args)
        const awaitingOutput = [outboundPreview?.text, approvalReason].filter(Boolean).join('\n') || undefined
        send({ type: 'item_update', id: item.id, patch: { state: 'awaiting', output: awaitingOutput } })
        send({ type: 'status', status: 'awaiting' })
        send({ type: 'approval_request', callId: approvalId, itemId: item.id })
        decision = await deps.requestApproval(approvalId)
        if (decision === 'allow_always') approvalOutput = `Approved and saved as a policy. ${approvalPolicy.explanation}`
      }
    }

    if (decision === 'deny') {
      const timelineItem = thread.timeline.find((entry) => entry.id === item.id)
      const automaticPolicyOutput = typeof timelineItem?.output === 'string' && /Auto-denied by saved policy/i.test(timelineItem.output)
        ? timelineItem.output
        : null
      const output = automaticPolicyOutput ?? denialOutput
      send({ type: 'item_update', id: item.id, patch: { state: 'denied', output, approval: output } })
      recordToolResult(call, call.function.name, `${output} Stop and ask how to proceed.`)
      return
    }

    send({ type: 'item_update', id: item.id, patch: { state: 'running', ...(approvalOutput ? { approval: approvalOutput } : {}) } })
    send({ type: 'status', status: 'tool' })

    let toolArgs = args
    let pluginBlocked = false
    let pluginBlockReason: string | undefined
    if (pluginRuntime) {
      const preCall = await pluginRuntime.runPreToolCall(call.function.name, args)
      toolArgs = preCall.args
      pluginBlocked = preCall.blocked
      pluginBlockReason = preCall.blockReason
    }
    if (pluginBlocked) {
      const output = pluginBlockReason || 'Blocked by a plugin.'
      send({ type: 'item_update', id: item.id, patch: { state: 'denied', output, approval: output } })
      recordToolResult(call, call.function.name, `${output} Stop and ask how to proceed.`)
      return
    }

    // Approval and plugin hooks may await; recheck Stop before dispatching a tool.
    if (signal.aborted) {
      const output = 'Cancelled before execution.'
      send({ type: 'item_update', id: item.id, patch: { state: 'error', output } })
      recordToolResult(call, call.function.name, `${output} Stop and ask how to proceed.`)
      return
    }
    if (call.function.name === 'ask_user') send({ type: 'status', status: 'awaiting' })
    const questionDeadline = call.function.name === 'ask_user' && deadlineAt ? AbortSignal.timeout(Math.max(1, Math.ceil(deadlineAt - Date.now()))) : undefined
    const toolSignal = questionDeadline ? AbortSignal.any([signal, questionDeadline]) : signal
    dispatchedMutation = !isReadOnlyTool(call.function.name, deps.toolContext?.mcpTools, deps.approvalMode)
    if (dispatchedMutation) recovery.markMutating(`tool:${call.id}`)
    const result = await executeRuntimeTool(pluginRuntime, call.function.name, toolArgs, { cwd, ...deps.toolContext, currentThread: thread, signal: toolSignal, userQuestionRunId: runId, userQuestionItemId: item.id }, turnBuiltinToolNames)

    if (questionDeadline?.aborted && !signal.aborted) state.budgetStop = 'time budget'
    toolConfirmed = true
    const confirmedControlledCommand = call.function.name === 'run_command' && turnBuiltinToolNames.has(call.function.name) && ['read-only', 'git-read', 'test', 'build'].includes(commandClass(String(toolArgs.command ?? ''))) && /^exit -?\d+(?:\s|·|$)/.test(result.meta ?? '')
    const uncertainFailure = dispatchedMutation && !result.ok && (!turnBuiltinToolNames.has(call.function.name) || !!externalEffectApprovalReason(call.function.name, toolArgs) || call.function.name === 'run_command' && !confirmedControlledCommand && (deps.toolContext?.allowSandboxNetwork === true || deps.toolContext?.backend?.kind === 'ssh'))
    recovery.toolCompleted(dispatchedMutation, result.ok, uncertainFailure)
    // An unconfirmed mutation pauses the run before another action can replay its effects.
    if (uncertainFailure) {
      state.uncertainActionStop = true
      recovery.pause('tool_uncertain', 'A dispatched action did not confirm its outcome; the run is paused before any replay or further tool action.')
    }

    await pluginRuntime.runPostToolCall(call.function.name, toolArgs, result)
    if (TODO_MUTATING_TOOLS.has(call.function.name)) state.todoMutationsSincePlanTouch++

    if (signal.aborted) {
      // Preserve confirmed late results after Stop as recovery evidence, without replaying the action.
      const mutationPatch = cancelledToolMutationPatch(result)
      if (mutationPatch) send({ type: 'item_update', id: item.id, patch: mutationPatch })
      recordToolResult(call, call.function.name, wrapUntrustedToolOutput(call.function.name, `Operation settled after Stop. Confirmed result: ${result.ok ? 'completed' : 'failed or uncertain'}. Do not replay automatically.\n${compressToolOutput({ tool: call.function.name, output: result.output, args: toolArgs, ok: result.ok }).content}`), true)
      return
    }
    if (callbacksAllowed() && deps.auditToolCall) {
      await deps.auditToolCall({ tool: call.function.name, args: toolArgs, resultOk: result.ok, output: result.output })
    }

    const subAgentPatch = result.subAgent ? {
      subAgent: result.subAgent,
      subAgentId: result.subAgent.id,
      parentAgentId: result.subAgent.parent.kind === 'agent' ? result.subAgent.parent.id : undefined,
      status: result.subAgent.status,
      args: {
        ...args,
        agentId: result.subAgent.id,
        subAgentId: result.subAgent.id,
        name: result.subAgent.name,
        task: result.subAgent.task,
        status: result.subAgent.status,
        result: result.subAgent.result,
        error: result.subAgent.error,
        parentAgentId: result.subAgent.parent.kind === 'agent' ? result.subAgent.parent.id : undefined,
      },
    } : {}

    let spillRelPath: string | null = null
    let spillWritten = false
    // Private conversations cannot spill tool output to managed files.
    if (!thread.incognito && !thread.phoneLocalOnly && !thread.googleLocalOnly && result.output.length > TOOL_OUTPUT_MAX_CHARS) {
      spillRelPath = toolOutputSpillPath(runId, ++state.spillSeq, call.function.name)
      try {
        if (deps.toolContext?.writeManagedOutput) {
          spillRelPath = await deps.toolContext.writeManagedOutput(spillRelPath, result.output)
          spillWritten = true
        }
      } catch {}
    }
    const bounded = compressToolOutput({ tool: call.function.name, output: result.output, args: toolArgs, ok: result.ok, meta: result.meta, spillRef: spillWritten ? spillRelPath : null })
    if (bounded.compressed) {
      ;(thread.contextFragments ??= []).push({
        source: 'tool_output', priority: 'normal', epoch: contextEpoch(thread),
        chars: bounded.content.length, estTokens: estimateTextTokens(bounded.content),
        truncated: true, originalChars: bounded.originalChars,
        ...(spillWritten && spillRelPath ? { note: spillRelPath } : {}),
      })
      if (thread.contextFragments.length > 50) thread.contextFragments = thread.contextFragments.slice(-50)
    }

    send({
      type: 'item_update',
      id: item.id,
      patch: { state: result.ok ? 'ok' : 'error', output: result.output, visualization: result.visualization, diff: result.diff, changedPaths: result.changedPaths, meta: result.meta, title: result.title || verb, path: result.path ?? (isVisualization ? undefined : target), command: result.command, appAction: result.appAction, ...(result.artifacts ? { artifacts: result.artifacts } : {}), ...(result.documentEvidence?.length ? { documentEvidence: result.documentEvidence } : {}), ...subAgentPatch },
    })

    const truncated = bounded.content

    let toolContent = truncated
    if (result.ok) {
      // Workspace changes refresh reads and checks; mutating actions retain lifetime repeat counts.
      if (result.changedPaths?.length) {
        mutatingActionSignatures.add(sig)
        refreshableActionSignatures.delete(sig)
        for (const priorSig of refreshableActionSignatures) {
          toolAttemptCounts.delete(priorSig)
          failCounts.delete(priorSig)
          repeatCounts.delete(priorSig)
        }
        resultHashCounts.clear()
      }
      failCounts.delete(sig)
      const repeats = (repeatCounts.get(sig) ?? 0) + 1
      repeatCounts.set(sig, repeats)
      const note = repeatGuardNote(repeats, guardPolicy?.repeatLimit)
      emitGuardWarning(`repeat:${sig}`, note)
      if (note) toolContent = `${toolContent}${note}`
    } else {
      const n = (failCounts.get(sig) ?? 0) + 1
      failCounts.set(sig, n)
      if (n >= THRASH_LIMIT) toolContent = `${truncated}\n\n[guardrail] This exact action has failed ${n} times in a row. Do not repeat it. Change your approach or explain the blocker to the user.`
    }

    const nameCount = (toolNameCounts.get(call.function.name) ?? 0) + 1
    toolNameCounts.set(call.function.name, nameCount)
    const searchNote = searchLoopNote(call.function.name, nameCount, guardPolicy?.searchLimit)
    emitGuardWarning(`search:${call.function.name}`, searchNote)
    if (searchNote) toolContent = `${toolContent}${searchNote}`

    else {
      const cadenceNote = toolCadenceNote(call.function.name, nameCount, guardPolicy?.toolCadenceLimit)
      emitGuardWarning(`cadence:${call.function.name}`, cadenceNote)
      if (cadenceNote) toolContent = `${toolContent}${cadenceNote}`
    }

    if (result.ok && NO_PROGRESS_TOOLS.has(call.function.name)) {
      const resultHash = createHash('sha1').update(result.output).digest('hex')
      const progressKey = `${sig}|${resultHash}`
      const priorCount = resultHashCounts.get(progressKey) ?? 0
      resultHashCounts.set(progressKey, priorCount + 1)
      const progressNote = noProgressNote(call.function.name, priorCount, guardPolicy?.noProgressLimit)
      emitGuardWarning(`no-progress:${progressKey}`, progressNote)
      if (progressNote) toolContent = `${toolContent}${progressNote}`
    }
    recordToolResult(call, call.function.name, wrapUntrustedToolOutput(call.function.name, toolContent), true)
    } finally {
      if (dispatchedMutation && !toolConfirmed) recovery.toolUncertain()
      recovery.end(`tool:${call.id}`)
      send({ type: 'lifecycle', phase: 'tool_end', ts: Date.now(), step, callId: call.id, tool: call.function.name })
    }
  }
  // Parallel completion changes only the timeline; model results retain the assistant's call order.
  await executeToolExecutionBatches(
    toolCalls,
    (call) => PARALLEL_READ_ONLY_TOOLS.has(call.function.name)
      && isReadOnlyTool(call.function.name, deps.toolContext?.mcpTools, deps.approvalMode),
    executeRequestedCall,
    (call) => {
      const result = toolResults.get(call.id)
      if (result) thread.apiMessages.push(result)
    },
    MAX_PARALLEL_READ_TOOLS,
    () => signal.aborted || state.uncertainActionStop || !!state.budgetStop,
  )

  send({ type: 'lifecycle', phase: 'turn_end', ts: Date.now(), step })
  if (state.budgetStop) thread.apiMessages = reconcileOrphanedToolCalls(thread.apiMessages)
  persist()
}
