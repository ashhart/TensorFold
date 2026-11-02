// Stable public entry point; run lifecycle, turns, and pure policies live in focused modules.

export {
  reconcileOrphanedToolCalls,
  cancelledToolMutationPatch,
} from './run-loop-history'
export {
  type RunDeps,
} from './run-loop-contracts'
export {
  resolveRunInstructionGraph,
  renderDurableProjectMemoryPrompt,
  replacePrimarySystemPrompt,
} from './run-loop-prompts'
export {
  PARALLEL_READ_ONLY_TOOLS,
  pluginToolSchemas,
  executeRuntimeTool,
  toolApprovalReasonForMode,
} from './run-loop-tool-policy'
export {
  repeatGuardNote,
  detectTextLoop,
  providerNeedsTextLoopBreaker,
  type TextTruncationAction,
  textTruncationAction,
  compactionMadeProgress,
  searchLoopNote,
  toolCadenceNote,
  noProgressNote,
  type VerifyOnStopSnapshot,
  verifyOnStopSnapshot,
  isTruncatedToolCallArgs,
} from './run-loop-guards'
export {
  type StreamRetryAction,
  isDelegatedTransport,
  streamRetryDecision,
  abortableDelay,
} from './run-loop-retry'
export {
  delegateChildToolSchemas,
  auxiliaryToolContext,
  parseDelegateTaskArgs,
  formatDelegateTaskResults,
} from './run-loop-auxiliary'
export {
  parseSummaryVerification,
  compactThread,
} from './run-loop-compaction'
export {
  runAgent,
} from './run-loop'
