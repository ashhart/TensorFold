import type { ModelRouterScoreRole } from './model-router'

// Agent typing:
// launch_subagent(s) items may pick a specialist preset instead of the generic
// delegate. Presets are advisory prompts plus a model-router role preference; the
// caller's explicit model/provider/account always wins. TensorFold sub-agents already
// run read-only, so scout/reviewer presets are enforcement-free framing.

export type SubAgentPresetId = 'scout' | 'reviewer' | 'security' | 'builder'

export interface SubAgentPreset {
  id: SubAgentPresetId
  label: string
  /** ModelRouter score role consulted when the caller did not pin a model. */
  role: ModelRouterScoreRole
  /** Prepended to the task so the child agent frames its work correctly. */
  promptPrefix: string
}

export const SUBAGENT_PRESETS: Record<SubAgentPresetId, SubAgentPreset> = {
  scout: {
    id: 'scout',
    label: 'Scout',
    role: 'scout',
    promptPrefix: 'You are a read-only research scout. Investigate quickly, do not propose edits, and return a dense factual report: what exists, where (exact paths), and the specific findings the caller asked for. End with a short "Key facts" list.',
  },
  reviewer: {
    id: 'reviewer',
    label: 'Reviewer',
    role: 'reviewer',
    promptPrefix: 'You are a read-only code reviewer. Verify correctness against the stated task. Return a verdict (pass / fail / needs-changes), concrete findings with file:line references, and a one-line confidence note. Do not rewrite the code; describe the fix.',
  },
  security: {
    id: 'security',
    label: 'Security reviewer',
    role: 'reviewer',
    promptPrefix: 'You are a read-only security reviewer. Hunt injection, authz, path-traversal, secret handling, and unsafe deserialization issues. Return findings ordered by severity (critical/high/medium/low) with file:line references and a concrete remediation per finding.',
  },
  builder: {
    id: 'builder',
    label: 'Builder',
    role: 'builder',
    promptPrefix: 'You are a focused implementation worker. Investigate and complete the task in one pass, prefer minimal diffs, and skip project-wide validation unless the task asks for it.',
  },
}

export function normalizeSubAgentPreset(value: unknown): SubAgentPreset | null {
  return typeof value === 'string' && value in SUBAGENT_PRESETS ? SUBAGENT_PRESETS[value as SubAgentPresetId] : null
}

export function applySubAgentPreset(preset: SubAgentPreset | null, task: string): string {
  if (!preset) return task
  return `${preset.promptPrefix}\n\n# Task\n${task}`
}
