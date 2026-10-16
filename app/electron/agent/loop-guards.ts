// Loop Mode — guards that keep autonomous phases honest:
//  - tool-evidence guard: a phase that produced no real tool activity is suspect.
//  - role-boundary guard: read-only roles can't mutate source; validators get
//    temp-only write latitude. Pure; the engine consults these per tool call.

import type { LoopPhaseId } from './loop-types'

export const READ_ONLY_ROLES: LoopPhaseId[] = ['product', 'architect']
const WRITE_TOOLS = new Set(['edit', 'write', 'create'])
const LOOP_STATE_RE = /\.(pi|vontra)-loop\//
const MUTATING_SHELL_RE = /\b(rm|mv|cp|mkdir|touch|tee|sed -i|git (add|commit|checkout|reset|rm))\b|>>?/

export function toolEvidenceVerdict(activity: string[]): { ok: boolean; useful: number } {
  const useful = activity.filter((a) => a && !LOOP_STATE_RE.test(a)).length
  return { ok: useful > 0, useful }
}

function pathOf(args: unknown): string {
  return args && typeof args === 'object' && 'path' in args ? String((args as { path: unknown }).path) : ''
}
function cmdOf(args: unknown): string {
  return args && typeof args === 'object' && 'command' in args ? String((args as { command: unknown }).command) : ''
}

function targetsSourceTree(command: string): boolean {
  if (!MUTATING_SHELL_RE.test(command)) return false
  if (/\/tmp\/|mktemp|\bTMPDIR\b/.test(command)) return false
  return /\bsrc\b|\.tsx?\b|\.json\b|\.md\b|package/.test(command)
}

export function roleBoundaryViolation(role: LoopPhaseId, tool: string, args: unknown, _cwd: string): string | undefined {
  const readOnly = READ_ONLY_ROLES.includes(role)
  if (readOnly && WRITE_TOOLS.has(tool)) return `${role} is read-only and may not ${tool} ${pathOf(args)}`.trim()
  if (tool === 'bash') {
    const cmd = cmdOf(args)
    if (readOnly && MUTATING_SHELL_RE.test(cmd) && !/\/tmp\/|mktemp/.test(cmd)) {
      return `${role} is read-only; mutating shell blocked: ${cmd.slice(0, 60)}`
    }
    if (targetsSourceTree(cmd) && role !== 'dev' && role !== 'docs') {
      return `${role} may not mutate source via shell: ${cmd.slice(0, 60)}`
    }
  }
  return undefined
}
