import type { LoopShellResult, RunContract, VerificationCheck, VerificationReport } from './loop-types'

const OUTPUT_LIMIT = 3000

export function verificationConfigurationIssue(contract: RunContract): string | null {
  const commands = contract.verificationCommands
  if (!Array.isArray(commands) || !commands.length) {
    return 'Configure at least one verification command before starting Loop Mode.'
  }
  if (commands.length > 32) return 'Configure at most 32 required verification commands.'
  if (commands.some(command => typeof command !== 'string' || !command.trim())) {
    return 'Every required verification command must contain a command to run.'
  }
  if (commands.some(command => command.length > 2000)) return 'Keep each verification command within 2000 characters.'
  if (commands.some(command => command.includes('\0'))) return 'Verification commands cannot contain NUL characters.'
  return null
}

export function verificationCheck(command: string, result: LoopShellResult): VerificationCheck {
  const exitCode = Number.isSafeInteger(result.code) && result.code! >= 0 ? result.code : null
  const outcome = result.cancelled ? 'cancelled'
    : result.timedOut ? 'timed_out'
      : result.signal || exitCode == null ? 'error'
        : exitCode === 0 ? 'passed' : 'failed'
  return { command, outcome, exitCode, output: `${result.stdout}\n${result.stderr}`.trim().slice(-OUTPUT_LIMIT) }
}

export function verificationError(command: string, error: unknown): VerificationCheck {
  const name = error instanceof Error ? error.name : ''
  const outcome = name === 'AbortError' ? 'cancelled' : name === 'TimeoutError' ? 'timed_out' : 'error'
  const output = (error instanceof Error ? error.message : String(error)).slice(-OUTPUT_LIMIT)
  return { command, outcome, exitCode: null, output }
}

export function verificationIssues(commands: readonly string[], report: VerificationReport): string[] {
  if (!report || !Array.isArray(report.checks)) return ['Required verification results are missing.']
  const issues: string[] = []
  if (report.cancelled) issues.push('Required verification was cancelled.')
  if (report.checks.length !== commands.length) issues.push('Required verification did not return every configured check.')
  for (const [index, command] of commands.entries()) {
    const check = report.checks[index]
    if (!check || check.command !== command) {
      issues.push(`Required verification result is missing for command ${index + 1}: ${command}`)
    } else if (check.outcome !== 'passed' || check.exitCode !== 0 || typeof check.output !== 'string') {
      issues.push(`Required verification did not pass: ${command} (${check.outcome}, exit ${check.exitCode ?? 'unknown'}).`)
    }
  }
  return issues
}

export function verificationEvidence(report: VerificationReport): string {
  if (!report || !Array.isArray(report.checks)) return 'Required verification results are missing.'
  return report.checks.map(check =>
    `$ ${check.command}\n[${check.outcome}; exit ${check.exitCode ?? 'unknown'}]\n${check.output}`,
  ).join('\n\n') || '(no verification commands completed)'
}
