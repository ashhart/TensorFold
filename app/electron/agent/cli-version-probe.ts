// Reads a CLI's `--version` run. A clean exit with a version on stdout counts, even with warnings on stderr.
export interface VersionProbeRun {
  code: number | null
  signal?: string | null
  stdout: string
  stderr: string
  error?: string | null
  timedOut?: boolean
}

export type VersionProbeResult = { version: string; problem: null } | { version: null; problem: string }

const VERSION = /\bv?\d+\.\d+/
const DETAIL_MAX = 120
const VERSION_MAX = 120

const lines = (text: string) => text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)

/** One short line for the UI: the home path becomes ~, whitespace collapses, long text ends in an ellipsis. */
export function shortProblem(text: string, home = ''): string {
  let line = lines(text)[0] ?? ''
  const root = home.replace(/\/+$/, '')
  if (root.length > 1) line = line.split(root).join('~')
  line = line.replace(/\s+/g, ' ')
  return line.length > DETAIL_MAX ? `${line.slice(0, DETAIL_MAX - 1).trimEnd()}…` : line
}

export function classifyVersionProbe(run: VersionProbeRun, home = ''): VersionProbeResult {
  const broken = (problem: string): VersionProbeResult => ({ version: null, problem })
  if (run.error) return broken(shortProblem(run.error, home) || 'Could not start')
  if (run.timedOut) return broken('No answer to --version')
  const stderr = shortProblem(run.stderr, home)
  const stdout = shortProblem(run.stdout, home)
  if (run.code !== 0) return broken(stderr || stdout || (run.code == null ? `Stopped by ${run.signal ?? 'a signal'}` : `Exited with code ${run.code}`))
  const version = lines(run.stdout).find((line) => VERSION.test(line))
  if (!version) return broken(stderr || (stdout ? `No version in "${stdout}"` : 'Printed no version'))
  return { version: version.slice(0, VERSION_MAX), problem: null }
}
