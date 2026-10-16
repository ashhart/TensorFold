// Remote PR publish — the one step the local PR loop (pr-loop.ts) deliberately stops short of.
// pr-loop.ts builds/reviews/approves a change entirely on a local worktree branch; this module
// completes the async issue→PR teammate loop by pushing that branch and opening a GitHub DRAFT
// PR via the `gh` CLI, but ONLY when the remote-publish gate is explicitly allowed
// (localPrRemotePublishGate) — remote publish is never implicit.
//
// Command construction is pure (unit-tested); execution goes through an injected runner
// (execFile-style argv, NEVER a shell string) so the PR body/title can't inject shell, and so
// the sequence is testable without a live push.

export interface PrPublishInput {
  branch: string
  baseBranch?: string | null // omit → gh uses the repo's default branch
  title: string
  bodyMarkdown: string
  worktree: string // cwd the git/gh commands run in
  draft?: boolean // default true
  remote?: string // default 'origin'
}

export interface PrPublishCommand {
  argv: string[]
  cwd: string
  label: string
}

/**
 * The exact command sequence: push the branch (with upstream tracking), then open the draft PR.
 * argv arrays (execFile semantics) — the title/body travel as single elements, so no shell
 * metacharacter in them is ever interpreted. Pure.
 */
export function buildPrPublishCommands(input: PrPublishInput): PrPublishCommand[] {
  const remote = input.remote || 'origin'
  const draft = input.draft !== false
  const push: PrPublishCommand = {
    argv: ['git', 'push', '-u', remote, input.branch],
    cwd: input.worktree,
    label: `Push ${input.branch} to ${remote}`,
  }
  const gh: string[] = ['gh', 'pr', 'create']
  if (draft) gh.push('--draft')
  if (input.baseBranch) gh.push('--base', input.baseBranch)
  gh.push('--head', input.branch, '--title', input.title, '--body', input.bodyMarkdown)
  return [push, { argv: gh, cwd: input.worktree, label: 'Open draft PR' }]
}

/** Pull the first github.com pull-request URL out of `gh pr create` output. Pure. */
export function parsePrUrlFromGhOutput(stdout: string): string | null {
  const m = String(stdout || '').match(/https?:\/\/[^\s]*github\.com\/[^\s]+\/pull\/\d+/i)
  return m ? m[0] : null
}

export interface PrPublishRunner {
  run(argv: string[], cwd: string): Promise<{ ok: boolean; stdout: string; stderr: string }>
}

export interface PrPublishStep { label: string; ok: boolean; detail?: string }
export interface PrPublishResult {
  ok: boolean
  prUrl?: string
  error?: string
  steps: PrPublishStep[]
}

/**
 * Publish an approved local PR to GitHub as a draft. Refuses entirely unless `gate.allowed`
 * (the localPrRemotePublishGate decision). Runs the push, then the PR create; stops at the
 * first failure. Success requires a parseable PR URL back from gh (so we never report a PR was
 * opened when we can't point at it).
 */
export async function publishLocalPr(
  input: PrPublishInput,
  gate: { allowed: boolean; reason: string },
  runner: PrPublishRunner,
): Promise<PrPublishResult> {
  if (!gate.allowed) return { ok: false, error: gate.reason, steps: [] }

  const commands = buildPrPublishCommands(input)
  const steps: PrPublishStep[] = []
  let lastStdout = ''
  for (const cmd of commands) {
    const res = await runner.run(cmd.argv, cmd.cwd)
    lastStdout = res.stdout
    if (!res.ok) {
      const detail = (res.stderr || res.stdout || '').trim()
      steps.push({ label: cmd.label, ok: false, detail })
      return { ok: false, error: `${cmd.label} failed: ${detail || 'command failed'}`, steps }
    }
    steps.push({ label: cmd.label, ok: true })
  }

  const prUrl = parsePrUrlFromGhOutput(lastStdout)
  if (!prUrl) {
    return { ok: false, error: 'The PR command succeeded but returned no PR URL to confirm.', steps }
  }
  return { ok: true, prUrl, steps }
}

/**
 * Production runner: execFile (no shell) with the worktree as cwd. Kept tiny and separate so the
 * module core stays pure/testable; the host injects this.
 */
export function nodeExecFilePrRunner(): PrPublishRunner {
  return {
    run: (argv: string[], cwd: string) =>
      new Promise((resolve) => {
        // Lazy require so importing this module never pulls in child_process where unused.
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { execFile } = require('node:child_process') as typeof import('node:child_process')
        execFile(argv[0], argv.slice(1), { cwd, maxBuffer: 8 * 1024 * 1024, timeout: 120_000 }, (err, stdout, stderr) => {
          resolve({ ok: !err, stdout: String(stdout || ''), stderr: String(stderr || (err ? err.message : '')) })
        })
      }),
  }
}
