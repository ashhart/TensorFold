// One-shot generator: turns the verified audit digest into a formatted Markdown report.
// Not part of the test suite. Reads /tmp/vontra-audit-digest.json, writes docs/security-audit-2026-06-21.md.
import { readFileSync, writeFileSync } from 'node:fs'

const d = JSON.parse(readFileSync('/tmp/vontra-audit-digest.json', 'utf8')).result
const dec = (s) =>
  String(s == null ? '' : s)
    .replace(/&gt;/g, '>').replace(/&lt;/g, '<').replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'").replace(/&amp;/g, '&')

const sevRank = { Critical: 0, High: 1, Medium: 2, Low: 3, Info: 4 }
const likRank = { High: 0, Medium: 1, Low: 2 }

const all = []
for (const dm of d.digest) for (const f of dm.findings) all.push({ dimension: dm.dimension, key: dm.key, ...f })
const real = all.filter((f) => f.isReal && f.verdict !== 'refuted')
const refuted = all.filter((f) => !f.isReal || f.verdict === 'refuted')
real.sort((a, b) => sevRank[a.severity] - sevRank[b.severity] || likRank[a.likelihood] - likRank[b.likelihood] || a.key.localeCompare(b.key))

const priorityOf = (f) =>
  f.severity === 'High' || (f.severity === 'Medium' && f.likelihood === 'High') ? 'Immediate'
  : f.severity === 'Medium' ? 'Short-term'
  : 'Long-term'

const dist = real.reduce((a, f) => ((a[f.severity] = (a[f.severity] || 0) + 1), a), {})

let id = 0
for (const f of real) f._id = ++id

const out = []
const w = (s = '') => out.push(s)

w('# Security Audit & Gap Analysis — TensorFold Studio')
w('')
w('**Target:** TensorFold Studio — Electron + React/Vite agentic AI coding harness (~40,000 LoC under `electron/`).')
w('**Audit type:** Senior application-security & cloud-architecture review (static code audit).')
w('**Date:** 2026-06-21.')
w('**Method:** Multi-agent fan-out across 8 framework dimensions, with every finding independently **adversarially verified** against the source before inclusion. Findings that a verifier could refute, or that a mitigating control negated, were dropped or downgraded.')
w('')
w('> **Important — point-in-time line references.** All `file:line` citations reflect the repository at audit time, *before* the planned `electron/main.ts` decomposition. After that refactor the line numbers will shift; the file/function references remain valid anchors.')
w('')
w('---')
w('')
w('## Executive summary')
w('')
w('TensorFold Studio has a **stronger-than-typical security baseline for an Electron application**. The classic Electron remote-code-execution pitfalls are correctly closed: `contextIsolation`, `sandbox`, and `nodeIntegration:false` are set on every `BrowserWindow`; the Content-Security-Policy is strict (`script-src \'self\'`, no `unsafe-inline`/`unsafe-eval`, no remote script); permission request/check handlers are denied; every IPC channel is wrapped with sender/frame validation and several sensitive channels carry server-side role checks; secrets are encrypted at rest with an app-managed AES-256-GCM key (owner-only `0600`, in the app private data dir); the app never touches the macOS Keychain (`safeStorage`) at all — so there is no Keychain prompt — and a settings file it cannot decrypt (e.g. a pre-refactor Keychain-encrypted blob) is backed up and the user re-enters keys rather than being silently overwritten; secrets are never written as plaintext; the remote-control server is loopback-by-default with a timing-safe token compare; and signed updates are required by default. The adversarial verification pass **confirmed these controls in code** and, as a result, **downgraded two finder-rated "High" issues to Medium and refuted one outright** — the result is calibrated, not inflated.')
w('')
w('**The residual risk is not in the Electron shell — it is in the *agentic trust boundaries*.** The issues that matter cluster around three questions unique to an autonomous coding agent:')
w('')
w('1. **Who is allowed to drive the agent?** Inbound Telegram authorization is keyed on *chat ID*, not *sender user ID*, so any member of an allow-listed group can issue commands (Finding 1).')
w('2. **What can the agent\'s tools reach?** The secret-file read blocklist is an incomplete denylist that is trivially bypassable (Finding 2), the `search` tool emits credential files that `read_file` blocks (Finding 13), and nothing stops the agent writing `.git/hooks` (Finding 3).')
w('3. **What third parties does it trust?** Read-only auto-approval trusts an MCP server\'s *self-asserted* `readOnlyHint`, letting a malicious server bypass the approval gate (Finding 4); the Docker isolation that is supposed to contain MCP servers permits `--network host` and unpinned images (Finding 8).')
w('')
w('A second cluster is **data-at-rest / egress hygiene**: archived threads are written in plaintext outside the encrypted store (Finding 12), the LAN orchestrator snapshot serves unredacted thread content (Finding 11), the tamper-evident audit log records raw tool arguments and command output without redaction (Finding 14), and "purge local memory" leaves residual sensitive data behind (Finding 15).')
w('')
w(`**Totals (post-verification):** ${real.length} confirmed/likely findings — **${dist.High || 0} High, ${dist.Medium || 0} Medium, ${dist.Low || 0} Low, ${dist.Info || 0} Info, 0 Critical.** ${refuted.length} finding refuted/dropped.`)
w('')
w('### Final risk rating: **MODERATE**')
w('')
w('No Critical issues and no unauthenticated RCE path; platform hardening is genuinely good. The rating is *Moderate* rather than *Low* because four High-severity weaknesses are realistically reachable under normal feature use — a shared Telegram group, an added MCP server, or the agent running in Full-Auto with a bypassable secret-read denylist — and several Medium data-exposure gaps would turn a single foothold into credential/PII disclosure. None of the Highs require exotic preconditions, but all require a feature to be *enabled* (bot, MCP, LAN orchestrator, full-auto), which keeps the realistic blast radius bounded.')
w('')
w('---')
w('')
w('## Severity distribution')
w('')
w('| Severity | Count |')
w('|---|---|')
for (const s of ['Critical', 'High', 'Medium', 'Low', 'Info']) w(`| ${s} | ${dist[s] || 0} |`)
w(`| **Total** | **${real.length}** |`)
w('')

// Top 10
w('## Top 10 risks')
w('')
w('| # | Severity / Likelihood | Finding | Dimension |')
w('|---|---|---|---|')
real.slice(0, 10).forEach((f, i) => w(`| ${i + 1} | ${f.severity} / ${f.likelihood} | ${dec(f.title)} | ${f.key} |`))
w('')

// Quick wins
const quickWins = real.filter((f) => f.effort === 'S' && ['High', 'Medium'].includes(f.severity))
w('## Quick wins (high value, ~<1 day each)')
w('')
for (const f of quickWins) w(`- **[${f.severity}] #${f._id} ${dec(f.title)}** — ${dec(f.remediation).split('. ')[0]}.`)
w('')

// Roadmap
const groups = { Immediate: [], 'Short-term': [], 'Long-term': [] }
for (const f of real) groups[priorityOf(f)].push(f)
w('## Remediation roadmap')
w('')
w('**Immediate (0–2 weeks)** — all High plus high-likelihood Medium:')
for (const f of groups['Immediate']) w(`- #${f._id} [${f.severity}] ${dec(f.title)} _(owner: ${f.owner}, effort: ${f.effort})_`)
w('')
w('**Short-term (1–3 months)** — remaining Medium:')
for (const f of groups['Short-term']) w(`- #${f._id} [${f.severity}] ${dec(f.title)} _(owner: ${f.owner}, effort: ${f.effort})_`)
w('')
w('**Long-term (3–6 months)** — Low/Info hardening & hygiene:')
for (const f of groups['Long-term']) w(`- #${f._id} [${f.severity}] ${dec(f.title)} _(owner: ${f.owner}, effort: ${f.effort})_`)
w('')
w('---')
w('')

// Detailed findings
w('## Detailed findings')
w('')
w('Grouped by severity. Each finding was verified against the code; the **Verification** line records the adversarial verifier\'s verdict and any mitigating control it found.')
w('')
let curSev = null
for (const f of real) {
  if (f.severity !== curSev) {
    curSev = f.severity
    w(`### ${curSev}`)
    w('')
  }
  w(`#### Finding ${f._id}: ${dec(f.title)}`)
  w('')
  w(`- **Description:** ${dec(f.description)}`)
  w('- **Evidence:**')
  for (const e of f.evidence || []) {
    w(`  - \`${e.file}:${e.lines}\``)
    if (e.snippet) {
      w('    ```')
      for (const line of dec(e.snippet).split('\n')) w(`    ${line}`)
      w('    ```')
    }
  }
  const sevNote = f.finderSeverity && f.finderSeverity !== f.severity ? ` _(finder rated ${f.finderSeverity}; adjusted by verification)_` : ''
  w(`- **Severity:** ${f.severity}${sevNote}`)
  w(`- **Likelihood:** ${f.likelihood}`)
  w(`- **Business Impact:** ${dec(f.businessImpact)}`)
  w(`- **Recommended Remediation:** ${dec(f.remediation)}`)
  w(`- **Owner:** ${f.owner}`)
  w(`- **Priority:** ${priorityOf(f)}`)
  w(`- **Estimated Effort:** ${f.effort}`)
  w(`- **Control / Framework Mapping:** ${dec(f.frameworkMapping)}`)
  w(`- **Evidence status:** ${f.finderStatus === 'needs-validation' ? 'Needs validation' : f.finderStatus === 'assumption' ? 'Assumption' : 'Confirmed'}`)
  const mit = dec(f.mitigatingControl).trim()
  w(`- **Verification:** ${f.verdict}${mit ? ` — mitigating control noted: ${mit}` : ' — no mitigating control found.'}`)
  w('')
}

// Controls observed
w('---')
w('')
w('## Confirmed controls (credit where due)')
w('')
w('These were positively verified in code and are *not* gaps — listed so they are not re-flagged and so regressions are noticeable:')
w('')
for (const dm of d.digest) {
  if (!dm.controlsObserved || dm.controlsObserved.length === 0) continue
  w(`**${dm.dimension}**`)
  for (const c of dm.controlsObserved) w(`- ${dec(c)}`)
  w('')
}

// Refuted
w('## Refuted / dropped during verification')
w('')
for (const f of refuted) w(`- **${dec(f.title)}** _(finder rated ${f.finderSeverity}; ${f.key})_ — ${dec(f.verifierRationale)}`)
w('')

// Missing evidence
w('## Questions & missing evidence (to complete the audit)')
w('')
w('This was a static code audit. The following require runtime/deployment confirmation or owner input before the corresponding items can be closed:')
w('')
for (const dm of d.digest) {
  if (!dm.missingEvidence || dm.missingEvidence.length === 0) continue
  w(`**${dm.dimension}**`)
  for (const m of dm.missingEvidence) w(`- ${dec(m)}`)
  w('')
}

w('---')
w('')
w('## Compliance alignment notes')
w('')
w('- **CIS Controls:** Strongest alignment on Control 3 (data protection — encryption-at-rest, denied-by-default) and Control 4 (secure config — Electron hardening). Gaps concentrate in Control 3 residue/disposal (Findings 12, 15, 28), Control 6 (access control — Findings 1, 19), and Control 15 (vendor/third-party — Findings 4, 8, 26).')
w('- **OWASP:** Primary exposure is A01 Broken Access Control (agent authz), A03 Injection (tool capability denylists), A08 Software/Data Integrity (update artifact, install scripts), and the OWASP LLM Top-10 (LLM01 prompt injection, LLM03 supply chain, LLM06 sensitive disclosure, LLM07 insecure plugin design).')
w('- **GDPR:** Findings 11, 12, 15 touch Art. 17 (erasure) and Art. 32 (security of processing). Relevant only if TensorFold Studio processes others\' personal data; for a single-user local dev tool the obligation is limited.')
w('- **SOC 2 / ISO 27001:** No formal control evidence was in scope (logging retention, IR runbooks, vendor management, backup/DR are largely **Needs validation** — they are organizational, not in the codebase).')
w('')
w('*Generated from the verified audit digest. Finding text is the verifiers\' validated wording.*')

writeFileSync('docs/security-audit-2026-06-21.md', out.join('\n') + '\n')
console.log(`Wrote docs/security-audit-2026-06-21.md (${out.length} lines, ${real.length} findings).`)
console.log('Priority counts:', Object.fromEntries(Object.entries(groups).map(([k, v]) => [k, v.length])))
console.log('Quick wins:', quickWins.length)
