import os from 'node:os'

const BLOCKED_COMMAND_PATTERNS: Array<{ re: RegExp; reason: string }> = [
  // Bare home / root target: `rm -rf /`, `rm -rf ~`, `rm -rf $HOME`, `rm -rf ${HOME}` (and quoted
  // forms). The target may be wrapped in matching quotes and `$HOME` may use brace syntax.
  { re: /\brm\s+(-[^\s]*[rf][^\s]*|--recursive|--force).*(?:\s|^)["']?(?:\/|~|\$HOME|\$\{HOME\})["']?(?:\s|$)/i, reason: 'refuses recursive deletion of home or root paths' },
  // `rm -rf`/`rm -fr` of any subpath under home (not just the bare home root caught above) --
  // e.g. `rm -rf $HOME/Documents`, `rm -rf ~/Documents/important`, `rm -rf "$HOME"/Documents`,
  // `rm -rf ${HOME}/Documents`. A closing quote/brace may sit between the token and the `/`.
  { re: /\brm\s+(?:-[^\s]*[rf][^\s]*|--recursive|--force)\b[\s\S]*?(?:~|\$HOME|\$\{HOME\})["']?\/[^\s"']+/i, reason: 'refuses recursive deletion under the home directory' },
  // `find ... -delete` / `find ... -exec rm ...` deletes regardless of target depth -- the
  // literal `rm` blocklist above never sees these since `find` invokes `rm` itself.
  { re: /(?:^|[\s;&|])find\b[\s\S]*?(?:-delete\b|-exec(?:dir)?\s+rm\b|-ok(?:dir)?\s+rm\b)/i, reason: 'refuses find-based recursive deletion (-delete / -exec rm)' },
  // `rsync --delete` mirrors a (typically empty) source over a home-relative target, deleting
  // everything not present in source -- equivalent to `rm -rf` of that target.
  { re: /(?:^|[\s;&|])rsync\b[\s\S]*--delete\b[\s\S]*?["']?(?:~|\$HOME)(?:\/[^\s"']*)?["']?(?:\s|$)/i, reason: 'refuses rsync --delete targeting the home directory' },
  { re: /\b(?:curl|wget)\b[\s\S]*(?:\|\s*(?:sh|bash|zsh|python|ruby|perl|node)\b|>\s*\/tmp\/)/i, reason: 'refuses download-and-execute shell pipelines' },
  // curl/wget writing the fetched payload to a file via -o/-O/--output(-document), where the
  // same command line then executes (chmod+x, sh/bash/zsh/python/ruby/perl/node, or `./script`)
  // -- the prior pattern only caught the `| sh` pipeline / `>` redirect forms.
  { re: /\b(?:curl|wget)\b[\s\S]*?(?:-o\s*\S|-O\b|--output(?:-document)?[\s=])[\s\S]*?(?:&&|;|\n|\|\|)[\s\S]*?\b(?:chmod\s+\+x|sh\b|bash\b|zsh\b|\.\/\S+|python3?\b|ruby\b|perl\b|node\b)/i, reason: 'refuses download-then-execute via curl/wget output flags' },
  // The same attack can use ordinary shell redirection rather than curl/wget's output
  // flags (`curl URL > payload.sh && bash payload.sh`). The old rule only recognized
  // redirects into /tmp, so a relative or home path bypassed Full Auto's last host-side
  // safety gate. Require a later execution step so benign data downloads remain usable.
  { re: /\b(?:curl|wget)\b[\s\S]*?\d*>>?\s*(?!&)[^\s;&|]+[\s\S]*?(?:&&|;|\n|\|\|)[\s\S]*?(?:(?<=[\s;&|])|^)(?:chmod\s+\+x|sh\b|bash\b|zsh\b|\.\/\S+|python3?\b|ruby\b|perl\b|node\b)/i, reason: 'refuses download-then-execute via shell redirection' },
  { re: /\b(?:nc|ncat|netcat|socat)\b[\s\S]*(?:-e|EXEC:)/i, reason: 'refuses reverse-shell style network commands' },
  { re: /\b(?:chmod|chown)\s+[\s\S]*(?:\/etc|\/System|\/Library|\/usr\/bin|\/bin)\b/i, reason: 'refuses permission changes to system paths' },
  { re: /\b(?:sudo|su|doas|pkexec|gosu)\b/i, reason: 'refuses privilege escalation from agent-controlled commands' },
  { re: /\b(?:security\s+find-(?:generic|internet)-password|ssh-add\s+-L|cat\s+~\/\.ssh|cat\s+\$HOME\/\.ssh)\b/i, reason: 'refuses commands that directly read credential stores' },
  { re: /(?:^|[\s;&|])(?:cat|less|more|sed|awk|grep|rg|find|head|tail|xxd|od|hexdump|strings|nl|tac|base64|openssl|dd|cp|mv|scp|rsync)\b[\s\S]*(?:~\/|\$HOME\/)?\.(?:ssh|aws|azure|gcloud|docker|kube)\b/i, reason: 'refuses direct reads of common secret directories' },
  { re: /(?:^|[\s;&|])(?:cat|less|more|sed|awk|grep|rg|head|tail|xxd|od|base64|openssl|nl|tac|cp|mv)\b[\s\S]*(?:^|[\s/])\.env(?:[.\s/]|$)/i, reason: 'refuses direct reads of .env files' },
  { re: /\b(?:python3?|node|ruby|perl)\b[\s\S]*(?:\.ssh|\.aws|\.env\b|id_rsa|id_ed25519|credentials|\.netrc)/i, reason: 'refuses scripting-language reads of credential files' },
  { re: /(?:^|[\s;&|])gh\s+(?:pr\s+create|release\s+create|repo\s+sync)\b/i, reason: 'run Git hygiene before publishing through GitHub CLI' },
]

const APPROVAL_COMMAND_PATTERNS: Array<{ re: RegExp; reason: string }> = [
  // Mirror approval-policy's isSecretReadCommand: a credential/secret read must require approval, so
  // Full Auto can't silently read & leak in-repo secrets. The binary list mirrors the BLOCKED
  // dump-tool set (xxd/od/strings/base64/dd/...) so a non-`cat` reader can't slip a secret through.
  { re: /(?:^|[\s;&|])(?:cat|grep|rg|sed|awk|head|tail|less|more|find|ls|xxd|od|hexdump|strings|nl|tac|base64|openssl|dd|cp|mv|scp|rsync)\b[\s\S]*(?:^|[/. _=-])(?:\.env|env\.local|id_rsa|id_ed25519|credentials?|token|secret|passwd|shadow|\.npmrc|\.pypirc|\.netrc|aws_access_key|private[_-]?key)(?:\b|[/. _-])/i, reason: 'reading credential/secret files requires explicit approval' },
  // Target-based credential read: require approval for ANY command that references a credential
  // file, regardless of the reading binary (so non-listed readers like cut/tr/bat — or a future
  // tool — can't slip a secret to stdout for exfiltration).
  { re: /(?:^|[\s;&|=<>(])(?:~\/|\$HOME\/|\.\/)?(?:[\w./-]*\/)?(?:\.env(?:\.[\w-]+)?|env\.local|id_rsa|id_ed25519|id_dsa|id_ecdsa|\.npmrc|\.pypirc|\.netrc|\.git-credentials|\.htpasswd|\.pgpass|credentials|aws_access_key|aws_credentials|private[_-]?key|kubeconfig|\.dockerconfigjson|(?:secrets?|credentials?|auth|token|service[-_]?account|gcp[-_]?key|vault)\.(?:json|ya?ml|env))\b/i, reason: 'reading credential/secret files requires explicit approval' },
  { re: /\.(?:pem|p12|pfx|keystore|jks)\b/i, reason: 'reading private-key / keystore files requires explicit approval' },
  { re: /(?:^|[\s;&|])(?:cat|less|more|sed|awk|grep|rg|head|tail|cut|tr|xxd|od|hexdump|strings|nl|tac|base64|openssl|dd|cp|mv|bat)\b[\s\S]*(?:\.ssh|\.aws|\.gcloud|\.azure|\.kube)\//i, reason: 'reading credential directories requires explicit approval' },
  // Archive tools bundle credential directories without a reader binary in the command, so the
  // reader-keyed patterns above never see them. Approval only: archiving itself is legitimate.
  { re: /(?:^|[\s;&|])(?:tar|zip|7z|gzip|cpio)\b[\s\S]*(?:(?:~\/|\$HOME\/)?\.(?:ssh|aws|azure|gcloud|kube|gnupg)\b|(?:^|[\s/])\.env(?:[.\s/]|$))/i, reason: 'archiving credential directories requires explicit approval' },
]

const GIT_COMMIT_REASON = 'use TensorFold commit so Git hygiene and AI co-author stripping run first'
const GIT_PUSH_REASON = 'run Git hygiene before pushing to remote version control'
const GIT_GLOBAL_OPTIONS_WITH_VALUE = new Set([
  '-C',
  '-c',
  '--git-dir',
  '--work-tree',
  '--namespace',
  '--exec-path',
  '--config-env',
])

function shellPolicySource(input: string): string {
  let text = String(input || '')
  // ANSI-C quoting decodes before word splitting; $'\\x72\\x6d' executes rm.
  text = text.replace(/\$'((?:[^'\\]|\\.)*)'/g, (_match, body: string) => `'${decodeAnsiCEscapes(body)}'`)
  return text
    .replace(/\$\{IFS\}/g, ' ')
    .replace(/\$\{[A-Za-z_][A-Za-z0-9_]*:-\}/g, '')
    .replace(/\$\{:-([^{}]*)\}/g, '$1')
    .replace(/\$(?=['"])/g, '')
}

function shellWords(input: string): string[] {
  const words: string[] = []
  let word = ''
  let quote: "'" | '"' | '`' | null = null
  let escaped = false
  for (const ch of input) {
    if (escaped) {
      word += ch
      escaped = false
      continue
    }
    if (ch === '\\') {
      escaped = true
      continue
    }
    if (quote) {
      if (ch === quote) quote = null
      else word += ch
      continue
    }
    if (ch === "'" || ch === '"' || ch === '`') {
      quote = ch
      continue
    }
    if (/\s/.test(ch)) {
      if (word) words.push(word)
      word = ''
      continue
    }
    word += ch
  }
  if (word) words.push(word)
  return words
}

/**
 * Collapse shell quote-concatenation and backslash escapes before policy matching.
 * zsh executes `r""m`, `r'm'`, and `r\m` as the same `rm` token; inspecting only
 * the raw spelling lets model-controlled commands hide destructive/network binaries
 * from every regex-based gate while the shell still executes them normally.
 */
export function commandPolicyText(input: string): string {
  return shellWords(shellPolicySource(input)).join(' ')
}

/**
 * Decodes the escape forms zsh honors inside $'...' — the ones that can build
 * arbitrary command bytes: hex/octal/unicode escapes, control letters, and the
 * standard C single-char escapes. Unknown escapes keep their literal character.
 */
export function decodeAnsiCEscapes(input: string): string {
  let out = ''
  for (let i = 0; i < input.length; i++) {
    const ch = input[i]
    if (ch !== '\\' || i + 1 >= input.length) { out += ch; continue }
    const next = input[++i]
    if (next === 'x') {
      const hex = input.slice(i + 1, i + 3).match(/^[0-9a-fA-F]{1,2}/)
      if (hex) { out += String.fromCharCode(parseInt(hex[0], 16)); i += hex[0].length; continue }
      out += 'x'; continue
    }
    if (next === 'u' || next === 'U') {
      const width = next === 'u' ? 4 : 8
      const hex = input.slice(i + 1, i + 1 + width).match(/^[0-9a-fA-F]+/)
      if (hex) {
        const code = parseInt(hex[0], 16)
        out += code <= 0x10ffff ? String.fromCodePoint(code) : ''
        i += hex[0].length
        continue
      }
      out += next; continue
    }
    if (next === '0') {
      const oct = input.slice(i + 1, i + 4).match(/^[0-7]{1,3}/)
      if (oct) { out += String.fromCharCode(parseInt(oct[0], 8)); i += oct[0].length; continue }
      out += '0'; continue
    }
    if (next === 'c' && i + 1 < input.length) {
      const ctrl = input[i + 1]
      if (/[a-zA-Z]/.test(ctrl)) { out += String.fromCharCode(ctrl.toUpperCase().charCodeAt(0) & 0x1f); i++; continue }
      out += 'c'; continue
    }
    const simple: Record<string, string> = { n: '\n', t: '\t', r: '\r', a: '\a', b: '\b', f: '\f', v: '\v', '\\': '\\', "'": "'", '"': '"', e: '\x1b' }
    out += simple[next] ?? next
  }
  return out
}

type ShellToken = { kind: 'word' | 'control' | 'redirect' | 'push' | 'pop', text: string }
type ShellFrame = { quote: "'" | '"' | null, delimiter: '(' | '`' }

/** Lex unquoted shell structure while preserving quoted and escaped argument text. */
function shellPolicyTokens(input: string): ShellToken[] {
  const tokens: ShellToken[] = []
  const frames: ShellFrame[] = []
  const source = shellPolicySource(input)
  let word = ''
  let quote: "'" | '"' | null = null
  let escaped = false
  const flush = () => {
    if (word) tokens.push({ kind: 'word', text: word })
    word = ''
  }
  for (let i = 0; i < source.length; i++) {
    const ch = source[i]
    if (escaped) { if (ch !== '\n') word += ch; escaped = false; continue }
    if (ch === '\\' && quote !== "'") { escaped = true; continue }
    if (quote === "'") {
      if (ch === "'") quote = null
      else word += ch
      continue
    }
    const top = frames[frames.length - 1]
    if (ch === '`') {
      flush()
      if (top?.delimiter === '`') {
        tokens.push({ kind: 'pop', text: '`' })
        quote = frames.pop()?.quote ?? null
      } else {
        tokens.push({ kind: 'push', text: '`' })
        frames.push({ quote, delimiter: '`' })
        quote = null
      }
      continue
    }
    const substitution = ch === '$' && source[i + 1] === '('
    if (substitution || (ch === '(' && !quote)) {
      flush()
      tokens.push({ kind: 'push', text: substitution ? '$(' : '(' })
      frames.push({ quote, delimiter: '(' })
      quote = null
      if (substitution) i++
      continue
    }
    if (ch === ')' && top?.delimiter === '(' && !quote) {
      flush()
      tokens.push({ kind: 'pop', text: ')' })
      quote = frames.pop()?.quote ?? null
      continue
    }
    if (ch === quote) { quote = null; continue }
    if (!quote && (ch === "'" || ch === '"')) { quote = ch; continue }
    if (quote) { word += ch; continue }
    if (ch === '\n' || ch === ';' || ch === '|' || (ch === '&' && !/[<>]/.test(source[i - 1] ?? ''))) {
      flush()
      tokens.push({ kind: 'control', text: ch })
      continue
    }
    if (ch === '>' || ch === '<') {
      flush()
      tokens.push({ kind: 'redirect', text: ch })
      continue
    }
    if (/\s/.test(ch)) { flush(); continue }
    word += ch
  }
  flush()
  return tokens
}

const COMMAND_WRAPPERS: Record<string, true> = { bash: true, command: true, env: true, nice: true, nohup: true, setsid: true, sh: true, stdbuf: true, time: true, timeout: true, xargs: true, zsh: true }
const WRAPPER_OPTION_OPERANDS: Record<string, number> = {
  'env:-C': 1, 'env:-S': 1, 'env:-u': 1, 'env:--chdir': 1, 'env:--split-string': 1, 'env:--unset': 1,
  'nice:-n': 1, 'nice:--adjustment': 1,
  'sh:-c': 1, 'bash:-c': 1, 'zsh:-c': 1,
  'stdbuf:-i': 1, 'stdbuf:-o': 1, 'stdbuf:-e': 1,
  'time:-f': 1, 'time:-o': 1, 'time:--format': 1, 'time:--output': 1,
  'xargs:-a': 1, 'xargs:-d': 1, 'xargs:-E': 1, 'xargs:-e': 1, 'xargs:-f': 1, 'xargs:-I': 1,
  'xargs:-L': 1, 'xargs:-n': 1, 'xargs:-P': 1, 'xargs:-s': 1,
  'xargs:--arg-file': 1, 'xargs:--delimiter': 1, 'xargs:--eof': 1, 'xargs:--exit': 1, 'xargs:--max-args': 1,
  'xargs:--max-lines': 1, 'xargs:--max-procs': 1, 'xargs:--replace': 1, 'xargs:--size': 1,
}
const RECURSIVE_WRAPPER_OPTIONS = new Set(['env:-S', 'env:--split-string', 'sh:-c', 'bash:-c', 'zsh:-c'])
const LOCAL_ONLY_EXECUTABLES = new Set([
  'awk', 'basename', 'cat', 'cd', 'chmod', 'command', 'cp', 'cut', 'date', 'dirname', 'echo', 'env',
  'false', 'head', 'less', 'ln', 'ls', 'mkdir', 'more', 'mv', 'nice', 'nl', 'nohup', 'pwd', 'printf',
  'readlink', 'realpath', 'rm', 'rmdir', 'sed', 'setsid', 'sleep', 'sort', 'stdbuf', 'tail', 'tee',
  'time', 'timeout', 'touch', 'tr', 'true', 'uniq', 'wc', 'which', 'xargs',
])

type CommandState = { start: boolean, wrapper: string | null, optionOperands: number, wrapperOperands: number, recurse: boolean }
type CommandInvocation = { name: string, start: number, end: number, text?: string }
const initialCommandState = (): CommandState => ({ start: true, wrapper: null, optionOperands: 0, wrapperOperands: 0, recurse: false })

function normalizedCommandTokens(input: string, invocations?: CommandInvocation[]): string {
  let state = initialCommandState()
  const states: CommandState[] = []
  const out: string[] = []
  let active: CommandInvocation[] = []
  const finishCommand = () => {
    for (const invocation of active) invocation.end = out.length
    active = []
  }
  for (const token of shellPolicyTokens(input)) {
    if (token.kind === 'push') {
      finishCommand()
      states.push({ ...state })
      state = initialCommandState()
      out.push(token.text)
      continue
    }
    if (token.kind === 'pop') {
      finishCommand()
      state = states.pop() ?? initialCommandState()
      out.push(token.text)
      continue
    }
    if (token.kind === 'control') {
      finishCommand()
      state = initialCommandState()
      out.push(token.text)
      continue
    }
    if (token.kind === 'redirect') { out.push(token.text); continue }
    if (!state.start) { out.push(token.text); continue }
    if (state.recurse) {
      out.push(normalizedCommandTokens(token.text, invocations))
      state = initialCommandState()
      state.start = false
      continue
    }
    if (state.optionOperands > 0) {
      state.optionOperands--
      out.push(token.text)
      continue
    }
    if (state.wrapperOperands > 0) {
      state.wrapperOperands--
      out.push(token.text)
      continue
    }
    if (/^[A-Za-z_][A-Za-z0-9_]*=/.test(token.text)) { out.push(token.text); continue }
    if (state.wrapper && token.text.startsWith('-')) {
      const lookup = `${state.wrapper}:${token.text}`
      state.optionOperands = token.text.includes('=') ? 0 : (WRAPPER_OPTION_OPERANDS[lookup] ?? 0)
      if (RECURSIVE_WRAPPER_OPTIONS.has(lookup)) state.recurse = true
      out.push(token.text)
      continue
    }
    const name = token.text.slice(token.text.lastIndexOf('/') + 1).toLowerCase()
    if (invocations) {
      const invocation = { name, start: out.length, end: out.length }
      invocations.push(invocation)
      active.push(invocation)
    }
    out.push(name)
    if (COMMAND_WRAPPERS[name]) {
      state.wrapper = name
      state.wrapperOperands = name === 'timeout' ? 1 : 0
    } else {
      state = initialCommandState()
      state.start = false
    }
  }
  finishCommand()
  if (invocations) {
    for (const invocation of invocations) {
      if (invocation.text === undefined) invocation.text = out.slice(invocation.start, invocation.end).join(' ')
    }
  }
  return out.join(' ')
}

/** Command-position executable paths and wrappers become their shell-equivalent executable names. */
export function classifyCommandTokens(input: string): string {
  return normalizedCommandTokens(input)
}

function commandPolicyCandidates(input: string): string[] {
  const canonical = commandPolicyText(input)
  const classified = classifyCommandTokens(input)
  return [...new Set([input, canonical, classified].filter(Boolean))]
}

function gitSubcommand(words: string[], gitIndex: number): string | null {
  for (let i = gitIndex + 1; i < words.length; i++) {
    const word = words[i]
    if (!word) continue
    if (word === '--') continue
    if (GIT_GLOBAL_OPTIONS_WITH_VALUE.has(word)) {
      i++
      continue
    }
    if ([...GIT_GLOBAL_OPTIONS_WITH_VALUE].some((opt) => word.startsWith(`${opt}=`))) continue
    if (word.startsWith('-C') && word.length > 2) continue
    if (word.startsWith('-c') && word.length > 2) continue
    if (word.startsWith('-')) continue
    return word.toLowerCase()
  }
  return null
}

// `git`, optionally invoked by an absolute/relative path (e.g. `/usr/bin/git`, `./git`), so the
// commit/push hygiene gate can't be bypassed by spelling out the binary's full path.
function isGitWord(word: string): boolean {
  const basename = word.replace(/^.*\//, '')
  return basename.toLowerCase() === 'git'
}

function blockedGitReason(command: string): string | null {
  for (const match of command.matchAll(/(?:^|[\s;&|"'`])(?:[^\s;&|"'`]*\/)?git\b([^;&|]*)/gi)) {
    const words = shellWords(`git${match[1] || ''}`)
    const subcommand = gitSubcommand(words, 0)
    if (subcommand === 'commit') return GIT_COMMIT_REASON
    if (subcommand === 'push') return GIT_PUSH_REASON
  }
  const words = shellWords(command)
  for (let i = 0; i < words.length; i++) {
    if (!isGitWord(words[i])) continue
    const subcommand = gitSubcommand(words, i)
    if (subcommand === 'commit') return GIT_COMMIT_REASON
    if (subcommand === 'push') return GIT_PUSH_REASON
  }
  return null
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

// `~` and `$HOME` are caught by the static patterns above, but a path can also reach the same
// directory via its resolved absolute form (e.g. `/Users/alice/Documents`) without ever spelling
// `~` or `$HOME`. Resolve the real home dir at call time and check for `rm -rf`/`rsync --delete`
// of any subpath under it.
function blockedHomeAbsoluteReason(command: string, home = os.homedir()): string | null {
  if (!home || home === '/' || home === '~') return null
  const escaped = escapeRegExp(home)
  // The resolved home appearing as a subpath target (e.g. `/Users/alice/Documents`).
  const rm = new RegExp(`\\brm\\s+(?:-[^\\s]*[rf][^\\s]*|--recursive|--force)\\b[\\s\\S]*?["']?${escaped}["']?\\/[^\\s"']+`, 'i')
  if (rm.test(command)) return 'refuses recursive deletion under the home directory'
  // The bare resolved home root itself (e.g. `rm -rf /Users/alice`), not only subpaths under it --
  // a trailing slash, closing quote, or whitespace/end must follow the home path.
  const rmRoot = new RegExp(`\\brm\\s+(?:-[^\\s]*[rf][^\\s]*|--recursive|--force)\\b[\\s\\S]*?["']?${escaped}["']?(?:\\/?["']?)(?:\\s|$)`, 'i')
  if (rmRoot.test(command)) return 'refuses recursive deletion of the home directory'
  const rsync = new RegExp(`\\brsync\\b[\\s\\S]*--delete\\b[\\s\\S]*?["']?${escaped}(?:\\/[^\\s"']*)?`, 'i')
  if (rsync.test(command)) return 'refuses rsync --delete targeting the home directory'
  return null
}

export function blockedCommandReason(command: string): string | null {
  const normalized = String(command || '').trim()
  if (!normalized) return 'command is empty'
  for (const candidate of commandPolicyCandidates(normalized)) {
    // Canonicalized spelling included: zsh runs `g${:-i}t push` as `git push`, so the
    // hygiene gate must match the command the shell will execute, not only the raw text.
    const gitReason = blockedGitReason(candidate)
    if (gitReason) return gitReason
    for (const { re, reason } of BLOCKED_COMMAND_PATTERNS) {
      if (re.test(candidate)) return reason
    }
    const homeReason = blockedHomeAbsoluteReason(candidate)
    if (homeReason) return homeReason
  }
  return null
}

export function commandApprovalReason(command: string): string | null {
  const normalized = String(command || '').trim()
  const classified = classifyCommandTokens(normalized)
  const networkReason = commandNetworkReason(normalized)
  if (networkReason) {
    if (networkReason === 'network fetch command') return 'network fetch commands require explicit approval'
    if (networkReason === 'network shell or file-transfer command') return 'network shell/file-transfer commands require explicit approval'
    if (networkReason === 'network package command') return 'package install/execution commands require explicit approval'
    if (networkReason === 'shell network redirection') return 'shell network redirection requires explicit approval'
    if (networkReason === 'inline interpreter network access') return 'inline interpreter network access requires explicit approval'
  }
  for (const candidate of [classified]) {
    for (const { re, reason } of APPROVAL_COMMAND_PATTERNS) {
      if (re.test(candidate)) return reason
    }
  }
  return null
}

/** Network-bearing shell syntax used by capability-scoped workers (including SSH backends). */
function unrecognizedUnsandboxedCommand(command: string): boolean {
  const invocations: CommandInvocation[] = []
  normalizedCommandTokens(command, invocations)
  return invocations.length === 0 || invocations.some(({ name }) => !LOCAL_ONLY_EXECUTABLES.has(name))
}

export function commandNetworkReason(command: string, failClosedUnsandboxed = false): string | null {
  const normalized = String(command || '').trim()
  if (!normalized) return null
  const directReasons = new Map([
    ['curl', 'network fetch command'], ['wget', 'network fetch command'], ['http', 'network fetch command'], ['https', 'network fetch command'],
    ['ssh', 'network shell or file-transfer command'], ['scp', 'network shell or file-transfer command'],
    ['sftp', 'network shell or file-transfer command'], ['rsync', 'network shell or file-transfer command'],
    ['nc', 'network shell or file-transfer command'], ['ncat', 'network shell or file-transfer command'],
    ['netcat', 'network shell or file-transfer command'], ['socat', 'network shell or file-transfer command'],
  ])
  const invocations: CommandInvocation[] = []
  normalizedCommandTokens(normalized, invocations)
  for (const invocation of invocations) {
    const direct = directReasons.get(invocation.name)
    if (direct) return direct
    const text = invocation.text ?? invocation.name
    if (/^(?:npm|pnpm|yarn|bun)\s+(?:install|add|dlx|exec|create)\b/i.test(text)) return 'network package command'
    if (/^(?:pip|pip3|python3?\s+-m\s+pip)\s+install\b/i.test(text)) return 'network package command'
    if (/^(?:python3?|node|ruby|perl|deno|bun|php)\b(?=[\s\S]*\s-{1,2}(?:c|e|m|M|r|eval|require|exec))[\s\S]*(?:socket|urllib|requests|http\.client|net\/http|io::socket|net::|\bnet\b|\bdgram\b|\bdns\b|https?:\/\/|fetch\s*\(|http\.server|\bcurl\b|\bwget\b|os\.system|subprocess|popen|\bsystem\s*\(|require\s*\(\s*['"](?:net|http|https|dgram|dns))/i.test(text)) return 'inline interpreter network access'
  }
  if (/\/dev\/(?:tcp|udp)\//i.test(normalized)) return 'shell network redirection'
  return failClosedUnsandboxed && unrecognizedUnsandboxedCommand(normalized) ? 'unrecognized shell command on an unsandboxed SSH host' : null
}

// Mirrors the shell credential-read block for the file-read tool, so the agent
// can't bypass blockedCommandReason by calling read_file on a known secret path.
const BLOCKED_READ_PATTERNS: Array<{ re: RegExp; reason: string }> = [
  { re: /(^|\/)\.env(\.[\w.-]+)?$/i, reason: 'refuses reading .env secret files' },
  { re: /(^|\/)\.(ssh|aws|azure|gcloud|kube|docker)(\/|$)/i, reason: 'refuses reading credential directories' },
  { re: /(^|\/)(id_rsa|id_dsa|id_ecdsa|id_ed25519|\.netrc|credentials)$/i, reason: 'refuses reading private keys / credential files' },
  // Credential carriers that use an extension (the bare-name patterns above miss these).
  { re: /(^|\/)(?:secrets?|credentials?|auth|token|service[-_]?account|gcp[-_]?key|aws_credentials|kubeconfig)\.(json|ya?ml|env)$/i, reason: 'refuses reading secrets/credentials config files' },
  { re: /(^|\/)(kubeconfig|\.dockerconfigjson)$/i, reason: 'refuses reading kube/docker credential files' },
  // Common in-repo credential carriers (esp. relevant to repo-driven reads like @imports).
  { re: /(^|\/)\.git(\/|$)/i, reason: 'refuses reading the .git directory' },
  { re: /(^|\/)(\.npmrc|\.pypirc|\.dockercfg|\.git-credentials|\.htpasswd|\.pgpass)$/i, reason: 'refuses reading package/registry credential files' },
  { re: /\.(pem|key|pfx|p12|keystore|jks)$/i, reason: 'refuses reading private-key / keystore files' },
  { re: /\.tfvars(\.json)?$/i, reason: 'refuses reading Terraform variable files (often secrets)' },
  { re: /(^|\/)(service[-_]?account|gcp[-_]?key|secrets?)\.(json|ya?ml)$/i, reason: 'refuses reading service-account / secrets files' },
]

export function blockedReadReason(relPath: string): string | null {
  const p = String(relPath || '').trim().replace(/^~\//, '').replace(/^\$HOME\//i, '')
  if (!p) return null
  for (const { re, reason } of BLOCKED_READ_PATTERNS) {
    if (re.test(p)) return reason
  }
  return null
}

// Workspace mutation tools are intentionally allowed to edit ordinary project files, but Git's
// private metadata is not ordinary project content. Writing `.git/config`, `.git/hooks/*`, or the
// worktree `.git` indirection file can turn an innocuous later Git action into host code execution
// or silently redirect the repository. Keep this path-based so write_file, edit_file, apply_patch,
// local workspaces, and SSH workspaces all share the same fail-closed rule.
const BLOCKED_WRITE_PATTERNS: Array<{ re: RegExp; reason: string }> = [
  { re: /(^|\/)\.git(\/|$)/i, reason: 'refuses writing Git private metadata (.git), including hooks and repository config' },
]

export function blockedWriteReason(relPath: string): string | null {
  const p = String(relPath || '')
    .trim()
    .replace(/\\/g, '/')
    .replace(/^\.\/+/, '')
  if (!p) return null
  for (const { re, reason } of BLOCKED_WRITE_PATTERNS) {
    if (re.test(p)) return reason
  }
  return null
}
