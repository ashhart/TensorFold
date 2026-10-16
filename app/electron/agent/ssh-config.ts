// OpenSSH client-config parsing for SSH host discovery and effective-settings
// resolution. These are pure functions: filesystem access only happens through
// readers injected by the caller, which keeps the whole module unit-testable
// without touching ~/.ssh or a real SSH server.
//
// Scope / limitations (intentional for v1):
//  - `Match` blocks are parsed but never applied (their directives match nothing).
//    Concrete `Host` aliases cover the host-picker use case.
//  - Token-level `=` is treated as a separator, matching OpenSSH's lexer.
//  - First-obtained-value-wins per keyword, matching OpenSSH semantics, except
//    `IdentityFile` which accumulates across all matching blocks.

export interface SshConfigDirective {
  keyword: string // lower-cased for matching
  originalKeyword: string
  args: string[]
}

export interface SshConfigHostBlock {
  patterns: string[]
  directives: SshConfigDirective[]
  /** True for `Match` blocks, which we parse but do not evaluate. */
  isMatch?: boolean
}

export interface SshConfigModel {
  /** Directives appearing before any Host/Match block apply to every host. */
  global: SshConfigDirective[]
  hosts: SshConfigHostBlock[]
}

export interface ResolvedSshSettings {
  alias: string
  hostName: string
  user?: string
  port: number
  identityFiles: string[]
  proxyJump?: string
  forwardAgent?: boolean
  /** First-obtained value per keyword (lower-cased), for less-common options. */
  values: Record<string, string>
}

const DEFAULT_PORT = 22

/**
 * Split a config line into tokens. Whitespace and unquoted `=` are separators;
 * double quotes group a token and are stripped. Mirrors OpenSSH's lexer closely
 * enough for the keywords we care about.
 */
export function tokenizeSshConfigLine(line: string): string[] {
  const tokens: string[] = []
  let cur = ''
  let inQuote = false
  let sawChar = false
  for (let i = 0; i < line.length; i++) {
    const ch = line[i]
    if (inQuote) {
      if (ch === '"') inQuote = false
      else cur += ch
      continue
    }
    if (ch === '"') {
      inQuote = true
      sawChar = true
      continue
    }
    if (ch === ' ' || ch === '\t' || ch === '=') {
      if (sawChar) {
        tokens.push(cur)
        cur = ''
        sawChar = false
      }
      continue
    }
    cur += ch
    sawChar = true
  }
  if (sawChar) tokens.push(cur)
  return tokens
}

/** Parse already-include-expanded config text into an ordered model. */
export function parseSshConfig(text: string): SshConfigModel {
  const global: SshConfigDirective[] = []
  const hosts: SshConfigHostBlock[] = []
  let current: SshConfigHostBlock | null = null

  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) continue
    const tokens = tokenizeSshConfigLine(line)
    if (!tokens.length) continue
    const originalKeyword = tokens[0]
    const keyword = originalKeyword.toLowerCase()
    const args = tokens.slice(1)

    if (keyword === 'host') {
      current = { patterns: args, directives: [] }
      hosts.push(current)
      continue
    }
    if (keyword === 'match') {
      // Parsed but not evaluated: empty patterns means it never matches a host.
      current = { patterns: [], directives: [], isMatch: true }
      hosts.push(current)
      continue
    }
    const directive: SshConfigDirective = { keyword, originalKeyword, args }
    if (current) current.directives.push(directive)
    else global.push(directive)
  }

  return { global, hosts }
}

function patternToRegExp(pattern: string): RegExp {
  const escaped = pattern
    .replace(/[.+^${}()|[\]\\]/g, '\\$&')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.')
  return new RegExp(`^${escaped}$`)
}

/** OpenSSH host matching: at least one positive pattern, no negated pattern. */
export function sshHostMatches(patterns: string[], alias: string): boolean {
  let matched = false
  for (const pattern of patterns) {
    if (pattern.startsWith('!')) {
      if (patternToRegExp(pattern.slice(1)).test(alias)) return false
    } else if (patternToRegExp(pattern).test(alias)) {
      matched = true
    }
  }
  return matched
}

function isConcreteHostPattern(pattern: string): boolean {
  return !/[*?!]/.test(pattern)
}

/**
 * List the concrete (non-wildcard) Host aliases declared in the config, in the
 * order they first appear. These are the entries worth showing in a host picker.
 */
export function listSshConfigHosts(text: string): string[] {
  const model = parseSshConfig(text)
  const out: string[] = []
  const seen = new Set<string>()
  for (const block of model.hosts) {
    if (block.isMatch) continue
    for (const pattern of block.patterns) {
      if (isConcreteHostPattern(pattern) && !seen.has(pattern)) {
        seen.add(pattern)
        out.push(pattern)
      }
    }
  }
  return out
}

/** Resolve the effective settings OpenSSH would use for a given host alias. */
export function resolveSshHostSettings(text: string, alias: string): ResolvedSshSettings {
  const model = parseSshConfig(text)
  const values: Record<string, string> = {}
  const identityFiles: string[] = []

  const apply = (directives: SshConfigDirective[]) => {
    for (const directive of directives) {
      if (directive.keyword === 'identityfile') {
        for (const file of directive.args) {
          if (!identityFiles.includes(file)) identityFiles.push(file)
        }
        continue
      }
      if (!(directive.keyword in values)) {
        values[directive.keyword] = directive.args.join(' ')
      }
    }
  }

  apply(model.global)
  for (const block of model.hosts) {
    if (block.isMatch) continue
    if (sshHostMatches(block.patterns, alias)) apply(block.directives)
  }

  const portNum = Number(values.port)
  return {
    alias,
    hostName: values.hostname || alias,
    user: values.user,
    port: Number.isInteger(portNum) && portNum > 0 ? portNum : DEFAULT_PORT,
    identityFiles,
    proxyJump: values.proxyjump,
    forwardAgent: values.forwardagent ? /^yes$/i.test(values.forwardagent) : undefined,
    values,
  }
}

/** Expand a leading `~` or `~/` to the given home directory. */
export function expandTilde(input: string, homeDir: string): string {
  if (input === '~') return homeDir
  if (input.startsWith('~/')) return `${homeDir}/${input.slice(2)}`
  return input
}

export interface SshIncludeResolver {
  /** Resolve an Include argument (may contain globs) to concrete file paths. */
  resolve: (pattern: string, baseDir: string) => string[]
  /** Read a resolved file; return null if unreadable. */
  read: (path: string) => string | null
  /** Base directory for relative Include patterns (typically ~/.ssh). */
  baseDir: string
  homeDir: string
}

const MAX_INCLUDE_DEPTH = 16

/**
 * Inline `Include` directives by textual substitution before parsing. Inlining
 * preserves Host-block context, so the expanded text parses identically to how
 * OpenSSH would read the include chain. Cycles are bounded by depth.
 */
export function expandSshConfigIncludes(
  text: string,
  resolver: SshIncludeResolver,
  depth = 0,
): string {
  if (depth >= MAX_INCLUDE_DEPTH) return ''
  const out: string[] = []
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim()
    if (!line || line.startsWith('#')) {
      out.push(raw)
      continue
    }
    const tokens = tokenizeSshConfigLine(line)
    if (tokens[0]?.toLowerCase() !== 'include') {
      out.push(raw)
      continue
    }
    for (const pattern of tokens.slice(1)) {
      const expanded = expandTilde(pattern, resolver.homeDir)
      for (const file of resolver.resolve(expanded, resolver.baseDir)) {
        const content = resolver.read(file)
        if (content != null) {
          out.push(expandSshConfigIncludes(content, resolver, depth + 1))
        }
      }
    }
  }
  return out.join('\n')
}
