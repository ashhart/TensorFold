// Child environment for the CLI transports (claude/codex/pi/cursor-agent/gemini). Never
// spread process.env into a CLI child: room seats run the CLI autonomously on transcripts
// other seats (or a poisoned file) can influence, and a read-only `env`/`printenv` is a
// legitimate non-mutating command — inherited *_API_KEY/*_TOKEN/*_SECRET vars would be
// printable into a seat reply that rooms then broadcast to seats on OTHER providers. Only
// allowlisted basics pass through; a provider's own credential reaches the child through
// provider.cliEnv or the transport's documented key var (CURSOR_API_KEY/GEMINI_API_KEY).
// This also subsumes local-crypto's VONTRA_SECRET_KEY_BASE64 scrub for these children.
const CLI_ENV_ALLOWLIST = [
  // runtime basics + locale/identity
  'PATH', 'HOME', 'TMPDIR', 'TMP', 'TEMP',
  'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'SHELL', 'USER', 'LOGNAME',
  // XDG config dirs (CLI config discovery)
  'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'XDG_CACHE_HOME', 'XDG_STATE_HOME',
  // platform essentials
  '__CF_USER_TEXT_ENCODING',
  'SystemRoot', 'SystemDrive', 'USERPROFILE', 'APPDATA', 'LOCALAPPDATA', 'COMSPEC', 'HOMEDRIVE', 'HOMEPATH', 'PATHEXT',
  // CLI config-dir overrides (paths, not credentials)
  'CODEX_HOME', 'CLAUDE_CONFIG_DIR', 'CLAUDE_HOME', 'CLAUDE_CONFIG_FILE',
  // connectivity, not credentials: interactive threads legitimately rely on ambient
  // proxy / custom-CA / git-ssh settings for the network and git work the CLI performs
  'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'NO_PROXY',
  'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
  'NODE_EXTRA_CA_CERTS', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
  'SSH_AUTH_SOCK',
]

/**
 * Build a CLI child's env from the allowlist plus caller-supplied extras (provider.cliEnv,
 * the transport's key mapping). Extras win verbatim over inherited values; everything in
 * `inherited` that is not allowlisted is dropped.
 */
export function cliSpawnEnv(extra?: Record<string, string>, inherited: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {}
  for (const key of CLI_ENV_ALLOWLIST) {
    const value = inherited[key]
    if (value !== undefined) env[key] = value
  }
  return { ...env, ...(extra ?? {}) }
}
