export const PROXY_ENV_KEYS = [
  'HTTPS_PROXY', 'HTTP_PROXY', 'ALL_PROXY', 'NO_PROXY',
  'https_proxy', 'http_proxy', 'all_proxy', 'no_proxy',
] as const

export type PullProxyResolver = (url: string) => Promise<string>
let systemProxyResolver: PullProxyResolver | undefined

// The desktop host supplies Electron's resolver; the engine service remains Electron-free.
export function setPullProxyResolver(resolve: PullProxyResolver): void {
  systemProxyResolver = resolve
}

function proxyAddress(value: string): string {
  try {
    const url = new URL(value)
    const port = url.port || (url.protocol === 'https:' ? '443' : url.protocol.startsWith('socks') ? '1080' : '80')
    return `${url.protocol}//${url.hostname}:${port}`
  } catch { return '[configured proxy]' }
}

function systemProxy(route: string): string | null {
  // Chromium orders PAC results by preference; DIRECT must not override an earlier proxy.
  const first = route.split(';').map((entry) => entry.trim()).find(Boolean)
  if (!first || first === 'DIRECT') return null
  const match = /^(PROXY|HTTP|HTTPS|SOCKS|SOCKS4|SOCKS5)\s+(\S+)$/i.exec(first)
  if (!match) throw new Error('The system proxy setting is unsupported. Set a proxy environment variable and retry.')
  const schemes: Record<string, string> = { PROXY: 'http', HTTP: 'http', HTTPS: 'https', SOCKS: 'socks4', SOCKS4: 'socks4', SOCKS5: 'socks5' }
  const scheme = schemes[match[1].toUpperCase()]
  const value = `${scheme}://${match[2]}`
  const url = new URL(value)
  if (!url.hostname || !/:\d+$/.test(match[2]) || (url.pathname !== '' && url.pathname !== '/') || url.search || url.hash) {
    throw new Error('The system proxy address is invalid. Check proxy settings and retry.')
  }
  return value
}

export async function pullEnvironment(
  env: NodeJS.ProcessEnv,
  platform: NodeJS.Platform = process.platform,
  resolve: PullProxyResolver | undefined = systemProxyResolver,
  timeoutMs = 5000,
): Promise<NodeJS.ProcessEnv> {
  const child = { ...env }
  if (platform !== 'darwin' || PROXY_ENV_KEYS.some((key) => env[key] !== undefined) || !resolve) return child
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    const route = await Promise.race([
      Promise.resolve().then(() => resolve(env.HF_ENDPOINT || 'https://huggingface.co')),
      new Promise<never>((_yes, no) => {
        timer = setTimeout(() => no(new Error('timeout')), timeoutMs)
      }),
    ])
    const proxy = systemProxy(route)
    if (proxy) child.HTTPS_PROXY = child.HTTP_PROXY = child.ALL_PROXY = proxy
    return child
  } catch {
    // Resolver errors can contain authenticated proxy URLs; never forward their text.
    throw new Error('Unable to read the system proxy, or the lookup timed out. Check proxy settings and retry.')
  } finally {
    if (timer) clearTimeout(timer)
  }
}

export function redactPullDiagnostic(text: string, env: NodeJS.ProcessEnv): string {
  const secrets = new Set<string>()
  for (const key of PROXY_ENV_KEYS) {
    if (key.toLowerCase() === 'no_proxy' || !env[key]) continue
    try {
      const value = env[key]!
      const url = new URL(value.includes('://') ? value : `http://${value}`)
      for (const value of [url.username, url.password]) {
        if (!value) continue
        secrets.add(value)
        try { secrets.add(decodeURIComponent(value)) } catch { /* Keep the encoded value. */ }
      }
    } catch { /* The diagnostic URL scrubber also removes unparseable proxy URLs. */ }
  }
  let safe = text.replace(/\b(?:https?|socks(?:4a?|5h?)?):\/\/[^\s"'<>]+/gi, (value) => proxyAddress(value))
  for (const secret of [...secrets].sort((a, b) => b.length - a.length)) safe = safe.split(secret).join('[redacted]')
  return safe
}

export function pullFailure(diagnostic: string, env: NodeJS.ProcessEnv, stalled = false): string {
  const proxy = ['HTTPS_PROXY', 'https_proxy', 'ALL_PROXY', 'all_proxy', 'HTTP_PROXY', 'http_proxy']
    .map((key) => env[key]).find(Boolean)
  const via = proxy ? ` through the proxy (${proxyAddress(proxy)})` : ''
  const safe = redactPullDiagnostic(diagnostic, env)
  if (stalled) return `Download timed out: no new model bytes were saved for 60 seconds. The Hugging Face Hub may be unreachable${via}. Check your connection and proxy settings, then retry.`
  if (/timeout|timed?\s*out/i.test(diagnostic)) return `The download connection to the Hugging Face Hub${via} timed out. Check your connection and proxy settings, then retry. ${safe}`
  if (/proxy|connect|unreachable|resolve|dns|network|tls|certificate/i.test(diagnostic)) return `Unable to connect to the Hugging Face Hub${via}. The connection or proxy may have refused the request. Check your connection and proxy settings, then retry. ${safe}`
  if (/^tensorfold-native pull exited with/.test(diagnostic)) return `The download stopped before completing. The Hugging Face Hub connection${via} may have failed. Check your connection and proxy settings, then retry. ${safe}`
  return safe || 'The download failed before completing. Check your connection and proxy settings, then retry.'
}
