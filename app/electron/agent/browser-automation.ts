import { isGloballyRoutableIp, isLoopbackHost, isPrivateHost } from './security'

export const BROWSER_AUTOMATION_ACTIONS = [
  'navigate',
  'click',
  'type',
  'fill_form',
  'tab_new',
  'tab_switch',
  'tab_close',
  'screenshot',
  'console_logs',
  'network_errors',
  'accessibility_snapshot',
] as const

export type BrowserAutomationAction = typeof BROWSER_AUTOMATION_ACTIONS[number]
export type BrowserAutomationLoadState = 'idle' | 'loading' | 'loaded' | 'error'

export interface BrowserAutomationFormField {
  selector?: string
  ref?: string
  value: string
}

export interface BrowserAutomationInput {
  action: BrowserAutomationAction
  sessionId?: string
  tabId?: string
  url?: string
  selector?: string
  ref?: string
  text?: string
  fields?: BrowserAutomationFormField[]
  note?: string
  includeSnapshot?: boolean
  captureScreenshot?: boolean
  screenshotName?: string
  waitMs?: number
}

export interface BrowserAutomationConsoleMessage {
  level: 'log' | 'info' | 'warning' | 'error'
  text: string
  at: number
}

export interface BrowserAutomationNetworkError {
  url: string
  method?: string
  status?: number | null
  errorText: string
  at: number
}

export interface BrowserAutomationScreenshot {
  path?: string
  title?: string
  capturedAt?: number
}

export interface BrowserAutomationResult {
  ok: boolean
  action: BrowserAutomationAction
  sessionId: string
  tabId: string
  url?: string
  title?: string
  loadState?: BrowserAutomationLoadState
  accessibilitySnapshot?: string
  screenshot?: BrowserAutomationScreenshot
  consoleMessages?: BrowserAutomationConsoleMessage[]
  networkErrors?: BrowserAutomationNetworkError[]
  note?: string
}

export type BrowserAutomationHostResolver = (hostname: string) => Promise<string[]>

export interface BrowserAutomationDestinationOptions {
  /** Exact URL.origin explicitly approved by a navigate/tab_new action for localhost UI testing. */
  allowedLoopbackOrigin?: string | null
}

const ACTIONS = new Set<string>(BROWSER_AUTOMATION_ACTIONS)

export function normalizeBrowserAutomationInput(input: Record<string, unknown>): BrowserAutomationInput {
  const action = String(input.action ?? '').trim().toLowerCase() as BrowserAutomationAction
  if (!ACTIONS.has(action)) throw new Error(`browser_automation action must be one of: ${BROWSER_AUTOMATION_ACTIONS.join(', ')}.`)

  const normalized: BrowserAutomationInput = {
    action,
    sessionId: cleanOptional(input.sessionId),
    tabId: cleanOptional(input.tabId),
    selector: cleanOptional(input.selector),
    ref: cleanOptional(input.ref),
    text: input.text == null ? undefined : String(input.text),
    note: cleanOptional(input.note),
    includeSnapshot: input.includeSnapshot === true,
    captureScreenshot: input.captureScreenshot === true,
    screenshotName: cleanOptional(input.screenshotName),
    waitMs: finitePositiveInt(input.waitMs, 0, 30_000),
  }

  if (input.url != null || action === 'navigate' || action === 'tab_new') {
    normalized.url = normalizeBrowserAutomationUrl(input.url)
  }

  if (action === 'click' && !normalized.selector && !normalized.ref) {
    throw new Error('browser_automation click requires a selector or ref.')
  }
  if (action === 'type') {
    if (!normalized.selector && !normalized.ref) throw new Error('browser_automation type requires a selector or ref.')
    if (!normalized.text || !normalized.text.trim()) throw new Error('browser_automation type requires text.')
  }
  if (action === 'fill_form') {
    normalized.fields = normalizeFields(input.fields)
    if (!normalized.fields.length) throw new Error('browser_automation fill_form requires at least one field.')
  }
  if ((action === 'tab_switch' || action === 'tab_close') && !normalized.tabId) {
    throw new Error(`browser_automation ${action} requires tabId.`)
  }

  return stripEmpty(normalized)
}

export function normalizeBrowserAutomationUrl(value: unknown): string {
  const raw = String(value ?? '').trim()
  if (!raw) throw new Error('browser_automation navigation requires a URL.')
  const withScheme = /^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `http://${raw}`
  const url = new URL(withScheme)
  if (!['http:', 'https:'].includes(url.protocol)) throw new Error('browser_automation URLs must use http or https.')
  if (url.username || url.password) throw new Error('browser_automation URLs must not include credentials.')
  if (isPrivateHost(url.hostname) && !isLoopbackHost(url.hostname)) {
    throw new Error('browser_automation URLs cannot target private, link-local, or metadata networks.')
  }
  return url.href
}

// Public-web SSRF boundary. Localhost is an explicit exception because browser automation
// is the product's UI-smoke mechanism (mission recipes and tool copy both require it); the
// exception is scoped to the exact protocol + hostname + port explicitly opened for the tab.
// A public-looking DNS name that resolves to loopback, RFC1918, link-local, ULA, metadata,
// or CGNAT is rejected.
export async function assertBrowserAutomationDestination(
  raw: string,
  resolveHost: BrowserAutomationHostResolver,
  options: BrowserAutomationDestinationOptions = {},
): Promise<void> {
  let url: URL
  try { url = new URL(raw) } catch { throw new Error('Browser automation destination is invalid.') }
  if (!['http:', 'https:', 'ws:', 'wss:'].includes(url.protocol)) {
    throw new Error('Browser automation network destinations must use http(s) or ws(s).')
  }
  if (url.username || url.password) throw new Error('Browser automation destinations must not include credentials.')
  if (isLoopbackHost(url.hostname)) {
    if (options.allowedLoopbackOrigin === url.origin) return
    throw new Error(`Browser automation blocked loopback origin ${url.origin} because this tab was not explicitly opened for that exact localhost origin.`)
  }
  if (isPrivateHost(url.hostname)) {
    throw new Error('Browser automation blocked a private, link-local, or metadata destination.')
  }

  let addresses: string[]
  try {
    addresses = await resolveHost(url.hostname)
  } catch (error: any) {
    throw new Error(`Browser automation could not safely resolve ${url.hostname}; navigation was blocked. (${error?.message || error})`)
  }
  if (!addresses.length) throw new Error(`Browser automation could not safely resolve ${url.hostname}; navigation was blocked.`)
  const unsafe = addresses.find((address) => !isGloballyRoutableIp(address))
  if (unsafe) {
    throw new Error(`Browser automation blocked ${url.hostname}: it resolves to a private or non-public address (${unsafe}).`)
  }
}

export async function assertBrowserAutomationRequestDestination(
  raw: string,
  resolveHost: BrowserAutomationHostResolver,
  options: BrowserAutomationDestinationOptions = {},
): Promise<void> {
  let url: URL
  try { url = new URL(raw) } catch { throw new Error('Browser automation request URL is invalid.') }
  // Non-network resources created by an already-isolated page do not create SSRF egress.
  if (url.protocol === 'data:' || url.protocol === 'blob:' || url.protocol === 'about:') return
  await assertBrowserAutomationDestination(url.href, resolveHost, options)
}

export function formatBrowserAutomationResult(result: BrowserAutomationResult): string {
  const lines = [
    `Action: ${result.action}`,
    `Session: ${result.sessionId}`,
    `Tab: ${result.tabId}`,
    result.url ? `URL: ${result.url}` : '',
    result.title ? `Title: ${result.title}` : '',
    result.loadState ? `Load state: ${result.loadState}` : '',
    result.note ? `Note: ${result.note}` : '',
    result.screenshot?.path ? `Screenshot: ${result.screenshot.path}${result.screenshot.title ? ` (${result.screenshot.title})` : ''}` : '',
  ].filter(Boolean)

  if (result.accessibilitySnapshot) {
    lines.push('', 'Accessibility snapshot:', result.accessibilitySnapshot)
  }
  if (result.consoleMessages?.length) {
    lines.push('', 'Console messages:')
    for (const message of result.consoleMessages) {
      lines.push(`- ${message.level}: ${message.text}`)
    }
  }
  if (result.networkErrors?.length) {
    lines.push('', 'Network errors:')
    for (const error of result.networkErrors) {
      const status = error.status != null ? `${error.status} ` : ''
      lines.push(`- ${error.method || 'GET'} ${error.url}: ${status}${error.errorText}`)
    }
  }

  return lines.join('\n')
}

function normalizeFields(value: unknown): BrowserAutomationFormField[] {
  const items = Array.isArray(value) ? value : []
  return items.map((item) => {
    const record = isRecord(item) ? item : {}
    const field: BrowserAutomationFormField = {
      selector: cleanOptional(record.selector),
      ref: cleanOptional(record.ref),
      value: String(record.value ?? ''),
    }
    if (!field.selector && !field.ref) throw new Error('browser_automation fill_form fields require a selector or ref.')
    return stripEmpty(field)
  })
}

function cleanOptional(value: unknown): string | undefined {
  const text = String(value ?? '').replace(/\s+/g, ' ').trim()
  return text || undefined
}

function finitePositiveInt(value: unknown, fallback: number, max: number): number | undefined {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return fallback || undefined
  return Math.min(Math.round(n), max)
}

function stripEmpty<T extends Record<string, any>>(value: T): T {
  const copy: Record<string, any> = {}
  for (const [key, item] of Object.entries(value)) {
    if (item !== undefined && item !== null && item !== '') copy[key] = item
  }
  return copy as T
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}
