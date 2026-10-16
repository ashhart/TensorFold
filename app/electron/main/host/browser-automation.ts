import { BrowserWindow, session, type Session } from 'electron'
import path from 'node:path'
import { promises as fs } from 'node:fs'
import { randomUUID } from 'node:crypto'
import { safeSegment, vontraPath } from '../../store'
import { assertBrowserAutomationDestination, assertBrowserAutomationRequestDestination } from '../../agent/browser-automation'
import { isLoopbackHost } from '../../agent/security'
import type { BrowserAutomationConsoleMessage, BrowserAutomationInput, BrowserAutomationNetworkError, BrowserAutomationResult } from '../../agent/browser-automation'
import { clearPersistentBrowserData } from './browser-data-purge'

interface BrowserAutomationTab {
  id: string
  window: BrowserWindow
  loadState: BrowserAutomationResult['loadState']
  consoleMessages: BrowserAutomationConsoleMessage[]
  networkErrors: BrowserAutomationNetworkError[]
  allowedLoopbackOrigin: string | null
}

interface BrowserAutomationSessionState {
  id: string
  tabs: Map<string, BrowserAutomationTab>
  activeTabId: string | null
}

const BROWSER_AUTOMATION_PARTITION = 'persist:vontra-browser-automation'
const browserAutomationSessions = new Map<string, BrowserAutomationSessionState>()
let browserAutomationNetworkHookInstalled = false
const HEADLESS_MODE = process.argv.includes('--headless-server') || process.env.VONTRA_HEADLESS === '1'

function browserAutomationSessionId(value: string | undefined): string {
  return value ? safeSegment(value) : 'default'
}

function ensureBrowserAutomationSession(id: string): BrowserAutomationSessionState {
  const existing = browserAutomationSessions.get(id)
  if (existing) return existing
  const next: BrowserAutomationSessionState = { id, tabs: new Map(), activeTabId: null }
  browserAutomationSessions.set(id, next)
  return next
}

function browserAutomationElectronSession() {
  const ses = session.fromPartition(BROWSER_AUTOMATION_PARTITION)
  ses.setPermissionRequestHandler((_contents, _permission, callback) => callback(false))
  ses.setPermissionCheckHandler(() => false)
  if (!browserAutomationNetworkHookInstalled) {
    browserAutomationNetworkHookInstalled = true
    // Guard every network request in the automation partition, not only the URL passed to
    // loadURL(). Chromium invokes onBeforeRequest again for redirect targets and subresources,
    // so a public page cannot pivot into an internal service. Resolution uses this SAME
    // Chromium session (and populates its resolver cache), avoiding a Node-DNS-check followed
    // by an unrelated browser resolution that is vulnerable to straightforward rebinding.
    ses.webRequest.onBeforeRequest((details, callback) => {
      const tab = browserAutomationTabForWebContents(details.webContentsId)
      void assertBrowserAutomationRequestDestination(
        details.url,
        (hostname) => resolveBrowserAutomationHost(ses, hostname),
        { allowedLoopbackOrigin: tab?.allowedLoopbackOrigin ?? null },
      ).then(
        () => {
          // A local UI tab that redirects/navigates onto the public web loses its localhost
          // exception before any later redirect or subresource can pivot back inside.
          if (tab && details.resourceType === 'mainFrame') {
            try { if (!isLoopbackHost(new URL(details.url).hostname)) tab.allowedLoopbackOrigin = null } catch { /* non-network URL */ }
          }
          callback({ cancel: false })
        },
        (error: any) => {
          if (tab) {
            pushBrowserAutomationNetworkError(tab, {
              url: details.url,
              method: details.method,
              status: null,
              errorText: String(error?.message || error || 'Blocked unsafe browser destination.'),
              at: Date.now(),
            })
          }
          callback({ cancel: true })
        },
      )
    })
    ses.webRequest.onCompleted((details) => {
      if (details.statusCode < 400) return
      const tab = browserAutomationTabForWebContents(details.webContentsId)
      if (!tab) return
      pushBrowserAutomationNetworkError(tab, {
        url: details.url,
        method: details.method,
        status: details.statusCode,
        errorText: details.statusLine || `HTTP ${details.statusCode}`,
        at: Date.now(),
      })
    })
    ses.webRequest.onErrorOccurred((details) => {
      const tab = browserAutomationTabForWebContents(details.webContentsId)
      if (!tab) return
      pushBrowserAutomationNetworkError(tab, {
        url: details.url,
        method: details.method,
        status: null,
        errorText: details.error,
        at: Date.now(),
      })
    })
  }
  return ses
}

async function resolveBrowserAutomationHost(ses: Session, hostname: string): Promise<string[]> {
  // Query both address families through Chromium. A mixed public/private answer is unsafe;
  // assertBrowserAutomationDestination rejects if ANY returned endpoint is private.
  const results = await Promise.allSettled([
    ses.resolveHost(hostname, { queryType: 'A', cacheUsage: 'allowed' }),
    ses.resolveHost(hostname, { queryType: 'AAAA', cacheUsage: 'allowed' }),
  ])
  const addresses = new Set<string>()
  for (const result of results) {
    if (result.status !== 'fulfilled') continue
    for (const endpoint of result.value.endpoints) if (endpoint.address) addresses.add(endpoint.address)
  }
  if (!addresses.size) {
    const failed = results.find((result): result is PromiseRejectedResult => result.status === 'rejected')
    throw failed?.reason ?? new Error(`No addresses resolved for ${hostname}.`)
  }
  return [...addresses]
}

function browserAutomationTabForWebContents(webContentsId: number | undefined): BrowserAutomationTab | null {
  if (webContentsId == null) return null
  for (const automation of browserAutomationSessions.values()) {
    for (const tab of automation.tabs.values()) {
      if (tab.window.webContents.id === webContentsId) return tab
    }
  }
  return null
}

function pushBrowserAutomationNetworkError(tab: BrowserAutomationTab, error: BrowserAutomationNetworkError) {
  tab.networkErrors = [...tab.networkErrors, error].slice(-80)
}

function pushBrowserAutomationConsoleMessage(tab: BrowserAutomationTab, message: BrowserAutomationConsoleMessage) {
  tab.consoleMessages = [...tab.consoleMessages, message].slice(-80)
}

function createBrowserAutomationTab(automation: BrowserAutomationSessionState): BrowserAutomationTab {
  browserAutomationElectronSession()
  const id = `tab-${randomUUID().slice(0, 8)}`
  const bw = new BrowserWindow({
    width: 1280,
    height: 800,
    show: false,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      sandbox: true,
      partition: BROWSER_AUTOMATION_PARTITION,
    },
  })
  const tab: BrowserAutomationTab = { id, window: bw, loadState: 'idle', consoleMessages: [], networkErrors: [], allowedLoopbackOrigin: null }
  // Remote pages cannot escape into unmanaged BrowserWindows. New tabs are created only by
  // the approval-gated tab_new tool action, which installs the same partition guard.
  bw.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
  bw.webContents.on('will-frame-navigate', (event) => {
    try {
      const protocol = new URL(event.url).protocol
      if (['http:', 'https:', 'data:', 'blob:', 'about:'].includes(protocol)) return
    } catch { /* block below */ }
    event.preventDefault()
  })
  bw.webContents.on('console-message', (_event, level, message) => {
    pushBrowserAutomationConsoleMessage(tab, {
      level: browserAutomationConsoleLevel(level),
      text: String(message || '').slice(0, 2000),
      at: Date.now(),
    })
  })
  bw.webContents.on('did-fail-load', (_event, _errorCode, errorDescription, validatedURL, isMainFrame) => {
    if (!isMainFrame) return
    tab.loadState = 'error'
    pushBrowserAutomationNetworkError(tab, {
      url: validatedURL || bw.webContents.getURL(),
      method: 'GET',
      status: null,
      errorText: errorDescription || 'Page load failed',
      at: Date.now(),
    })
  })
  bw.on('closed', () => {
    for (const state of browserAutomationSessions.values()) {
      if (!state.tabs.delete(id)) continue
      if (state.activeTabId === id) state.activeTabId = state.tabs.keys().next().value ?? null
    }
  })
  automation.tabs.set(id, tab)
  automation.activeTabId = id
  return tab
}

function activeBrowserAutomationTab(automation: BrowserAutomationSessionState): BrowserAutomationTab {
  const active = automation.activeTabId ? automation.tabs.get(automation.activeTabId) : null
  return active ?? createBrowserAutomationTab(automation)
}

function browserAutomationConsoleLevel(level: number | string): BrowserAutomationConsoleMessage['level'] {
  if (typeof level === 'string') {
    const value = level.toLowerCase()
    if (value === 'warning' || value === 'error' || value === 'info') return value
    return 'log'
  }
  if (level >= 3) return 'error'
  if (level === 2) return 'warning'
  if (level === 1) return 'info'
  return 'log'
}

export async function runBrowserAutomation(input: BrowserAutomationInput): Promise<BrowserAutomationResult> {
  const sessionId = browserAutomationSessionId(input.sessionId)
  if (HEADLESS_MODE) {
    return {
      ok: false,
      action: input.action,
      sessionId,
      tabId: input.tabId ?? 'unavailable',
      loadState: 'error',
      note: 'Browser automation is unavailable in headless mode.',
    }
  }

  const automation = ensureBrowserAutomationSession(sessionId)
  let tab = input.tabId ? automation.tabs.get(input.tabId) : activeBrowserAutomationTab(automation)
  if (!tab && input.action === 'tab_switch') throw new Error(`Browser automation tab not found: ${input.tabId}`)
  if (!tab) tab = createBrowserAutomationTab(automation)
  let ok = true
  let note = input.note

  try {
    switch (input.action) {
      case 'navigate':
        await loadBrowserAutomationUrl(tab, input.url!)
        break
      case 'tab_new':
        tab = createBrowserAutomationTab(automation)
        await loadBrowserAutomationUrl(tab, input.url!)
        break
      case 'tab_switch':
        automation.activeTabId = tab.id
        note = note || `Switched to ${tab.id}.`
        break
      case 'tab_close': {
        const closingId = tab.id
        tab.window.close()
        automation.tabs.delete(closingId)
        if (automation.activeTabId === closingId) automation.activeTabId = automation.tabs.keys().next().value ?? null
        return {
          ok: true,
          action: input.action,
          sessionId,
          tabId: closingId,
          loadState: 'idle',
          note: note || `Closed ${closingId}.`,
        }
      }
      case 'click':
      case 'type':
      case 'fill_form':
        note = await runBrowserDomAction(tab, input)
        break
      case 'screenshot':
      case 'console_logs':
      case 'network_errors':
      case 'accessibility_snapshot':
        break
    }
  } catch (error: any) {
    ok = false
    note = String(error?.message || error)
    tab.loadState = tab.loadState === 'loading' ? 'error' : tab.loadState
  }

  if (input.waitMs) await new Promise((resolve) => setTimeout(resolve, input.waitMs))

  let accessibilitySnapshot: string | undefined
  if (input.includeSnapshot || ['navigate', 'tab_new', 'click', 'type', 'fill_form', 'accessibility_snapshot'].includes(input.action)) {
    accessibilitySnapshot = await browserAutomationAccessibilitySnapshot(tab).catch((error) => `Snapshot unavailable: ${String(error?.message || error)}`)
  }

  const screenshot = (input.captureScreenshot || input.action === 'screenshot')
    ? await captureBrowserAutomationScreenshot(tab, input).catch((error) => ({ title: `Screenshot failed: ${String(error?.message || error)}` }))
    : undefined

  const title = await tab.window.webContents.executeJavaScript('document.title || ""', true).catch(() => '')
  return {
    ok,
    action: input.action,
    sessionId,
    tabId: tab.id,
    url: tab.window.webContents.getURL() || input.url,
    title: String(title || ''),
    loadState: tab.loadState,
    accessibilitySnapshot,
    screenshot,
    consoleMessages: tab.consoleMessages.slice(-20),
    networkErrors: tab.networkErrors.slice(-20),
    note,
  }
}

async function loadBrowserAutomationUrl(tab: BrowserAutomationTab, url: string) {
  const ses = browserAutomationElectronSession()
  const destination = new URL(url)
  tab.allowedLoopbackOrigin = isLoopbackHost(destination.hostname) ? destination.origin : null
  // Give the tool a useful synchronous action error; the partition hook repeats this check
  // immediately before the request and covers every server redirect.
  await assertBrowserAutomationDestination(
    url,
    (hostname) => resolveBrowserAutomationHost(ses, hostname),
    { allowedLoopbackOrigin: tab.allowedLoopbackOrigin },
  )
  tab.loadState = 'loading'
  await tab.window.webContents.loadURL(url)
  tab.loadState = 'loaded'
}

async function runBrowserDomAction(tab: BrowserAutomationTab, input: BrowserAutomationInput): Promise<string | undefined> {
  return tab.window.webContents.executeJavaScript(`(() => {
    const input = ${JSON.stringify(input)};
    const interestingSelector = 'a,button,input,textarea,select,[role],summary,[onclick],[contenteditable="true"],h1,h2,h3,h4,h5,h6';
    const interesting = () => Array.from(document.querySelectorAll(interestingSelector)).filter((el) => {
      const style = window.getComputedStyle(el);
      return style && style.display !== 'none' && style.visibility !== 'hidden';
    });
    const target = () => {
      if (input.selector) return document.querySelector(input.selector);
      if (input.ref && /^e\\d+$/.test(input.ref)) return interesting()[Number(input.ref.slice(1)) - 1] || null;
      return null;
    };
    const setValue = (el, value) => {
      if (!el) throw new Error('Browser automation target not found.');
      el.scrollIntoView({ block: 'center', inline: 'center' });
      el.focus?.();
      if ('value' in el) {
        el.value = value;
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
      } else {
        el.textContent = value;
        el.dispatchEvent(new InputEvent('input', { bubbles: true, data: value }));
      }
    };
    if (input.action === 'click') {
      const el = target();
      if (!el) throw new Error('Browser automation click target not found.');
      el.scrollIntoView({ block: 'center', inline: 'center' });
      el.click();
      return 'Clicked target.';
    }
    if (input.action === 'type') {
      const el = target();
      if (!el) throw new Error('Browser automation type target not found.');
      const current = 'value' in el ? String(el.value || '') : String(el.textContent || '');
      setValue(el, current + String(input.text || ''));
      return 'Typed text into target.';
    }
    if (input.action === 'fill_form') {
      for (const field of input.fields || []) {
        const el = field.selector ? document.querySelector(field.selector) : (/^e\\d+$/.test(field.ref || '') ? interesting()[Number(field.ref.slice(1)) - 1] : null);
        setValue(el, String(field.value || ''));
      }
      return 'Filled form fields.';
    }
    return undefined;
  })()`, true)
}

async function browserAutomationAccessibilitySnapshot(tab: BrowserAutomationTab): Promise<string> {
  const snapshot = await tab.window.webContents.executeJavaScript(`(() => {
    const selector = 'a,button,input,textarea,select,[role],summary,[onclick],[contenteditable="true"],h1,h2,h3,h4,h5,h6';
    const roleFor = (el) => {
      const explicit = el.getAttribute('role');
      if (explicit) return explicit;
      const tag = el.tagName.toLowerCase();
      if (/^h[1-6]$/.test(tag)) return 'heading';
      if (tag === 'a') return 'link';
      if (tag === 'input') return el.type === 'checkbox' ? 'checkbox' : 'textbox';
      if (tag === 'textarea') return 'textbox';
      return tag;
    };
    const nameFor = (el) => (
      el.getAttribute('aria-label')
      || el.getAttribute('placeholder')
      || el.getAttribute('title')
      || el.innerText
      || el.textContent
      || el.getAttribute('name')
      || el.getAttribute('id')
      || ''
    ).replace(/\\s+/g, ' ').trim().slice(0, 120);
    const visible = (el) => {
      const style = window.getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      return style && style.display !== 'none' && style.visibility !== 'hidden' && rect.width >= 0 && rect.height >= 0;
    };
    return Array.from(document.querySelectorAll(selector)).filter(visible).slice(0, 160).map((el, index) => {
      const role = roleFor(el);
      const name = nameFor(el) || role;
      const level = role === 'heading' ? ' [level=' + el.tagName.slice(1) + ']' : '';
      return '- ' + role + ' "' + name.replace(/"/g, '\\\\"') + '"' + level + ' [ref=e' + (index + 1) + ']';
    }).join('\\n') || '(no accessible elements found)';
  })()`, true)
  return String(snapshot || '(no accessible elements found)')
}

async function captureBrowserAutomationScreenshot(tab: BrowserAutomationTab, input: BrowserAutomationInput) {
  const image = await tab.window.webContents.capturePage()
  const dir = vontraPath('browser-automation')
  await fs.mkdir(dir, { recursive: true })
  const base = safeSegment(input.screenshotName || `${tab.id}-${Date.now()}`)
  const file = path.join(dir, base.endsWith('.png') ? base : `${base}.png`)
  // Write to a sibling temp file then rename, so a crash mid-write never leaves a partial PNG
  // at the final screenshot path that a later reader would mistake for a complete image.
  const temp = `${file}.tmp-${process.pid}-${randomUUID()}`
  try {
    await fs.writeFile(temp, image.toPNG())
    await fs.rename(temp, file)
  } catch (e) {
    await fs.rm(temp, { force: true }).catch(() => {})
    throw e
  }
  return { path: file, title: input.screenshotName || tab.window.webContents.getTitle(), capturedAt: Date.now() }
}

export function closeBrowserAutomationSessions() {
  for (const automation of browserAutomationSessions.values()) {
    for (const tab of automation.tabs.values()) {
      if (!tab.window.isDestroyed()) tab.window.close()
    }
    automation.tabs.clear()
    automation.activeTabId = null
  }
  browserAutomationSessions.clear()
}

/** Close every hidden automation window and erase the persistent Chromium partition. */
export async function purgeBrowserAutomationSessions(): Promise<void> {
  closeBrowserAutomationSessions()
  const ses = session.fromPartition(BROWSER_AUTOMATION_PARTITION)
  await clearPersistentBrowserData(ses)
}
