import { BrowserWindow, screen } from 'electron'
import type { ComputerIndicator } from '../../agent/computer-use/runner'

const STOP_URL = 'tensorfold-stop:now'
const WIDTH = 380
const HEIGHT = 40

const escapeHtml = (value: string) => value.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!)

function page(label: string): string {
  return `<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'"><style>
html,body{margin:0;height:100%;background:transparent;font:500 13px -apple-system,BlinkMacSystemFont,sans-serif;-webkit-user-select:none}
.bar{box-sizing:border-box;height:100%;display:flex;align-items:center;gap:10px;padding:0 6px 0 14px;border-radius:20px;background:rgba(17,18,22,.92);color:#ecedf1;border:1px solid rgba(90,120,255,.55)}
.dot{width:8px;height:8px;border-radius:50%;background:linear-gradient(100deg,#ff7a5c,#8b4dff,#3d6bff)}
.label{flex:1;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
a{color:#fff;text-decoration:none;background:#d42f3c;border-radius:14px;padding:5px 12px;font-weight:600}
</style></head><body><div class="bar"><span class="dot"></span><span class="label">TensorFold is operating ${escapeHtml(label)}</span><a href="${STOP_URL}">Stop</a></div></body></html>`
}

// A small always-on-top bar while TensorFold operates another app; it runs no script, and its Stop link is caught here.
export function createComputerIndicator(): ComputerIndicator {
  let win: BrowserWindow | null = null
  let onStop: (() => void) | null = null
  let shownLabel: string | null = null

  function create(): BrowserWindow {
    const next = new BrowserWindow({
      width: WIDTH, height: HEIGHT, show: false, frame: false, transparent: true, resizable: false, movable: false,
      minimizable: false, maximizable: false, fullscreenable: false, focusable: false, skipTaskbar: true, hasShadow: false,
      webPreferences: { sandbox: true, contextIsolation: true, nodeIntegration: false, javascript: false, spellcheck: false, partition: 'tensorfold-indicator' },
    })
    next.setAlwaysOnTop(true, 'screen-saver')
    next.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
    next.webContents.setWindowOpenHandler(() => ({ action: 'deny' }))
    next.webContents.on('will-navigate', (event, url) => {
      event.preventDefault()
      if (url === STOP_URL) onStop?.()
    })
    next.on('closed', () => { if (win === next) { win = null; shownLabel = null } })
    return next
  }

  return {
    show(label, stop) {
      onStop = stop
      if (!win || win.isDestroyed()) win = create()
      const area = screen.getPrimaryDisplay().workArea
      win.setBounds({ x: Math.round(area.x + (area.width - WIDTH) / 2), y: area.y + 8, width: WIDTH, height: HEIGHT })
      if (shownLabel !== label) { shownLabel = label; void win.loadURL(`data:text/html;charset=utf-8,${encodeURIComponent(page(label))}`) }
      if (!win.isVisible()) win.showInactive()
    },
    hide() {
      onStop = null
      if (win && !win.isDestroyed()) win.hide()
    },
  }
}
