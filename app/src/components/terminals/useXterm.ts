import { useEffect, useRef } from 'react'
import { createTerminalSanitizer } from '../../lib/terminal-sanitize'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import '@xterm/xterm/css/xterm.css'
import { api } from '../../lib/ipc'
import { useStore } from '../../lib/store'
import { terminalSessionKey, terminalSessionRoute } from '../../lib/terminal-sessions'

// Mounts an xterm.js terminal into a container and wires it to the session's IPC stream:
// PTY data -> term.write; keystrokes -> writeTerminal; container resize -> resizeTerminal.
export function useXterm(sessionKey: string) {
  const containerRef = useRef<HTMLDivElement>(null)
  const session = useStore((s) => {
    for (const sessions of Object.values(s.terminals)) {
      const found = sessions.find((candidate) => terminalSessionKey(candidate) === sessionKey)
      if (found) return found
    }
    return null
  })
  const routeKey = (() => {
    const route = terminalSessionRoute(session)
    return route.kind === 'orchestrator' ? `orchestrator:${route.hostId}` : route.kind
  })()
  const sessionId = session?.id ?? ''
  const remoteHostId = routeKey.startsWith('orchestrator:') ? routeKey.slice('orchestrator:'.length) : null
  const localSession = routeKey === 'local'
  const controllable = localSession || !!remoteHostId

  useEffect(() => {
    const el = containerRef.current
    if (!el) return
    // On the glass theme, let the terminal read through to the window vibrancy: allowTransparency +
    // a translucent background (still dark enough that text stays legible). Solid otherwise.
    const glass = document.documentElement.classList.contains('translucent')
    const term = new Terminal({
      fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
      fontSize: 12, cursorBlink: true,
      allowTransparency: glass,
      theme: { background: glass ? 'rgba(11,11,13,0.6)' : '#0b0b0d', foreground: '#e6e6e6' },
    })
    const fit = new FitAddon()
    term.loadAddon(fit)
    term.open(el)
    fit.fit()
    // Focus the terminal so keystrokes (arrows, Enter, everything) go to the PTY and
    // not to the launch button that was just clicked — that focus bug made Enter
    // re-trigger the button and spawn another CLI tab.
    term.focus()
    const focusSoon = setTimeout(() => term.focus(), 0)
    // Clicking anywhere in the pane (incl. the padding around xterm) re-focuses it.
    const onPointerDown = () => term.focus()
    el.addEventListener('pointerdown', onPointerDown)

    // Restore scrollback on (re)mount: subscribe to live data first and QUEUE it, then replay the
    // PTY's buffered backlog before flushing the queue — so a re-opened pane shows its history
    // instead of a blank screen (the PTY outlives the xterm). Subscribe-first avoids dropping any
    // output that arrives while the backlog is in flight; at worst a tiny overlap repeats a line.
    let restored = false
    let disposed = false
    const sanitizeLive = createTerminalSanitizer()
    const sanitizeBacklog = createTerminalSanitizer()
    const queued: string[] = []
    const offData = api.onTerminalData(({ sessionId: sid, chunk, orchestratorHostId = null }) => {
      if (sid !== sessionId) return
      if (orchestratorHostId ? orchestratorHostId !== remoteHostId : !localSession) return
      const safe = sanitizeLive(chunk)
      if (restored) term.write(safe)
      else queued.push(safe)
    })
    const flushQueue = () => { for (const c of queued) term.write(c); queued.length = 0; restored = true }
    const backlogPromise = localSession ? api.terminalBacklog(sessionId) : Promise.resolve('')
    void backlogPromise.then((backlog) => {
      if (disposed) return
      if (backlog) term.write(sanitizeBacklog(backlog))
      flushQueue()
    }).catch(() => { if (!disposed) flushQueue() })
    const offInput = term.onData((data) => {
      if (!controllable) return
      void (remoteHostId ? api.remoteTerminalWrite(remoteHostId, sessionId, data) : api.writeTerminal(sessionId, data))
    })
    const pushSize = () => {
      try { fit.fit() } catch { /* not attached yet */ }
      if (!controllable) return
      void (remoteHostId ? api.remoteTerminalResize(remoteHostId, sessionId, term.cols, term.rows) : api.resizeTerminal(sessionId, term.cols, term.rows))
    }
    const ro = new ResizeObserver(pushSize)
    ro.observe(el)
    pushSize()
    // Re-fit after the browser lays out and the renderer measures the real cell height. The first
    // synchronous fit can over-propose rows (it uses a not-yet-measured cell height), which renders
    // the terminal a few px taller than the pane and clips the bottom row — e.g. pi's status line.
    const raf = requestAnimationFrame(pushSize)
    const t1 = setTimeout(pushSize, 80)
    const t2 = setTimeout(pushSize, 300)

    return () => { disposed = true; cancelAnimationFrame(raf); clearTimeout(t1); clearTimeout(t2); clearTimeout(focusSoon); el.removeEventListener('pointerdown', onPointerDown); offData(); offInput.dispose(); ro.disconnect(); term.dispose() }
  }, [controllable, localSession, remoteHostId, sessionId])

  return containerRef
}
