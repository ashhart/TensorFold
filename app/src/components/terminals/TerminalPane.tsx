import { useXterm } from './useXterm'

// One live terminal pane bound to a session id. Fills its parent.
export function TerminalPane({ sessionKey }: { sessionKey: string }) {
  const ref = useXterm(sessionKey)
  // terminal-surface: solid by default, transparent on the glass theme (the xterm paints its own
  // translucent bg). A touch more bottom padding so a sub-pixel over-fit can't clip the last row.
  return <div ref={ref} className="terminal-surface h-full w-full overflow-hidden rounded-md px-1.5 pt-1.5 pb-2.5" />
}
