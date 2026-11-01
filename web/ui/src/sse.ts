/** Server-sent events, LF or CRLF, in arbitrary chunks: yields the `data:` payloads of complete events. */
export function sseEvents(buffer: string): { events: string[]; rest: string } {
  const norm = buffer.replace(/\r\n/g, '\n')
  const parts = norm.split('\n\n')
  const rest = parts.pop() ?? ''
  const events: string[] = []
  for (const part of parts) {
    const data = part.split('\n').filter((l) => l.startsWith('data:')).map((l) => l.slice(5).trim()).join('\n')
    if (data) events.push(data)
  }
  return { events, rest }
}
