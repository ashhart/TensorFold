import type { EngineConnection, EngineStatus } from './types'

export function engineBaseUrl(status: EngineStatus | null | undefined): string | null {
  const port = status?.port
  return status?.state === 'ready' && status.model && typeof port === 'number' && Number.isInteger(port) && port > 0 && port <= 65535
    ? `http://127.0.0.1:${port}/v1` : null
}

const quote = (text: string) => `'${text.replace(/'/g, `'\\''`)}'`

// Client examples must keep the launch key on the supervised loopback endpoint.
export function engineCurl(connection: EngineConnection): string {
  const url = new URL(connection.baseUrl)
  if (url.protocol !== 'http:' || url.hostname !== '127.0.0.1' || !url.port || url.pathname !== '/v1' || url.search || url.hash || url.username || url.password) {
    throw new Error('Expected the local TensorFold engine connection.')
  }
  return [
    `curl ${quote(`${url.href}/chat/completions`)} \\`,
    `  -H ${quote(`Authorization: Bearer ${connection.apiKey}`)} \\`,
    `  -H 'Content-Type: application/json' \\`,
    `  -d ${quote(JSON.stringify({ model: connection.model, messages: [{ role: 'user', content: 'Hello from my local model.' }], stream: true }))}`,
  ].join('\n')
}

export function formatTokenRate(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value.toLocaleString('en-US', { maximumFractionDigits: 1 }) : '—'
}
