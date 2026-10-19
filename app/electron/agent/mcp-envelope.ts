export interface McpEnvelope {
  jsonrpc: '2.0'
  id?: string | number | null
  method?: string
  params?: Record<string, any>
  result?: any
  error?: { code: number; message: string; data?: unknown }
}

export function isRecord(value: unknown): value is Record<string, any> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

function isId(value: unknown): value is string | number {
  return typeof value === 'string' || typeof value === 'number' && Number.isFinite(value)
}

export function isMcpEnvelope(value: unknown): value is McpEnvelope {
  if (!isRecord(value) || value.jsonrpc !== '2.0') return false
  if (Object.hasOwn(value, 'method')) {
    return typeof value.method === 'string' && value.method.length > 0
      && (!Object.hasOwn(value, 'id') || isId(value.id))
      && (!Object.hasOwn(value, 'params') || isRecord(value.params))
      && !Object.hasOwn(value, 'result') && !Object.hasOwn(value, 'error')
  }
  const hasResult = Object.hasOwn(value, 'result'), hasError = Object.hasOwn(value, 'error')
  if (hasResult === hasError || !isId(value.id) && !(hasError && value.id === null)) return false
  if (Object.hasOwn(value, 'params')) return false
  return hasResult || isRecord(value.error) && Number.isInteger(value.error.code) && typeof value.error.message === 'string'
}
