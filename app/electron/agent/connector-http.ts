import { fetchWithRetry } from './connector-retry'
import { type PinnedHostResolver } from './pinned-fetch'
import { createConnectorFetch } from './secure-fetch'
import { DEFAULT_JSON_RESPONSE_MAX_BYTES, readBoundedJsonResponse } from './bounded-response'

export const CONNECTOR_JSON_MAX_BYTES = DEFAULT_JSON_RESPONSE_MAX_BYTES

export interface ConnectorHttpDeps {
  fetchImpl?: typeof fetch
  lookup?: PinnedHostResolver
}

/** Read a connector JSON response without allowing a hostile peer to allocate unbounded memory. */
export async function readBoundedConnectorJson(
  response: Pick<Response, 'headers' | 'body' | 'text' | 'json'>,
  label: string,
  maxBytes = CONNECTOR_JSON_MAX_BYTES,
): Promise<any> {
  return readBoundedJsonResponse(response, label, maxBytes)
}

/**
 * Fixed/configured connector request: validate the exact origin, resolve once, pin that answer
 * into the socket dispatcher, reject redirects, retry transient responses, then bound JSON.
 */
export async function fetchConnectorJson(
  baseUrl: string,
  label: string,
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1] = {},
  deps: ConnectorHttpDeps = {},
): Promise<{ response: Response; json: any }> {
  const secureFetch = createConnectorFetch(
    baseUrl,
    label,
    { lookup: deps.lookup },
    deps.fetchImpl ?? fetch,
  )
  const response = await fetchWithRetry(input, { ...init, redirect: 'error' }, { fetchImpl: secureFetch })
  const json = await readBoundedConnectorJson(response, label)
  return { response, json }
}
