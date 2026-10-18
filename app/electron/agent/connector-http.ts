import { fetchWithRetry } from './connector-retry'
import { type PinnedHostResolver } from './pinned-fetch'
import type { ConnectorEndpointOptions } from './security'
import type { ConnectorRetryOptions } from './connector-retry'
import { createConnectorFetch } from './secure-fetch'
import { DEFAULT_JSON_RESPONSE_MAX_BYTES, readBoundedJsonResponse } from './bounded-response'

export const CONNECTOR_JSON_MAX_BYTES = DEFAULT_JSON_RESPONSE_MAX_BYTES

export interface ConnectorHttpDeps {
  fetchImpl?: typeof fetch
  lookup?: PinnedHostResolver
  allowPrivateNetwork?: boolean
  retry?: Omit<ConnectorRetryOptions, 'fetchImpl'>
}

/** Read a connector JSON response without allowing a hostile peer to allocate unbounded memory. */
export async function readBoundedConnectorJson(
  response: Pick<Response, 'headers' | 'body' | 'text' | 'json'>,
  label: string,
  maxBytes = CONNECTOR_JSON_MAX_BYTES,
): Promise<any> {
  return readBoundedJsonResponse(response, label, maxBytes)
}

/** Origin-pinned response fetch shared by connectors that parse different response shapes. */
export async function fetchConnectorResponse(
  baseUrl: string,
  label: string,
  input: Parameters<typeof fetch>[0],
  init: Parameters<typeof fetch>[1] = {},
  deps: ConnectorHttpDeps = {},
): Promise<Response> {
  const endpoint: ConnectorEndpointOptions = { lookup: deps.lookup, allowPrivateNetwork: deps.allowPrivateNetwork === true }
  const secureFetch = createConnectorFetch(baseUrl, label, endpoint, deps.fetchImpl ?? fetch)
  return fetchWithRetry(input, { ...init, redirect: 'error' }, { ...deps.retry, fetchImpl: secureFetch })
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
  const response = await fetchConnectorResponse(baseUrl, label, input, init, deps)
  const json = await readBoundedConnectorJson(response, label)
  return { response, json }
}
