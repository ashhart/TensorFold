import { isManagedComputeProvider } from '../../shared/provider-trust'
import { fetchManagedCompute } from './managed-compute-transport'
import type { Provider } from './types'
import {
  resolveConnectorRequest,
  resolveLocalProviderRequest,
  resolveLocalServiceRequest,
  resolveProviderRequest,
  resolveRemoteOrchestratorRequest,
  type ConnectorEndpointOptions,
} from './security'
import { createPinnedFetch } from './pinned-fetch'

function requestUrl(input: Parameters<typeof fetch>[0]): string {
  if (typeof input === 'string') return input
  if (input instanceof URL) return input.toString()
  return input.url
}

export function createProviderFetch(provider: Provider, fetchImpl: typeof fetch = fetch, assertRequestAllowed?: () => void): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    if (isManagedComputeProvider(provider)) {
      if (typeof input !== 'string' && !(input instanceof URL)) throw new Error('Managed compute requests require the explicit app-owned URL and request body.')
      return fetchManagedCompute(provider, requestUrl(input), init ?? {}, assertRequestAllowed)
    }
    const target = await resolveProviderRequest(provider, requestUrl(input))
    assertRequestAllowed?.()
    return createPinnedFetch(target, fetchImpl)(input, init)
  }) as typeof fetch
}

export function createConnectorFetch(
  baseUrl: string,
  label: string,
  options: ConnectorEndpointOptions = {},
  fetchImpl: typeof fetch = fetch,
): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const target = await resolveConnectorRequest(baseUrl, label, options, requestUrl(input))
    return createPinnedFetch(target, fetchImpl)(input, init)
  }) as typeof fetch
}

export function createLocalServiceFetch(baseUrl: string, label: string, fetchImpl: typeof fetch = fetch): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const target = await resolveLocalServiceRequest(baseUrl, label, requestUrl(input))
    return createPinnedFetch(target, fetchImpl)(input, init)
  }) as typeof fetch
}

export function createLocalProviderFetch(baseUrl: string, fetchImpl: typeof fetch = fetch, assertRequestAllowed?: () => void): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const target = await resolveLocalProviderRequest(baseUrl, requestUrl(input))
    assertRequestAllowed?.()
    return createPinnedFetch(target, fetchImpl)(input, init)
  }) as typeof fetch
}

export function createRemoteOrchestratorFetch(baseUrl: string, fetchImpl: typeof fetch = fetch): typeof fetch {
  return (async (input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const { target } = await resolveRemoteOrchestratorRequest(baseUrl, requestUrl(input))
    return createPinnedFetch(target, fetchImpl)(input, init)
  }) as typeof fetch
}
