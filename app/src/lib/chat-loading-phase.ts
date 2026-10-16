import { isBuiltInProvider, isExternalProvider } from '../../shared/provider-trust'
import type { EngineStatus, Provider } from './types'

/** Presentation only: never infer another provider's state from the built-in engine. */
export function builtinChatLoadingPhase(input: { status?: string; provider?: Provider | null; model?: string | null; engine?: EngineStatus | null; remoteExecution?: boolean }): 'Loading model' | undefined {
  return input.status === 'thinking' && !input.remoteExecution && isBuiltInProvider(input.provider) && !isExternalProvider(input.provider)
    && input.engine?.available === true && input.engine.state === 'starting' && !!input.model && input.engine.model === input.model
    ? 'Loading model' : undefined
}
