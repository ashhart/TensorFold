import type { Provider } from './types'

export const CODEX_CLI_MODELS = ['codex-cli-default', 'gpt-5.4', 'gpt-5.5', 'gpt-5.5-codex-high', 'gpt-5.5-extra-high']
export const CLAUDE_CLI_MODELS = ['claude-code-default', 'opus', 'sonnet']
// OpenCode (VST-033): 'opencode-default' lets the server pick; users add providerID/modelID.
export const OPENCODE_MODELS = ['opencode-default']
// Kilo is an OpenCode fork on the same protocol; same model-id story.
export const KILO_MODELS = ['kilo-default']
// Pi resolves models itself (--model provider/id); 'pi-default' lets pi pick its default.
export const PI_CLI_MODELS = ['pi-default']
export const CURSOR_CLI_MODELS = ['cursor-default']
export const GEMINI_CLI_MODELS = ['gemini-default']

export function defaultCliModels(transport?: Provider['transport'] | null): string[] {
  if (transport === 'codex-cli') return CODEX_CLI_MODELS
  if (transport === 'claude-cli') return CLAUDE_CLI_MODELS
  if (transport === 'opencode') return OPENCODE_MODELS
  if (transport === 'kilo') return KILO_MODELS
  if (transport === 'pi-cli') return PI_CLI_MODELS
  if (transport === 'cursor-cli') return CURSOR_CLI_MODELS
  if (transport === 'gemini-cli') return GEMINI_CLI_MODELS
  return []
}

export function cliModelCatalog(transport: Provider['transport'] | undefined, models?: string[] | null): string[] {
  const clean = (models ?? []).map((model) => model.trim()).filter(Boolean)
  const defaults = defaultCliModels(transport)
  if (!defaults.length) return Array.from(new Set(clean))
  return Array.from(new Set([...clean, ...defaults]))
}
