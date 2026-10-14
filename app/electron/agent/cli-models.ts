import type { Provider } from './types'

export const CODEX_CLI_MODELS = ['codex-cli-default', 'gpt-5.4', 'gpt-5.5', 'gpt-5.5-codex-high', 'gpt-5.5-extra-high']
export const CLAUDE_CLI_MODELS = ['claude-code-default', 'opus', 'sonnet']
// OpenCode drives whatever models its own config exposes; 'opencode-default' lets the server
// pick, and users add `providerID/modelID` ids in Settings.
export const OPENCODE_MODELS = ['opencode-default']
// Kilo is an OpenCode fork on the same server protocol; same model-id story.
export const KILO_MODELS = ['kilo-default']
// Pi resolves models itself (--model provider/id); 'pi-default' lets pi pick its default.
export const PI_CLI_MODELS = ['pi-default']
export const CURSOR_CLI_MODELS = ['cursor-default']
export const GEMINI_CLI_MODELS = ['gemini-default']

const LEGACY_CODEX_CLI_MODELS = ['codex-cli-default']
const LEGACY_CLAUDE_CLI_MODELS = ['claude-code-default', 'ultracode', 'opus-4.8', 'sonnet', 'opus']

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

export function isLegacyCliModelList(transport: Provider['transport'] | undefined, models?: string[] | null): boolean {
  const clean = (models ?? []).map((model) => model.trim()).filter(Boolean)
  if (!clean.length) return true
  const legacy = transport === 'codex-cli'
    ? LEGACY_CODEX_CLI_MODELS
    : transport === 'claude-cli'
      ? LEGACY_CLAUDE_CLI_MODELS
      : []
  if (!legacy.length) return false
  // Migrate ONLY a saved list that EXACTLY equals a known OLD default set (order-independent) — e.g.
  // the codex sentinel ['codex-cli-default'] or the old Claude default incl. its obsolete aliases.
  // A curated subset of current models (['opus','sonnet']) or a single override (['opus']) is NOT a
  // legacy default set and must survive load — resetting those was the silent every-launch reset.
  const cleanSet = new Set(clean)
  return cleanSet.size === legacy.length && legacy.every((model) => cleanSet.has(model))
}

export function normalizeLegacyCliModelList(transport: Provider['transport'] | undefined, models?: string[] | null): string[] {
  const defaults = defaultCliModels(transport)
  const clean = Array.from(new Set((models ?? []).map((model) => model.trim()).filter(Boolean)))
  if (!defaults.length) return clean
  return isLegacyCliModelList(transport, clean) ? defaults : clean
}
