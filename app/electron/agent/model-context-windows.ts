// Task #41: model-aware context window registry.
//
// The auto-compaction trigger used to be a single user-configured number that
// ignored the model. A 256k user setting wasted a Sonnet's real 200k window;
// a tiny local model would happily blow past its 4k window. This module maps a
// model id to its *real* context window so the compaction threshold can scale
// per model instead of using one static number for everything.
//
// Order matters: families are matched most-specific first. Unknown models fall
// back to parsing an "Nk" hint in the name, and finally return null so callers
// keep their existing (user-configured) behaviour.

interface ModelWindowRule {
  match: RegExp
  window: number
}

// Most-specific patterns first. Each window is the published/nominal context
// size in tokens. Keep this list conservative — when in doubt, omit and let the
// name-inference fallback handle it.
const MODEL_WINDOWS: ModelWindowRule[] = [
  // --- Anthropic Claude (all current families ship a 200k window) ---
  { match: /claude.*(sonnet|opus|haiku)/i, window: 200_000 },
  { match: /claude-3/i, window: 200_000 },
  { match: /\bclaude\b/i, window: 200_000 },

  // --- Google Gemini (1.5/2.x expose ~1M; flagship default) ---
  { match: /gemini.*1\.5.*(pro|flash)/i, window: 1_000_000 },
  { match: /gemini.*(2\.0|2\.5|exp)/i, window: 1_000_000 },
  { match: /\bgemini\b/i, window: 1_000_000 },

  // --- OpenAI GPT-5 family ---
  { match: /gpt-5\.5|gpt-5/i, window: 205_000 },

  // --- OpenAI GPT-4 family (4o / 4-turbo / o1 all 128k) ---
  { match: /gpt-4o|gpt-4-turbo|gpt-4\.1|gpt-4-1106|gpt-4-0125/i, window: 128_000 },
  { match: /\bo1\b|\bo3\b|\bo4\b/i, window: 128_000 },
  { match: /gpt-4-32k/i, window: 32_768 },
  { match: /\bgpt-4\b/i, window: 8_192 },
  { match: /gpt-3\.5-turbo-16k/i, window: 16_384 },
  { match: /gpt-3\.5/i, window: 16_385 },

  // --- Local / open families ---
  // Llama 3.x carries a long 128k window; legacy Llama 2 is 4k.
  { match: /llama[-_. :]?3/i, window: 128_000 },
  { match: /llama[-_. :]?2/i, window: 4_096 },
  { match: /qwen[-_. :]?2\.5|qwen2\.5/i, window: 32_768 },
  { match: /deepseek[-_. :]?(r1|coder|v[0-9])/i, window: 32_768 },
  { match: /mistral[-_. :]?(nemo|large|small)/i, window: 128_000 },
  { match: /\bmixtral\b/i, window: 32_768 },
  { match: /gemma/i, window: 8_192 },
  { match: /phi[-_. :]?4|phi4/i, window: 16_000 },
  { match: /phi[-_. :]?3|phi3/i, window: 128_000 },
]

/**
 * Look up the real context window (in tokens) for a model id.
 * Returns null when the model is unknown and has no parseable "Nk" hint, so the
 * caller can keep its existing configured limit instead of guessing.
 */
export function getModelContextWindow(modelId: string | null | undefined): number | null {
  const id = (modelId ?? '').trim()
  if (!id) return null
  for (const rule of MODEL_WINDOWS) {
    if (rule.match.test(id)) return rule.window
  }
  return inferContextFromModelName(id)
}

/**
 * Recommended compaction trigger for a given context window: 85% of the window,
 * leaving headroom for the response, clamped to a safe [4k, 512k] band.
 * 85% is the industry-standard safety margin; the clamp keeps tiny windows from
 * compacting on the first message and keeps huge windows from buffering forever.
 */
export function recommendedCompactionThreshold(window: number): number {
  const FLOOR = 4_000
  const CEILING = 512_000
  if (!Number.isFinite(window) || window <= 0) return FLOOR
  const target = Math.floor(window * 0.85)
  return Math.min(CEILING, Math.max(FLOOR, target))
}

/**
 * Recommended number of recent messages to preserve verbatim through a
 * compaction, scaled to the model's context window. A bigger window can afford
 * to keep more recent turns intact (so the model doesn't lose the immediate
 * working context right after a compaction), while a tiny window must keep the
 * tail short or the compaction can't make progress.
 *
 * Scales roughly as one extra kept message per 16k of window above a 6-message
 * baseline, clamped to [4, 24]. The 6-message baseline preserves the prior
 * flat `KEEP_RECENT = 6` behaviour for a ~32k model; a 200k Claude/GPT keeps 16,
 * a 1M Gemini keeps the ceiling. Unknown/null window falls back to the
 * baseline (6) so behaviour matches the pre-adaptive default.
 */
export function recommendedKeepRecent(window: number | null): number {
  const BASELINE = 6
  const FLOOR = 4
  const CEILING = 24
  if (!window || !Number.isFinite(window) || window <= 0) return BASELINE
  const scaled = BASELINE + Math.round((window - 32_000) / 16_000)
  return Math.min(CEILING, Math.max(FLOOR, scaled))
}

/** Parse an "Nk" hint embedded in a model name (e.g. "foo-32k" -> 32000). */
function inferContextFromModelName(model: string): number | null {
  const match = model.toLowerCase().match(/(?:^|[-_. :])(\d{1,4})k(?:[-_. :]|$)/)
  if (!match) return null
  const size = Number(match[1])
  if (!Number.isFinite(size) || size < 4) return null
  return size * 1_000
}
