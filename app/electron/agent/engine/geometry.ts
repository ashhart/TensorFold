// The engine's geometry admission, mirrored for the library scan: a checkpoint whose sizes the
// engine refuses is not offered as ready, so the picker never offers a model the first message
// would reject. The engine's own rules are in zig/src/families/qwen3_5/config.zig:104 and
// zig/src/families/lfm2_moe/config.zig:111.
//
// The mirror refuses a proven mismatch only: a dimension the engine pins, read from the config,
// that differs from the pinned value. A config that does not carry the dimension is no opinion,
// so a checkpoint the engine accepts is never refused here.

export interface GeometryAdmission {
  admitted: boolean
  reason: string | null
}

interface PinnedGeometry {
  title: string
  engineError: string
  section: string
  dims: Record<string, number>
}

const QWEN35: PinnedGeometry = {
  title: 'Qwen3.5',
  engineError: 'UnsupportedQwenGeometry',
  section: 'text_config',
  dims: {
    hidden_size: 2048,
    vocab_size: 248320,
    num_hidden_layers: 24,
    intermediate_size: 6144,
    linear_num_key_heads: 16,
    linear_num_value_heads: 16,
    linear_key_head_dim: 128,
    linear_value_head_dim: 128,
    linear_conv_kernel_dim: 4,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    head_dim: 256,
    full_attention_interval: 4,
  },
}

const QWEN27: PinnedGeometry = {
  title: 'Qwen3.5 27B',
  engineError: 'UnqualifiedQwenShape',
  section: 'text_config',
  dims: {
    hidden_size: 5120,
    vocab_size: 248320,
    num_hidden_layers: 64,
    intermediate_size: 17408,
    linear_num_key_heads: 16,
    linear_num_value_heads: 48,
    linear_key_head_dim: 128,
    linear_value_head_dim: 128,
    linear_conv_kernel_dim: 4,
    num_attention_heads: 24,
    num_key_value_heads: 4,
    head_dim: 256,
  },
}

const LFM2_MOE: PinnedGeometry = {
  title: 'LFM2.5',
  engineError: 'UnsupportedLfmGeometry',
  section: '',
  dims: {
    hidden_size: 2048,
    vocab_size: 128000,
    num_hidden_layers: 40,
    intermediate_size: 11776,
    moe_intermediate_size: 1536,
    num_experts: 64,
    num_experts_per_tok: 4,
    num_dense_layers: 2,
    conv_L_cache: 3,
    num_attention_heads: 32,
    num_key_value_heads: 8,
  },
}

const PINNED: Record<string, PinnedGeometry> = {
  qwen3_5: QWEN35,
  lfm2_moe: LFM2_MOE,
}

function plain(value: unknown): value is Record<string, unknown> {
  return !!value && typeof value === 'object' && !Array.isArray(value)
}

// The engine's layer types: Qwen3.5 alternates a linear attention layer with a full one every fourth layer.
function layerTypesMatch(kind: string, layers: unknown): boolean {
  if (!Array.isArray(layers) || layers.length === 0) return true
  if (kind !== 'qwen3_5') return true
  return layers.every((value, index) => value === (index % 4 !== 3 ? 'linear_attention' : 'full_attention'))
}

export function admitGeometry(modelType: string, config: unknown): GeometryAdmission {
  const defaultGeometry = PINNED[modelType]
  if (!defaultGeometry) return { admitted: true, reason: null }
  if (!plain(config)) return { admitted: true, reason: null }
  const section = defaultGeometry.section ? config[defaultGeometry.section] : config
  if (!plain(section)) return { admitted: true, reason: null }
  // The native dispatcher sends a tied head or the 2B hidden size to the 2B engine.
  const useQwen27 = modelType === 'qwen3_5' && section.hidden_size === QWEN27.dims.hidden_size &&
    config.hidden_size !== QWEN35.dims.hidden_size && config.tie_word_embeddings !== true && section.tie_word_embeddings !== true
  const pinned = useQwen27 ? QWEN27 : defaultGeometry
  for (const [key, expected] of Object.entries(pinned.dims)) {
    const value = section[key]
    if (value === undefined || value === null) continue
    if (value !== expected) {
      return {
        admitted: false,
        reason: `This checkpoint's ${key.replace(/_/g, ' ')} is ${value}, and the bundled engine serves one ${pinned.title} geometry; the engine refuses it (${pinned.engineError}).`,
      }
    }
  }
  const interval = section.full_attention_interval
  const wrongQwen27Interval = useQwen27 && typeof interval === 'number' && interval > 0 && interval !== 4
  if (wrongQwen27Interval || !layerTypesMatch(modelType, section.layer_types)) {
    return {
      admitted: false,
      reason: `This checkpoint's layer types are not the ones the bundled engine serves for ${pinned.title}; the engine refuses it (${pinned.engineError}).`,
    }
  }
  return { admitted: true, reason: null }
}
