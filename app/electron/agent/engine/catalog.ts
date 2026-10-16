import type { SuggestedModel } from '../types'

// What 1.0.10 says about its own models (README "Model families" and cli/hub.zig titles). The engine's
// `capabilities --json` stays the authority on which model types it serves.

export const FAMILY_TITLES: Record<string, string> = {
  nemotron_h: 'Nemotron 3.5 Lightning',
  qwen4_exp: 'Qwen3.8 Flash Next',
  glm5_next: 'GLM-5.3-Flash',
  qwen3_5: 'Qwen3.5-2B or Qwen3.8-27B',
  glm_moe_dsa: 'GLM-5.3 (EXLR8)',
}

// Checkpoints that serve with a separate drafter, passed as `serve --drafter` and pulled alongside.
export const COMPANION_DRAFTERS: Record<string, string> = {
  'TensorFold/Qwen3.8-27B-MLX-4bit': 'z-lab/Qwen3.8-27B-DFlash2',
}

export function drafterFor(repo: string): string | null {
  return COMPANION_DRAFTERS[repo] ?? null
}

// The Metal checkpoints 1.0.10 names and qualifies on a single Mac. Each is offered only after the
// device's memory and disk checks pass, so the largest ones appear only on Macs that can hold them.
export const SUGGESTED_MODELS: SuggestedModel[] = [
  {
    repo: 'mlx-community/Qwen3.5-2B-MLX-4bit',
    revision: '93760be4f1f69842a46bc13dbdc0f19e291392a3',
    title: 'Qwen3.5-2B',
    note: 'The pinned 4-bit checkpoint. Small enough for most Macs.',
  },
  {
    repo: 'TensorFold/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit',
    revision: null,
    title: 'Nemotron 3.5 Lightning 30B-A3B',
    note: 'Qualified on Metal from M1 to M5.',
  },
  {
    repo: 'TensorFold/Qwen3.8-27B-MLX-4bit',
    revision: null,
    title: 'Qwen3.8-27B',
    note: 'Downloads with its DFlash2 drafter. Qualified on M5 Max and M3 Ultra.',
  },
  {
    repo: 'TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP',
    revision: null,
    title: 'Qwen3.8 Flash Next',
    note: 'The 6-bit checkpoint, about 147 GiB, with built-in multi-token drafting. Needs a Mac with 256 GB of memory; qualified on M5 Ultra.',
  },
]
