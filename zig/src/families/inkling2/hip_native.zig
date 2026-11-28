//! Inkling 2 Mini on one HIP rank, text only: the serve flags it refuses before any load.
const std = @import("std");
const api = @import("engine_api");

/// config.json's name for Inkling 2 Mini, as the CUDA and Metal families read it.
pub const model_type = "inkling2_loop";

/// The largest --prompt-cache-gib budget taken, as on Metal (native/inkling2.zig).
const most_cache_gib: f64 = 1024 * 1024;

/// Why these flags cannot open Inkling 2 Mini on HIP; the host's shared refusals follow (native/hip_families.zig).
pub fn refusal(o: api.Open) ?[]const u8 {
    if (o.tp != 1) return "inkling2_loop: this HIP engine serves one rank; serve with --tp 1";
    if (o.cfg_enabled or o.cfg_batched) return "inkling2_loop: classifier-free guidance is unsupported; serve without --cfg-gamma, --cfg-allow-request-gamma and --cfg-batched";
    if (o.loops != null) return "inkling2_loop: the model has exactly three core passes; --loops cannot override them";
    if (o.mtp_drafts != null or o.mtp_confidence != null) return "inkling2_loop: external draft controls are unsupported; serve without --mtp-drafts and --mtp-confidence";
    if (o.precision != null) return "inkling2_loop: the checkpoint's NVFP4, MXFP8 and BF16 arithmetic is fixed; serve without --precision";
    if (o.prompt_int8) return "inkling2_loop: the prompt matmuls keep the checkpoint's arithmetic; serve without --prompt-precision int8";
    if (o.learn != null or o.learn_conversations) return "inkling2_loop: learned disk states are not implemented; serve without --learn and --learn-conversations";
    // a negative budget is the host's own refusal (native/hip.zig checkGroup)
    if (o.prompt_cache_gib) |g| if (!std.math.isFinite(g) or g > most_cache_gib) return "inkling2_loop: --prompt-cache-gib must be a finite budget of at most 1048576 GiB; serve with a smaller one";
    return null;
}
