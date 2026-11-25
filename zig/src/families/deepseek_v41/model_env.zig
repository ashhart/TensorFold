//! DeepSeek-V4.1 serving's TF_DSV41_CONTEXT and TF_DSV41_OWN_PREFILL readers, re-exported by cuda_model.zig.

/// TF_DSV41_CONTEXT: the slot's positions; null: block.Options' 4,096
pub fn contextEnv() !?i64 {
    return @import("sys").environment.read(?i64, "TF_DSV41_CONTEXT", null);
}

/// TF_DSV41_OWN_PREFILL=0: prompts as 16-row decode windows (every rank must agree: the plans differ)
pub fn ownPrefillEnv() !bool {
    return @import("sys").environment.read(bool, "TF_DSV41_OWN_PREFILL", true);
}
