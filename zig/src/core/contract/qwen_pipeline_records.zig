//! Compiled resources are pending; runtime admission also refuses any regression from the original kernel.
const registry = @import("registry.zig");
pub const records = [_]registry.CudaRecord{.{
    .launch = .{ .name = "Qwen prompt pipeline", .symbol = "tf_qwen_prompt_pipeline", .image = "prefill_attention", .threads = 256, .dynamic = 98304, .opt_in = true, .min_sm = 121 },
    .usage = registry.none,
}};
