const std = @import("std");
const weights = @import("families/qwen3_5/backend/vulkan/weights.zig");
const state = @import("families/qwen3_5/backend/vulkan/state.zig");

test {
    std.testing.refAllDecls(weights);
    std.testing.refAllDecls(weights.Model);
    std.testing.refAllDecls(state);
    std.testing.refAllDecls(state.State);
}
