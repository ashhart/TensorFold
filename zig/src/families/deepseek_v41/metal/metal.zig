//! DeepSeek V4.1 on Metal, the backend package: the decode forward (1-16 rows).
const std = @import("std");

pub const host = @import("dsv41");
pub const pipelines = @import("pipelines.zig");
pub const glue = @import("glue.zig");
pub const mv = @import("mv.zig");
pub const weights = @import("weights.zig");
pub const mapped = @import("mapped.zig");
pub const model_weights = @import("model_weights.zig");
pub const kv = @import("kv.zig");
pub const rope = @import("rope.zig");
pub const scratch = @import("scratch.zig");
pub const ctx = @import("ctx.zig");
pub const layer = @import("layer.zig");
pub const attn = @import("attn.zig");
pub const indexer = @import("indexer.zig");
pub const moe = @import("moe.zig");
pub const forward = @import("forward.zig");
pub const engram = pipelines.engram;

test {
    std.testing.refAllDecls(@This());
    _ = ctx;
    _ = forward;
}
