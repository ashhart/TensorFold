//! DeepSeek V4.1 on Metal, the backend package: the decode forward (1-16 rows), the prompt path, the served lanes.
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
pub const prompt = @import("prompt.zig");
pub const prompt_layer = @import("prompt_layer.zig");
pub const prompt_scratch = @import("prompt_scratch.zig");
pub const dspark = @import("dspark.zig");
pub const dspark_plan = @import("dspark_plan.zig");
pub const dspark_stage = @import("dspark_stage.zig");
pub const dspark_weights = @import("dspark_weights.zig");
pub const engram = pipelines.engram;
pub const snapshots = @import("snapshots.zig");
pub const target = @import("target.zig");
pub const pass = @import("pass.zig");
pub const serve = @import("serve.zig");
pub const vision = @import("vision.zig");
pub const vision_weights = @import("vision_weights.zig");
pub const vision_tower = @import("vision_tower.zig");
pub const vision_prep = @import("vision_prep.zig");
pub const vision_rows = @import("vision_rows.zig");
pub const vision_prepare = @import("vision_prepare.zig");
pub const boot = @import("boot.zig");

test {
    std.testing.refAllDecls(@This());
    _ = ctx;
    _ = forward;
    _ = prompt;
    _ = prompt_scratch;
    _ = dspark;
    _ = dspark_plan;
    _ = pass;
    _ = vision;
    _ = vision_prep;
    _ = vision_rows;
    _ = vision_prepare;
}
