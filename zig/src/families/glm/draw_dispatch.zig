//! The sampler's production binding path, also used by the checkpoint-free Metal gate.
const std = @import("std");
const mtl = @import("metal");
const draw = @import("draw_rule.zig");
pub const rules = draw;

pub fn encode(enc: mtl.ComputeEncoder, pipeline: mtl.Pipeline, logits: mtl.Buffer, logits_off: usize, picks: mtl.Buffer, picks_off: usize, payload: *const draw.Payload) void {
    enc.setPipeline(pipeline);
    enc.setBuffer(logits, logits_off, 0);
    enc.setBytes(std.mem.asBytes(payload), 1);
    enc.setBuffer(picks, picks_off, 2);
    enc.dispatchGroups(mtl.Size.of(payload.header.n, 1, 1), mtl.Size.of(1024, 1, 1));
}
