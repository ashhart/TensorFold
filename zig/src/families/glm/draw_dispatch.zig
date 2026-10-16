//! The sampler's production binding path (shared tf_sample_full, rules through gpu_rule).
const std = @import("std");
const mtl = @import("metal");
const draw = @import("draw_rule.zig");
pub const rules = draw;

/// The shared kernel's per-row binding arrays (its buffer ABI), one row a rule, in row order.
pub const Bindings = struct {
    seeds: [2 * draw.max_rows]u32 = @splat(0),
    positions: [draw.max_rows]u32 = @splat(0),
    cfgs: [4 * draw.max_rows]f32 = @splat(0),
    caps: [draw.max_rows]u32 = @splat(0),

    /// Pack the already canonical gpu_rule.rule values without recomputing temperature or min-p.
    pub fn init(payload: *const draw.Payload) Bindings {
        var b: Bindings = .{};
        for (payload.rules[0..payload.header.n], 0..) |rule, r| {
            b.seeds[2 * r ..][0..2].* = .{ rule.seed_lo, rule.seed_hi };
            b.positions[r] = rule.position;
            b.cfgs[4 * r ..][0..4].* = .{ rule.inv_t, rule.top_p, rule.near, rule.min_log };
            b.caps[r] = rule.top_k;
        }
        return b;
    }
};

/// The shared kernel's function name for a payload: tf_sample_full, `_ids` when rows map ids.
pub fn kernelName(mapped: bool) [:0]const u8 {
    return if (mapped) "tf_sample_full_ids" else "tf_sample_full";
}

/// The shared sampler over a run of sampled rows: one tf_sample_full dispatch, one threadgroup a row.
fn encodeSampled(enc: mtl.ComputeEncoder, pipeline: mtl.Pipeline, logits: mtl.Buffer, logits_off: usize, picks: mtl.Buffer, picks_off: usize, payload: *const draw.Payload) void {
    const bindings = Bindings.init(payload);
    const rows = payload.header.n;
    enc.setPipeline(pipeline);
    enc.setBuffer(logits, logits_off, 0);
    enc.setBytes(std.mem.sliceAsBytes(bindings.seeds[0 .. 2 * rows]), 1);
    enc.setBytes(std.mem.sliceAsBytes(bindings.positions[0..rows]), 2);
    enc.setBytes(std.mem.sliceAsBytes(bindings.cfgs[0 .. 4 * rows]), 3);
    enc.setBytes(std.mem.sliceAsBytes(bindings.caps[0..rows]), 4);
    enc.setBuffer(picks, picks_off, 5);
    enc.setBytes(std.mem.asBytes(&payload.rules[0].vocab), 7); // the shared kernel reads V at 7 even unmapped
    enc.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(1024, 1, 1));
}

/// One glm_argmax dispatch over a greedy run (out[row] indexes the picks from `picks_off`).
fn encodeGreedy(enc: mtl.ComputeEncoder, pipeline: mtl.Pipeline, logits: mtl.Buffer, logits_off: usize, picks: mtl.Buffer, picks_off: usize, vocab: u32, run: usize) void {
    enc.setPipeline(pipeline);
    enc.setBuffer(logits, logits_off, 0);
    enc.setBuffer(picks, picks_off, 1);
    enc.setBytes(std.mem.asBytes(&vocab), 2);
    enc.dispatchGroups(mtl.Size.of(run, 1, 1), mtl.Size.of(1024, 1, 1));
}

/// One head draw: the rows split into maximal runs of one mode (sampled tf_sample_full, else argmax).
pub fn encode(enc: mtl.ComputeEncoder, sample_pipeline: mtl.Pipeline, argmax_pipeline: mtl.Pipeline, logits: mtl.Buffer, logits_off: usize, picks: mtl.Buffer, picks_off: usize, payload: *const draw.Payload, pitch_words: usize) void {
    var r: usize = 0;
    while (r < payload.header.n) {
        const sampled_run = !payload.rules[r].isGreedy();
        var run: usize = 1;
        while (r + run < payload.header.n and (!payload.rules[r + run].isGreedy()) == sampled_run) run += 1;
        const row_off = r * pitch_words * 2; // a row's bf16 pitch in bytes
        const run_payload = payload.run(r, run);
        if (sampled_run) {
            encodeSampled(enc, sample_pipeline, logits, logits_off + row_off, picks, picks_off + r * 4, &run_payload);
        } else {
            encodeGreedy(enc, argmax_pipeline, logits, logits_off + row_off, picks, picks_off + r * 4, @intCast(pitch_words), run);
        }
        r += run;
    }
}
