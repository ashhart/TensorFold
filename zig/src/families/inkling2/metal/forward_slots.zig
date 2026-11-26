const mtl = @import("metal");
const ck = @import("../../../core/checkpoint_metal.zig");
const layer_plan = @import("../plan.zig");
const glue = @import("glue.zig");
const model_file = @import("model.zig");
const Model = model_file.Model;
const RArgs = extern struct { count: u32, boundary: u32 = 0 };
const ap = @import("attention_plan.zig");
const attention = @import("attention.zig");
const state = @import("state.zig");

fn at(buf: mtl.Buffer, off: usize) state.Ref {
    return .{ .buf = buf, .off = off };
}

fn a(x_: state.Ref) attention.Ref {
    return .{ .buf = x_.buf, .off = x_.off };
}

fn r(t: ck.Tensor) state.Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}
fn runtime(m: *const Model, e: mtl.ComputeEncoder, op: model_file.RuntimeOp, refs: []const state.Ref, count: usize, boundary: usize) void {
    e.setPipeline(m.runtime[@backingInt(op)]);
    for (refs, 0..) |r_, i| e.setBuffer(r_.buf, r_.off, i);
    e.setValue(RArgs{ .count = @intCast(count), .boundary = @intCast(boundary) }, refs.len);
    e.dispatchThreads(mtl.Size.of(count, 1, 1), mtl.Size.of(256, 1, 1));
}
fn copyBf16(m: *const Model, e: mtl.ComputeEncoder, from: state.Ref, to: state.Ref, count: usize) void {
    runtime(m, e, .copy_bf16, &.{ from, to }, count, 0);
}

/// One non-tree stream's rows of a visit's attention.
pub const Attend = struct {
    base: usize,
    rows: usize,
    rr: ap.Rows,
    layer: ap.Layer,
    meta: attention.RowMeta,
    kv: attention.Cache,
};

pub const SlotBatch = struct { base: usize, rr: ap.Rows, meta: attention.RowMeta, len: usize };

pub fn slotBatch(batch: []const Attend, s: *state.Scratch, out_slots: *[ap.max_slots]attention.Slot) SlotBatch {
    var batch_rows: usize = 0;
    var last_pos: u32 = 0;
    for (batch, 0..) |it, k| {
        batch_rows += it.rows;
        last_pos = @max(last_pos, it.rr.last_pos);
        out_slots[k] = .{ .kv = it.kv, .ring = it.layer.ring, .chunk = it.layer.chunk, .last_pos = it.rr.last_pos, .rows = @intCast(it.rows) };
    }
    const b = batch[0].base;
    return .{
        .base = b,
        .rr = .{ .rows = @intCast(batch_rows), .heads = 24, .kv_heads = 8, .last_pos = last_pos, .streams = @intCast(batch.len) },
        .meta = .{ .pos = a(at(s.pos, b * 4)), .seg_of = a(at(s.slot_of, b * 4)), .segs = a(at(s.segs, 0)) },
        .len = batch.len,
    };
}

pub fn stageConvInput(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool) void {
    const projected = if (key) at(s.k_raw, base * state.kv_width * 2) else at(s.v_raw, base * state.kv_width * 2);
    const taps = cache.staged(v.cache_slot, if (key) 0 else 1, 0);
    copyBf16(m, e, cache.history(v.cache_slot, if (key) 0 else 1), taps, 3 * state.kv_width);
    copyBf16(m, e, projected, cache.staged(v.cache_slot, if (key) 0 else 1, 3), rows * state.kv_width);
}

pub fn launchConvInput(m: *const Model, e: mtl.ComputeEncoder, s: *state.Scratch, cache: *state.Cache, v: *const layer_plan.Visit, base: usize, rows: usize, key: bool) void {
    const lp = m.weight_plan.at(v.layer);
    const w = &m.weights.layers[v.layer];
    const raw = if (key) at(s.k_conv, base * state.kv_width * 2) else at(s.v_conv, base * state.kv_width * 2);
    const normalized = if (key) at(s.k_norm, 0) else if (v.hasMessages() or lp.kind == .global) at(s.v_norm, 0) else at(s.q_norm, 0);
    const taps = cache.staged(v.cache_slot, if (key) 0 else 1, 0);
    switch (if (key) w.attn.k_conv else w.attn.v_conv) {
        .static => |weight| glue.convStatic(&m.glue, e, taps, r(weight), raw, @intCast(rows), state.kv_width),
        .dynamic => |d| glue.convDynamic(&m.glue, e, taps, at(normalized.buf, normalized.off + base * state.hidden * 2), r(d.static_w), r(d.projection), raw, @intCast(rows), state.kv_width),
    }
}
