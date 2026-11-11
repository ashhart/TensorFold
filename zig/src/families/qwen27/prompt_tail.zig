//! A prompt pass that keeps no row's logits: its last layer writes keys and values only, when nothing else reads it.
const std = @import("std");
const mtl = @import("metal");
const forward = @import("forward.zig");
const wts = @import("gpu_weights.zig");
const attn = @import("attention.zig");
const Frame = @import("gpu_frame.zig").Frame;
const Runner = @import("decode_round.zig").Runner;

/// What reads the last layer's rows besides the head, and whether the pass keeps a row at all.
pub const Facts = struct {
    /// The pass keeps no row's logits (a prompt chunk before the last).
    keeps_none: bool,
    /// A drafter's or MTP head's tap on the last layer's rows.
    last_tapped: bool = false,
    /// A tensor-parallel pair: each layer's seams are counted on both ranks.
    pair: bool = false,
    /// The Neural Engine's MLP share steps through every layer.
    ane: bool = false,
    last_attention: bool = true,
};

pub fn keysOnly(f: Facts) bool {
    return f.keeps_none and !f.last_tapped and !f.pair and !f.ane and f.last_attention;
}

/// Whether `r`'s pass at `head` runs its last layer's keys and values alone (these facts on the runner).
pub fn on(r: *const Runner, head: forward.Head) bool {
    const last = r.model.config.layers - 1;
    return keysOnly(.{
        .keeps_none = head == .none,
        .last_tapped = r.capture_taps and std.mem.indexOfScalar(usize, r.taps.ids[0..r.taps.count], last) != null,
        .pair = r.reduce != null,
        .ane = r.model.ane != null,
        .last_attention = r.model.weights.layers[last].mixer == .attention,
    });
}

/// The last layer of a pass that keeps no row: k and v into the cache, as the runner's attentionForward writes them.
pub fn keysForward(ptr: *anyopaque, e: mtl.ComputeEncoder, block: usize, weights: wts.Attention, f: *Frame, rows: u32) !void {
    const r: *Runner = @ptrCast(@alignCast(ptr));
    const plan = (r.current_attention orelse return error.RoundNotReady).*;
    const layer = r.attention_index[block];
    var views: [attn.planning.max_streams]attn.Cache = undefined;
    const caches = r.layerCaches(layer, r.current_round orelse return error.RoundNotReady, &views);
    try r.bindings[layer].upload(plan, caches);
    try r.model.kernels.quant(e, weights.kv, f.get(.input), forward.sums(f, .sums, rows), f.get(.dims), f.get(.mixed), rows);
    e.barrier();
    const kv: u32 = @intCast(r.model.config.kvDim());
    try r.model.glue.unstack(e, f.get(.mixed), f.get(.key), .{ .rows = rows, .width = kv, .stride = 2 * kv, .offset = 0 });
    try r.model.glue.unstack(e, f.get(.mixed), f.get(.value), .{ .rows = rows, .width = kv, .stride = 2 * kv, .offset = kv });
    e.barrier();
    try r.attention.keysOnly(e, &r.scratch, &r.bindings[layer], plan, .{ .qg = f.get(.q_gate), .key = f.get(.key), .q_gain = wts.ref(weights.q_norm), .k_gain = wts.ref(weights.k_norm), .norm_q = f.get(.norm_query), .norm_k = f.get(.norm_key), .query = f.get(.query), .rotated_key = f.get(.rotated_key), .positions = f.get(.positions), .rows = rows, .eps = r.model.config.eps, .theta = r.model.config.rope_theta, .h_end = r.current_placed.h_end, .w_end = r.current_placed.w_end, .plane = f.capacity, .geometry = r.geometry }, f.get(.value), caches);
}

test "qwen27 prompt tail: each planted reader of the last layer's rows refuses it" {
    const open: Facts = .{ .keeps_none = true };
    try std.testing.expect(keysOnly(open));
    var refused: usize = 0;
    inline for (.{ "last_tapped", "pair", "ane" }) |name| {
        var f = open;
        @field(f, name) = true;
        if (!keysOnly(f)) refused += 1 else std.debug.print("tail kept with {s}\n", .{name});
    }
    for ([_]Facts{ .{ .keeps_none = false }, .{ .keeps_none = true, .last_attention = false } }) |f| {
        if (!keysOnly(f)) refused += 1;
    }
    try std.testing.expectEqual(@as(usize, 5), refused);
}
