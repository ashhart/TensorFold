//! TF_Q35_ATTN_SKIP_EMPTY through the forward's launch trace: off, today's launches; on, only the empty one left out.
const std = @import("std");
const t = std.testing;
const forward = @import("cuda_forward.zig");
const engine = @import("cuda_engine.zig");
const geom = @import("cuda_geometry.zig");
const layer_defs = @import("cuda_layers.zig");
const Skip = @import("cuda_shapes.zig").Skip;
const Trace = @import("cuda_launch_trace.zig").Trace;

const partial8 = 21;
const rows8 = 22;

fn fnHandle(n: usize) @import("cuda").Function {
    return .{ .d = undefined, .handle = @ptrFromInt(n) };
}

/// One decode attention layer and its dense MLP (an FP8 cache), traced at `skip`; the tags it noted.
fn traced(skip: Skip, out: *[64]Trace.Tag) !usize {
    var trace = Trace.init(t.allocator);
    defer trace.deinit();
    @import("cuda_launch_trace.zig").active = &trace;
    defer @import("cuda_launch_trace.zig").active = null;
    const bf: layer_defs.Proj = .{ .bf = .{ .w = 0, .n = 1, .k = 1 } };
    const attn: layer_defs.Attention = .{ .q = bf, .k = bf, .v = bf, .o = bf, .q_norm = 0, .k_norm = 0 };
    const mlp: layer_defs.Mlp = .{ .dense = .{ .gate = bf, .up = bf, .down = bf } };
    var ls = [_]layer_defs.Layer{.{ .input_norm = 0, .post_norm = 0, .mlp = mlp, .mixer = .{ .attention = attn } }};
    var e: engine.Engine = undefined;
    e.gpa = t.allocator;
    e.g = .{ .hidden = 64, .intermediate = 128, .layers = 1, .vocab = 16, .k_heads = 1, .v_heads = 1, .q_heads = 2, .kv_heads = 1, .eps = 1e-6, .tied = true, .context = 4096, .linear = @splat(false), .embeds = false };
    e.layers = &ls;
    e.fns = undefined;
    e.fns.head_norm = fnHandle(1);
    e.fns.head_norm_shift = fnHandle(2);
    e.fns.attn_merge = fnHandle(23);
    e.fns.attn_gate = fnHandle(24);
    e.fns.mlp_act = fnHandle(25);
    e.fp8 = undefined;
    e.fp8.rope = fnHandle(3);
    e.fp8.attn_partial8_whole = fnHandle(partial8);
    e.fp8.attn_rows8_whole = fnHandle(rows8);
    e.affine = undefined;
    e.affine.pdl = false;
    e.kv8 = true;
    e.rope_table = 0;
    e.tables.ptr = 0;
    e.features = 0;
    e.int8 = .{};
    e.learning = null;
    e.mlp = null;
    e.pockets = .{ .gpa = t.allocator, .d = undefined, .hidden = 64 };
    var no_pockets: [1]@typeInfo(@FieldType(@TypeOf(e.pockets), "layers")).pointer.child = .{null}; // layer 0: none
    e.pockets.layers = &no_pockets;
    e.taught = 0;
    e.n_taps = 0;
    inline for (.{ "x", "xs", "kraw", "vraw", "qraw", "qn", "kn", "y", "mix", "r", "pm", "pl", "po", "gate", "up", "act" }) |f| @field(e, f) = 0;
    var shape = engine.Engine.Shape.init(1, 1, true, false, 1);
    shape.part = .layers; // the layer only: no final norm or head
    shape.skip = skip;
    try forward.body(&e, shape);
    @memcpy(out[0..trace.tags.items.len], trace.tags.items);
    return trace.tags.items.len;
}

fn without(tags: []const Trace.Tag, handle: usize, out: *[64]Trace.Tag) usize {
    var n: usize = 0;
    for (tags) |tag| {
        if (std.meta.eql(tag, Trace.Tag{ .launch = handle })) continue;
        out[n] = tag;
        n += 1;
    }
    return n;
}

test "attention skip: off launches both attention kernels; on leaves out exactly the empty one, nothing else" {
    var today: [64]Trace.Tag = undefined;
    const n = try traced(.none, &today);
    var seen: [2]usize = .{ 0, 0 };
    for (today[0..n]) |tag| {
        if (std.meta.eql(tag, Trace.Tag{ .launch = partial8 })) seen[0] += 1;
        if (std.meta.eql(tag, Trace.Tag{ .launch = rows8 })) seen[1] += 1;
    }
    try t.expectEqual([2]usize{ 1, 1 }, seen);
    for ([_]Skip{ .rows, .partial }, [_]usize{ rows8, partial8 }) |skip, left_out| {
        var got: [64]Trace.Tag = undefined;
        const m = try traced(skip, &got);
        var want: [64]Trace.Tag = undefined;
        const w = without(today[0..n], left_out, &want);
        try t.expectEqual(w, m);
        for (want[0..w], got[0..m]) |a, b| try t.expect(std.meta.eql(a, b));
    }
}
