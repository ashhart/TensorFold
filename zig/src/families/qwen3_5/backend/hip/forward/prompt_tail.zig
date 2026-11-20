//! Qwen 3.5's HIP facts for core's prompt tail: every reader of the last layer's rows, and when that layer fits.
const std = @import("std");
const hip = @import("hip");
const tail = @import("core").prompt_tail;
const view = @import("../model/view.zig");

pub const rows = tail.rows;

/// What a span's caller reads: the last row's logits and the MTP head's input (`logits`), or every row (`readers`).
pub const Reads = tail.Reads;

/// Every other reader of the last layer's rows, and what that layer must be to stop past its keys and values.
pub const Facts = struct {
    /// A drafter's tap on the last layer: its ring takes every row.
    last_tapped: bool = false,
    /// A trace that reads each layer's rows (the cancel poll reads none).
    traced: bool = false,
    /// An image prompt: its rows take the images' M-RoPE positions, not checked on a device under the tail.
    images: bool = false,
    /// rocBLAS runs the products, whose bits vary with the row count: a tail would not keep the read row's.
    plain: bool = false,
    /// A product past the last layer's keys is `.dense` (a widened chat checkpoint, an unquantized tensor): unchecked.
    dense: bool = false,
    /// Full attention: the layer's only state is its keys and values.
    last_full: bool = true,

    /// The first row the last layer runs past its keys and values for `len` rows read as `reads` (len: none).
    pub fn first(f: Facts, reads: Reads, len: usize) usize {
        if (!f.last_full) return 0;
        const r: Reads = .{ .logits = reads.logits, .readers = reads.readers or f.last_tapped or f.traced or f.images };
        if (tail.unread(r)) return len;
        return tail.start(.{ .total = len, .nseg = 1, .readers = r.readers, .fits = !f.plain and !f.dense }) orelse 0;
    }
};

/// The facts of model `m` under `o`, whose trace reads the rows (`traced`) or not.
pub fn of(m: *const view.Model, o: hip.ops.Ops, traced: bool) Facts {
    const last = m.layers.len - 1;
    return .{
        .last_tapped = m.taps.of(last) != null,
        .traced = traced,
        .images = if (m.frame) |fr| fr.runs.len != 0 else false,
        .plain = o.blas != null,
        .dense = switch (m.layers[last]) {
            .full => |f| densePastKeys(f),
            .linear => false,
        },
        .last_full = m.layers[last] == .full,
    };
}

/// Whether q, o or the MLP of `f` takes the dense kernels; k and v run every row either way.
fn densePastKeys(f: view.Full) bool {
    const mlp = switch (f.mlp) {
        .dense => |d| isDense(d.gate) or isDense(d.up) or isDense(d.down),
        .moe => |r| isDense(r.experts.fused) or isDense(r.experts.down) or
            if (r.experts.shared) |sh| isDense(sh.gate_up) or isDense(sh.down) else false,
    };
    return isDense(f.q) or isDense(f.o) or mlp;
}

fn isDense(p: view.Projection) bool {
    return p.handle == .dense;
}

test "qwen3_5 hip prompt tail: an open chunk keeps its last tile, or only its keys when nothing reads it" {
    const open: Facts = .{};
    try std.testing.expectEqual(@as(usize, 4096 - rows), open.first(.{ .logits = true, .readers = false }, 4096));
    try std.testing.expectEqual(@as(usize, 4096), open.first(.{ .logits = false, .readers = false }, 4096));
    try std.testing.expectEqual(@as(usize, 0), open.first(.{ .logits = true, .readers = false }, rows));
    try std.testing.expectEqual(@as(usize, 0), open.first(.{ .logits = true, .readers = true }, 4096));
    try std.testing.expectEqual(@as(usize, 0), open.first(.{ .logits = false, .readers = true }, 4096));
}

test "qwen3_5 hip prompt tail: each planted reader of the other rows refuses the tail and the keys-only pass" {
    const open: Facts = .{};
    var refused: usize = 0;
    inline for (.{ "last_tapped", "traced", "images" }) |name| {
        var f = open;
        @field(f, name) = true;
        const keys = f.first(.{ .logits = false, .readers = false }, 4096) == 0;
        const last = f.first(.{ .logits = true, .readers = false }, 4096) == 0;
        if (keys and last) refused += 1 else std.debug.print("prompt tail kept with {s}\n", .{name});
    }
    var f = open;
    f.last_full = false;
    if (f.first(.{ .logits = false, .readers = false }, 4096) == 0 and f.first(.{ .logits = true, .readers = false }, 4096) == 0) refused += 1;
    try std.testing.expectEqual(@as(usize, 4), refused);
    // rocBLAS and the dense kernels: k and v run every row as before, so only the tail is refused
    inline for (.{ "plain", "dense" }) |name| {
        f = open;
        @field(f, name) = true;
        try std.testing.expectEqual(@as(usize, 0), f.first(.{ .logits = true, .readers = false }, 4096));
        try std.testing.expectEqual(@as(usize, 4096), f.first(.{ .logits = false, .readers = false }, 4096));
    }
}

test "qwen3_5 hip prompt tail: the model's taps, images, rocBLAS, dense products and last layer are read as facts" {
    const mlx: view.Projection = .{ .n = 64, .k = 64, .handle = .{ .mlx = undefined } };
    const dense: view.Projection = .{ .n = 64, .k = 64, .handle = .{ .dense = .{ .weight = 0 } } };
    var full: view.Full = undefined;
    inline for (.{ "q", "k", "v", "o" }) |name| @field(full, name) = mlx;
    full.mlp = .{ .dense = .{ .gate = mlx, .up = mlx, .down = mlx } };
    var layers = [_]view.Layer{ .{ .linear = undefined }, .{ .full = full } };
    var m: view.Model = undefined;
    m.layers = &layers;
    m.taps = .{};
    m.frame = null;
    const o: hip.ops.Ops = .{ .l = undefined, .bf16 = true, .stream = null, .arena = undefined };
    try std.testing.expectEqual(Facts{}, of(&m, o, false));
    try std.testing.expect(of(&m, o, true).traced);
    m.taps.layers[0] = 1;
    m.taps.n = 1;
    try std.testing.expect(of(&m, o, false).last_tapped);
    m.taps.layers[0] = 0;
    try std.testing.expect(!of(&m, o, false).last_tapped);
    const runs = [_]view.ImageRun{.{ .at = 3, .rows = 4, .features = 0 }};
    m.frame = .{ .positions = 0, .runs = &runs };
    try std.testing.expect(of(&m, o, false).images);
    m.frame = .{ .positions = 0, .runs = &.{} };
    try std.testing.expect(!of(&m, o, false).images);
    var plain = o;
    plain.blas = @ptrFromInt(0x100);
    try std.testing.expect(of(&m, plain, false).plain);
    // dense k and v run every row as on main; a dense q, o or MLP product is a fact
    layers[1].full.k = dense;
    layers[1].full.v = dense;
    try std.testing.expect(!of(&m, o, false).dense);
    layers[1].full.o = dense;
    try std.testing.expect(of(&m, o, false).dense);
    layers[1].full.o = mlx;
    layers[1].full.mlp.dense.down = dense;
    try std.testing.expect(of(&m, o, false).dense);
    var routed: view.Routed = undefined;
    routed.experts.fused = mlx;
    routed.experts.down = mlx;
    routed.experts.shared = null;
    layers[1].full.mlp = .{ .moe = routed };
    try std.testing.expect(!of(&m, o, false).dense);
    layers[1].full.mlp.moe.experts.shared = .{ .gate_up = dense, .down = mlx };
    try std.testing.expect(of(&m, o, false).dense);
    layers[1] = .{ .linear = undefined };
    try std.testing.expect(!of(&m, o, false).last_full);
}
