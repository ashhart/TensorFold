//! The latent MoE with its shared expert on the side stream against the one-stream MoE, on the checkpoint's weights.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const hold_bytes = 256 << 20;
const shared_streams = [_]usize{ 2, 4, 8 };

/// MODEL in both precisions: every latent MoE layer at 1 to max_rows rows, forked against one stream, byte for byte.
pub fn run(gpu: check.Gpu, model: []const u8) !void {
    for ([_]nemotron.Precision{ .checkpoint, .full }) |p| try layers(gpu, model, p);
}

/// Each layer, lone at every width and shared by 2, 4 and 8 streams, side stream held back, against one stream.
fn layers(gpu: check.Gpu, model: []const u8, precision: nemotron.Precision) !void {
    const gpa = gpu.gpa;
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 4096, .graphs = false, .precision = precision });
    defer e.deinit();
    const c = e.c;
    var plain = e.forward(null);
    plain.side = null;
    const forked = e.forward(null);
    const side = forked.side orelse return error.NoSideStream;
    var moes: std.ArrayList(nemotron.weights.MoE) = .empty;
    defer moes.deinit(gpa);
    for (e.w.blocks) |blk| if (blk.kind == .moe) try moes.append(gpa, blk.moe);
    if (e.w.mtp) |m| try moes.append(gpa, m.moe);
    var hold = try cuda.DeviceBuffer.alloc(e.ctx.d, hold_bytes);
    defer hold.free();
    const most = nemotron.state.windowRows(c);
    const x = try gpa.alloc(u16, most * c.hidden);
    defer gpa.free(x);
    const want = try gpa.alloc(u8, most * c.hidden * 2);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, want.len);
    defer gpa.free(got);
    var prng = std.Random.DefaultPrng.init(0x5eed_f04c);
    const t: Trial = .{ .e = e, .plain = &plain, .held = .{ .k = &e.k, .s = side.s }, .hold = hold.ptr, .x = x, .want = want, .got = got, .rng = prng.random() };
    var segs: [nemotron.engine.max_streams]nemotron.Seg = undefined;
    var what: [96]u8 = undefined;
    var cases: usize = 0;
    for (moes.items, 0..) |moe, layer| {
        for (1..most + 1) |rows| {
            try t.same(&forked, moe, rows, try std.fmt.bufPrint(&what, "shared-fork ({t}): layer {d} at {d} rows", .{ precision, layer, rows }));
            cases += 1;
        }
        for (shared_streams) |n| for ([_]usize{ n, most }) |rows| {
            const window = forked.routedWindow(split(&segs, &e.b, n, rows));
            if (window.side == null) return error.SharedWindowWithoutSideStream;
            try t.same(&window, moe, rows, try std.fmt.bufPrint(&what, "shared-fork ({t}): layer {d}, {d} streams of {d} rows", .{ precision, layer, n, rows }));
            cases += 1;
        };
    }
    check.pass("shared-fork ({t}): {d} (layer, rows) cases, lone and shared, the forked delta byte-equal to the one-stream delta", .{ precision, cases });
}

/// `rows` rows over `n` streams' segments in turn, as Engine.verifyShared lays a shared window out.
fn split(segs: []nemotron.Seg, b: *const nemotron.state.Buffers, n: usize, rows: usize) []const nemotron.Seg {
    var row0: usize = 0;
    for (segs[0..n], 0..) |*s, k| {
        const r = rows / n + @intFromBool(k < rows % n);
        s.* = .{ .b = b, .row0 = row0, .rows = r, .sampled = false };
        row0 += r;
    }
    return segs[0..n];
}

const Trial = struct {
    e: *nemotron.Engine,
    plain: *const nemotron.Forward,
    held: nemotron.kernels.Ops, // the side stream: copies across `hold` keep it behind the main stream at each fork
    hold: u64,
    x: []u16,
    want: []u8,
    got: []u8,
    rng: std.Random,

    /// One layer on random rows: `forked`'s delta byte-equal to the one-stream delta.
    fn same(t: Trial, forked: *const nemotron.Forward, moe: nemotron.weights.MoE, rows: usize, what: []const u8) !void {
        const e = t.e;
        const c = e.c;
        const b = &e.b;
        for (t.x[0 .. rows * c.hidden]) |*v| v.* = @truncate(@as(u32, @bitCast((t.rng.float(f32) - 0.5) * 4.0)) >> 16);
        try e.ops().upload(b.y, std.mem.sliceAsBytes(t.x[0 .. rows * c.hidden]));
        const len = rows * c.hidden * 2;
        try nemotron.latent.run(t.plain, moe, rows);
        try e.ops().download(t.want[0..len], b.delta);
        // the one-stream pass left the right shared rows behind: a sum that skips the join must read NaN instead
        const shared_rows = b.ymoe + @as(u64, rows * c.top_k * c.ownShares()) * c.latent * 4;
        try e.ops().fill32(shared_rows, 0x7fc00000, rows * c.ownShares() * c.hidden);
        for (0..8) |_| try t.held.copy(t.hold + hold_bytes / 2, t.hold, hold_bytes / 2);
        try nemotron.latent.run(forked, moe, rows);
        try e.ops().download(t.got[0..len], b.delta);
        try e.stream.synchronize();
        try check.sameBytes(what, t.got[0..len], t.want[0..len]);
    }
};
