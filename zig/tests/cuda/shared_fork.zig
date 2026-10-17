//! The latent MoE with its shared expert on the side stream against the one-stream MoE, on the checkpoint's weights.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const hold_bytes = 256 << 20;

/// MODEL in both precisions: every latent MoE layer at 1 to max_rows rows, forked against one stream, byte for byte.
pub fn run(gpu: check.Gpu, model: []const u8) !void {
    for ([_]nemotron.Precision{ .checkpoint, .full }) |p| try layers(gpu, model, p);
}

/// Each layer's delta with the side stream held back by copies before its fork, against the one-stream delta.
fn layers(gpu: check.Gpu, model: []const u8, precision: nemotron.Precision) !void {
    const gpa = gpu.gpa;
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 4096, .graphs = false, .precision = precision });
    defer e.deinit();
    const c = e.c;
    const b = &e.b;
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
    const held: nemotron.kernels.Ops = .{ .k = &e.k, .s = side.s };
    const most = nemotron.state.max_rows;
    const x = try gpa.alloc(u16, most * c.hidden);
    defer gpa.free(x);
    const want = try gpa.alloc(u8, most * c.hidden * 2);
    defer gpa.free(want);
    const got = try gpa.alloc(u8, want.len);
    defer gpa.free(got);
    var prng = std.Random.DefaultPrng.init(0x5eed_f04c);
    const rng = prng.random();
    var cases: usize = 0;
    for (moes.items, 0..) |moe, layer| for (1..most + 1) |rows| {
        for (x[0 .. rows * c.hidden]) |*v| v.* = @truncate(@as(u32, @bitCast((rng.float(f32) - 0.5) * 4.0)) >> 16);
        try e.ops().upload(b.y, std.mem.sliceAsBytes(x[0 .. rows * c.hidden]));
        const len = rows * c.hidden * 2;
        try nemotron.latent.run(&plain, moe, rows);
        try e.ops().download(want[0..len], b.delta);
        // the one-stream pass left the right shared rows behind: a sum that skips the join must read NaN instead
        const shared_rows = b.ymoe + @as(u64, rows * c.top_k * c.ownShares()) * c.latent * 4;
        try e.ops().fill32(shared_rows, 0x7fc00000, rows * c.ownShares() * c.hidden);
        for (0..8) |_| try held.copy(hold.ptr + hold_bytes / 2, hold.ptr, hold_bytes / 2);
        try nemotron.latent.run(&forked, moe, rows);
        try e.ops().download(got[0..len], b.delta);
        try e.stream.synchronize();
        var what: [64]u8 = undefined;
        try check.sameBytes(try std.fmt.bufPrint(&what, "shared-fork ({t}): layer {d} at {d} rows", .{ precision, layer, rows }), got[0..len], want[0..len]);
        cases += 1;
    };
    check.pass("shared-fork ({t}): {d} (layer, rows) cases, the forked delta byte-equal to the one-stream delta", .{ precision, cases });
}
