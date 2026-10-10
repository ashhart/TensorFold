//! Block-FP8 projections against the Python lane matmul's bytes (oracle/fp8_lane.py), direct and through qlinear.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const Fixture = @import("fixture.zig").Fixture;
const Gpu = check.Gpu;

const fp8 = cuda.fp8;
const qmmf = cuda.qmmf;

pub fn lane(gpu: Gpu, dir: []const u8) !void {
    var fx = try Fixture.open(gpu.gpa, gpu.io, dir);
    defer fx.deinit();
    const a = gpu.gpa;
    const n: usize = @intCast(try fx.int("n"));
    const k: usize = @intCast(try fx.int("k"));
    const npad = fp8.padded(n);

    const codes = try fx.bytes("codes");
    defer a.free(codes);
    const inv_bytes = try fx.bytes("scale_inv");
    defer a.free(inv_bytes);
    const inv = std.mem.bytesAsSlice(f32, @as([]align(1) u8, inv_bytes));
    const inv_aligned = try a.alloc(f32, inv.len);
    defer a.free(inv_aligned);
    for (inv, inv_aligned) |v, *o| o.* = v;

    const w8 = try a.alloc(u8, npad * k);
    defer a.free(w8);
    fp8.packCodes(w8, codes, n, k);
    const want_w8 = try fx.bytes("w8");
    defer a.free(want_w8);
    try check.sameBytes("packed codes against _fragment_order", w8, want_w8);
    const bs = try a.alloc(f32, npad * (k / fp8.group));
    defer a.free(bs);
    fp8.packScales(bs, inv_aligned, n, k);
    const want_bs = try fx.bytes("bs");
    defer a.free(want_bs);
    try check.sameBytes("scale tiles against from_rows", std.mem.sliceAsBytes(bs), want_bs);
    check.pass("fp8 repack: [{d}, {d}] codes and scale tiles equal the Python layout", .{ n, k });

    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, w8);
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(bs));
    defer ds.free();
    const cap = try gpu.ctx.capability();
    var l = try qmmf.Lane.load(gpu.d, cap / 10);
    defer l.unload();
    var stream = try cuda.Stream.init(gpu.d, false); // blocking: launches wait for fromHost's legacy-stream copies
    defer stream.deinit();
    const w = fp8.weight(dw.ptr, ds.ptr, n, k);

    var it = std.mem.tokenizeScalar(u8, try fx.string("rows"), ',');
    while (it.next()) |tok| {
        const m = try std.fmt.parseInt(usize, tok, 10);
        var name: [16]u8 = undefined;
        const x = try fx.bytes(try std.fmt.bufPrint(&name, "x{d}", .{m}));
        defer a.free(x);
        const want = try fx.bytes(try std.fmt.bufPrint(&name, "y{d}", .{m}));
        defer a.free(want);
        var dx = try cuda.DeviceBuffer.fromHost(gpu.d, x);
        defer dx.free();
        var dy = try cuda.DeviceBuffer.alloc(gpu.d, m * n * 2);
        defer dy.free();
        const pb = l.partBytes(m, w);
        var dp = try cuda.DeviceBuffer.alloc(gpu.d, @max(pb, 4));
        defer dp.free();
        try l.matmul(stream, dx.ptr, k, m, w, dy.ptr, dp.ptr);
        try stream.synchronize();
        const got = try check.download(gpu, dy);
        defer a.free(got);
        try check.sameBytes(try std.fmt.bufPrint(&name, "y{d}", .{m}), got, want);
        try viaLinear(gpu, l, stream, w, dx.ptr, k, m, dp.ptr, want);
        const p = qmmf.plan(m, n, k, l.clusters);
        check.pass("fp8 lane: {d} rows x [{d}, {d}] equal Python's bytes (tile {d}, {d} slices, fused {}, cluster {})", .{ m, n, k, p.bm, p.sk, p.fused, p.cluster });
    }
}

/// The same rows through qlinear's decode and prompt calls: a weight view must not change a bit.
fn viaLinear(gpu: Gpu, l: qmmf.Lane, s: cuda.Stream, w: qmmf.Weight, x: u64, k: usize, m: usize, part: u64, want: []const u8) !void {
    const lin: cuda.qlinear.Linear = .{ .lane = &l };
    var dz = try cuda.DeviceBuffer.alloc(gpu.d, want.len);
    defer dz.free();
    for ([_]bool{ false, true }) |prompt| {
        const in: cuda.qlinear.Rows = .{ .x = x, .ldx = k, .part = part };
        try dz.fill8(0xff, s.handle); // a call that writes nothing fails rather than pass on the last call's bytes
        if (prompt) try lin.prompt(s, .{ .fp8g = w }, in, dz.ptr, m) else try lin.decode(s, .{ .fp8g = w }, in, dz.ptr, m);
        try s.synchronize();
        const got = try check.download(gpu, dz);
        defer gpu.gpa.free(got);
        try check.sameBytes(if (prompt) "qlinear prompt" else "qlinear decode", got, want);
    }
    check.pass("qlinear: {d} rows through decode and prompt equal the lane matmul's bytes", .{m});
}
