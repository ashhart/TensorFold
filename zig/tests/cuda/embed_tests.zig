//! embed_rows.cu's MLX affine rows (2 to 6 or 8 bits, a bf16 scale) against float64's rounding points, bit for bit.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rows = @import("norm_rows_tests.zig");
const ec = @import("embed_check.zig");

const bfr = core.bf16_round;
const counts = rows.counts;
const most = rows.most;
const Rig = rows.Rig;
const vocab = 300;

/// sqrt(d) as MLX multiplies by it: its bf16 value.
pub fn scaleOf(d: usize) f32 {
    return bfr.round32(@sqrt(@as(f32, @floatFromInt(d))));
}

/// A table of `bits` wide codes: scales and biases in the checkpoint's ranges (-0.18 to 0.32, -4.1 to 2.7).
fn table(r: *Rig, d: usize, bits: u4, group: usize) !struct { w: []u32, s: []u16, b: []u16 } {
    const codes = try r.a.alloc(u32, vocab * d);
    for (codes) |*q| q.* = r.rng.int(u32) & ((@as(u32, 1) << bits) - 1);
    const s = try r.a.alloc(u16, vocab * d / group);
    const b = try r.a.alloc(u16, vocab * d / group);
    for (s, b) |*x, *y| {
        x.* = bfr.bits(r.rng.float(f32) * 0.5 - 0.18);
        y.* = bfr.bits(r.rng.float(f32) * 6.8 - 4.1);
    }
    return .{ .w = try ec.pack(r.a, codes, bits), .s = s, .b = b };
}

/// One form at every row count: the 64-row call bit-equal to float64, the smaller calls its leading rows.
fn form(e: *const cuda.embed_rows.EmbedRows, r: *Rig, d: usize, bits: u4, group: usize, scale: ?f32, stats: *ec.Stats) !void {
    const t = try table(r, d, bits, group);
    const ids = try r.a.alloc(i32, most);
    for (ids, 0..) |*x, i| x.* = @intCast(if (i == 0) vocab - 1 else r.rng.uintLessThan(usize, vocab));
    const did = try r.dev(std.mem.sliceAsBytes(ids));
    const dw = try r.dev(std.mem.sliceAsBytes(t.w));
    const ds = try r.dev(std.mem.sliceAsBytes(t.s));
    const db = try r.dev(std.mem.sliceAsBytes(t.b));
    var os: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        os[c] = try r.out(m * d * 2);
        try e.affine(r.s, did, dw, ds, db, os[c], m, d, .{ .bits = bits, .group = group, .scale = scale });
    }
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "embed affine {d}-bit, groups of {d}, {d} wide, scale {?d}", .{ bits, group, d, scale });
    const o = try r.back(u16, os[counts.len - 1], most * d);
    var msg: [200]u8 = undefined;
    const k: ec.Case = .{ .rows = most, .d = d, .bits = bits, .group = group, .scale = scale, .ids = ids, .w = t.w, .s = t.s, .b = t.b };
    if (try ec.verify(k, o, stats, &msg)) |why| try check.expect(false, "{s}: {s}", .{ label, why });
    for (counts[0 .. counts.len - 1], 0..) |m, c|
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, os[c], m * d)), std.mem.sliceAsBytes(o[0 .. m * d]));
    if (bits == 4 and group != 128 and scale == null) { // the new kernel at scale 1 gives tf_embed_rows_affine4's bytes
        const one = try r.out(most * d * 2);
        try e.affine(r.s, did, dw, ds, db, one, most, d, .{ .bits = 4, .group = group, .scale = 1 });
        try check.sameBytes("embed affine at scale 1 and affine4", std.mem.sliceAsBytes(try r.back(u16, one, most * d)), std.mem.sliceAsBytes(o));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(31);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
        r.s.deinit();
    }
    var e = try cuda.embed_rows.EmbedRows.load(gpu.d);
    defer e.unload();
    var stats: ec.Stats = .{};
    for ([_]u4{ 2, 3, 4, 5, 6, 8 }) |bits| {
        for ([_]usize{ 3840, 5376 }) |d| try form(&e, &r, d, bits, 64, scaleOf(d), &stats); // Gemma 4 12B, 31B
        try form(&e, &r, 3840, bits, 64, null, &stats);
        for ([_]usize{ 64, 1984, 2048, 2112 }) |d| try form(&e, &r, d, bits, 64, scaleOf(3840), &stats); // a block's edges
        try form(&e, &r, 1024, bits, 32, scaleOf(1024), &stats);
        try form(&e, &r, 1024, bits, 128, null, &stats);
    }
    try form(&e, &r, 2560, 4, 32, null, &stats); // affine4's group of 32
    const z: u64 = 0;
    for ([_]cuda.embed_rows.EmbedRows.Affine{ .{ .bits = 7 }, .{ .bits = 1 }, .{ .group = 48 }, .{ .scale = @sqrt(@as(f32, 3840)) }, .{ .scale = std.math.inf(f32) } }) |o|
        try std.testing.expectError(error.RowShape, e.affine(r.s, z, z, z, z, z, 1, 3840, o));
    try std.testing.expectError(error.RowShape, e.affine(r.s, z, z, z, z, 8, 1, 3840, .{ .bits = 8, .scale = 62 }));
    check.pass("embed rows MLX affine: {d} outputs bit-equal to float64's rounding points (2 to 6 and 8 bits, groups of 32, 64 and 128, scaled and not); every row count's bits equal; scale 1 gives affine4's bytes", .{stats.equal});
}
