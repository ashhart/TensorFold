//! GLP final against FP64/FP32 math and the real mhc_cuda final at width 5120 (Matt Suiche/weightless' program head).

const std = @import("std");
const cuda = @import("cuda");
const dsv41 = @import("dsv41_kernels");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");

pub const FinalCase = struct {
    name: []const u8,
    width: usize,
    rows: usize,
    xs: usize = 0, // 0: dense
    nw16: bool,
    seed: u64,
};

const eps: f32 = 1e-6;

/// Streams [rows, xs], the carried pre-mix [rows, 4] (row 0 one-hot, as the head's first value), norm weights.
const Data = struct {
    x: []u16,
    pre: []f32,
    nw: []u8,
    a: std.mem.Allocator,

    fn init(a: std.mem.Allocator, rng: *ref.Rng, width: usize, rows: usize, xs: usize, nw16: bool) !Data {
        const x = try a.alloc(u16, rows * xs);
        errdefer a.free(x);
        for (x, 0..) |*v, i| v.* = ref.padWord(i);
        for (0..rows) |r| for (x[r * xs ..][0 .. 4 * width]) |*v| {
            v.* = ref.spread(rng);
        };
        const pre = try a.alloc(f32, rows * 4);
        errdefer a.free(pre);
        for (pre, 0..) |*q, i| q.* = if (i < 4) (if (i == 0) 1 else 0) else @floatCast(rng.unit() * 1.5);
        const nw = try a.alloc(u8, width * if (nw16) @as(usize, 2) else 4);
        for (0..width) |k| {
            if (nw16) {
                std.mem.writeInt(u16, nw[2 * k ..][0..2], ref.rne32(@floatCast(0.5 + rng.unit())), .little);
            } else {
                std.mem.writeInt(u32, nw[4 * k ..][0..4], @bitCast(@as(f32, @floatCast(0.5 + rng.unit()))), .little);
            }
        }
        return .{ .x = x, .pre = pre, .nw = nw, .a = a };
    }

    fn deinit(s: Data) void {
        s.a.free(s.x);
        s.a.free(s.pre);
        s.a.free(s.nw);
    }
};

pub fn finalCase(c: rig.Ctx, k: FinalCase) !void {
    const a = c.gpu.gpa;
    const w = k.width;
    const xs = if (k.xs == 0) 4 * w else k.xs;
    var rng = ref.Rng.init(k.seed);
    const data = try Data.init(a, &rng, w, k.rows, xs, k.nw16);
    defer data.deinit();
    const out0 = try a.alloc(u16, k.rows * w);
    defer a.free(out0);
    for (out0, 0..) |*v, i| v.* = ref.padWord(i ^ 0x2a) ^ 0x8000;
    var xb = try rig.Buf.init(c.gpu, rig.bytes(data.x));
    defer xb.free();
    var pb = try rig.Buf.init(c.gpu, rig.bytes(data.pre));
    defer pb.free();
    var nb = try rig.Buf.init(c.gpu, data.nw);
    defer nb.free();
    var ob = try rig.Buf.init(c.gpu, rig.bytes(out0));
    defer ob.free();
    const ym = try a.alloc(u16, w);
    defer a.free(ym);
    const y64 = try a.alloc(f64, w);
    defer a.free(y64);
    var t: ref.Tally = .{};
    var first: ?[]align(16) u8 = null;
    defer if (first) |f| a.free(f);
    var repeat_diff: usize = 0;
    for (0..5) |rep| {
        try ob.write(rig.bytes(out0)); // dirty output memory every repeat
        try c.ops.final(.{
            .x = xb.ptr(),
            .xs = @intCast(xs),
            .pre = pb.ptr(),
            .nw = nb.ptr(),
            .out = ob.ptr(),
            .eps = eps,
            .nwbf16 = @intFromBool(k.nw16),
            .R = @intCast(k.rows),
            .width = @intCast(w),
        });
        try c.sync();
        const got_b = try ob.read(a);
        if (rep > 0) {
            if (!std.mem.eql(u8, got_b, first.?)) repeat_diff += 1;
            a.free(got_b);
            continue;
        }
        first = got_b;
        const got = rig.words(got_b);
        for (0..k.rows) |r| {
            const xin = data.x[r * xs ..][0 .. 4 * w];
            const q: [4]f32 = data.pre[r * 4 ..][0..4].*;
            ref.finalRow(xin, q, data.nw, k.nw16, eps, ym);
            ref.finalRef(xin, q, data.nw, k.nw16, eps, y64);
            for (0..w) |i| t.add(r * w + i, got[r * w + i], ym[i], y64[i], 0x1p-6 * @abs(y64[i]) + 0x1p-133); // two bf16 ulps
        }
    }
    const guards = try xb.guards(a) and try pb.guards(a) and try nb.guards(a) and try ob.guards(a);
    const xk = try xb.read(a);
    defer a.free(xk);
    const kept = std.mem.eql(u8, xk, rig.bytes(data.x));
    if (!t.clean() or repeat_diff != 0 or !guards or !kept or t.max_ulp > 2) {
        std.debug.print("  {d} elements: {d} off the host model, {d} past 2 ulp of FP64 (first {?d}), max {d} ulp, repeats {d}, guards {any}, inputs kept {any}\n", .{ t.n, t.model_diff, t.over, t.first, t.max_ulp, repeat_diff, guards, kept });
        return rig.fail("final {s}: seed 0x{x}", .{ k.name, k.seed });
    }
    check.pass("glp final {s} (norm weight {s}): {d} elements, {d} off the host model of mHC final, {d} FP64 rounding flips (max {d} ulp, bound 2)," ++
        " 5 repeats identical on dirty output, guards intact, seed 0x{x}", .{ k.name, if (k.nw16) "bf16" else "fp32", t.n, t.model_diff, t.flips, t.max_ulp, k.seed });
}

/// Feed identical streams through mhc_cuda mode 3 and GLP final; require identical normalized rows.
pub fn mhcEqual(c: rig.Ctx, k: *const dsv41.Kernels, rows: usize, nw16: bool, seed: u64) !void {
    const a = c.gpu.gpa;
    const D = 5120;
    var rng = ref.Rng.init(seed);
    const data = try Data.init(a, &rng, D, rows, 4 * D, nw16);
    defer data.deinit();
    const zeros = try a.alloc(u8, rows * 4 * D * 2 + 4 * 40 * 32 * 4 * rows);
    defer a.free(zeros);
    @memset(zeros, 0);
    const post0 = try a.alloc(f32, rows * 4);
    defer a.free(post0);
    @memset(post0, 0);
    const comb = try a.alloc(f32, rows * 16);
    defer a.free(comb);
    for (0..rows) |r| for (0..16) |i| {
        comb[r * 16 + i] = if (i / 4 == i % 4) 1 else 0;
    };
    var xb = try rig.Buf.init(c.gpu, rig.bytes(data.x));
    defer xb.free();
    var pb = try rig.Buf.init(c.gpu, rig.bytes(data.pre));
    defer pb.free();
    var nb = try rig.Buf.init(c.gpu, data.nw);
    defer nb.free();
    var xo = try rig.Buf.init(c.gpu, zeros[0 .. rows * 4 * D * 2]); // the new streams
    defer xo.free();
    var gb = try rig.Buf.init(c.gpu, zeros[0 .. rows * D * 2]); // the gathered partials: zero, world 1
    defer gb.free();
    var post = try rig.Buf.init(c.gpu, rig.bytes(post0));
    defer post.free();
    var cb = try rig.Buf.init(c.gpu, rig.bytes(comb));
    defer cb.free();
    var part = try rig.Buf.init(c.gpu, zeros[0 .. rows * 4 * 40 * 32 * 4]);
    defer part.free();
    var coll = try rig.Buf.init(c.gpu, zeros[0 .. rows * D * 2]); // the collapsed row mHC also writes
    defer coll.free();
    var cnt = try rig.Buf.init(c.gpu, zeros[0..16]);
    defer cnt.free();
    var out_m = try rig.Buf.init(c.gpu, zeros[0 .. rows * D * 2]);
    defer out_m.free();
    var out_g = try rig.Buf.init(c.gpu, zeros[0 .. rows * D * 2]);
    defer out_g.free();

    const o = k.others(c.stream);
    const args: dsv41.ops.mhc.Args = .{
        .x = xb.ptr(),
        .xs = 4 * D,
        .xout = xo.ptr(),
        .g = gb.ptr(),
        .gr = @intCast(rows * D),
        .world = 1,
        .gbf16 = 1,
        .post = post.ptr(),
        .comb = cb.ptr(),
        .pre = pb.ptr(),
        .part = part.ptr(),
        .c = coll.ptr(),
        .nw = nb.ptr(),
        .nwbf16 = @intFromBool(nw16),
        .out = out_m.ptr(),
        .cnt = cnt.ptr(),
        .R = @intCast(rows),
        .eps = eps,
    };
    try o.mhcBoundary(args, 3, false);
    try c.sync();
    const new_streams = try xo.read(a);
    defer a.free(new_streams);
    if (!std.mem.eql(u8, new_streams, rig.bytes(data.x))) return rig.fail("mhc final {d} rows: the identity post changed the streams", .{rows});
    try c.ops.final(.{ .x = xb.ptr(), .xs = 4 * D, .pre = pb.ptr(), .nw = nb.ptr(), .out = out_g.ptr(), .eps = eps, .nwbf16 = @intFromBool(nw16), .R = @intCast(rows), .width = D });
    try c.sync();
    const m = try out_m.read(a);
    defer a.free(m);
    const g = try out_g.read(a);
    defer a.free(g);
    var diff: usize = 0;
    for (rig.words(m), rig.words(g)) |p, q| {
        if (!ref.same(p, q)) diff += 1;
    }
    if (diff != 0) return rig.fail("final {d} rows (norm weight {s}): {d} of {d} words differ from mhc_cuda's final", .{ rows, if (nw16) "bf16" else "fp32", diff, rows * D });
    check.pass("glp final == mhc_cuda final (mode 3), {d} rows, norm weight {s}: {d} words bit-identical on the same streams, seed 0x{x}", .{ rows, if (nw16) "bf16" else "fp32", rows * D, seed });
}
