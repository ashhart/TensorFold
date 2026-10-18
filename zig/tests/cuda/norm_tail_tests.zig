//! norms.cu's bf16 residual tails at Gemma 4's widths and the 16-byte tiles' edges: float64 ranges, exact sums, bits.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rows = @import("norm_rows_tests.zig");
const tail_ = @import("norms_tail_check.zig");

const bfr = core.bf16_round;
const counts = rows.counts;
const most = rows.most;
const eps: f32 = 1e-6;

/// A tail's options as a family calls them.
const Form = struct { name: []const u8, round: bool, residual: bool, branch: bool, scalar: bool };

const input_norm: Form = .{ .name = "input norm (MLX)", .round = true, .residual = false, .branch = false, .scalar = false };
const attn_tail: Form = .{ .name = "attention tail (MLX)", .round = true, .residual = true, .branch = true, .scalar = false };
const dense_tail: Form = .{ .name = "dense tail with layer scalar (MLX)", .round = true, .residual = true, .branch = true, .scalar = true };
const forms = [_]Form{
    input_norm,
    attn_tail,
    dense_tail,
    .{ .name = "scalar alone (MLX)", .round = true, .residual = false, .branch = false, .scalar = true },
    .{ .name = "dense tail, rounded once", .round = false, .residual = true, .branch = true, .scalar = true },
    .{ .name = "branch norm alone, rounded once", .round = false, .residual = false, .branch = true, .scalar = false },
};

fn case(r: *rows.Rig, n: *const cuda.norms.Norms, width: usize, f: Form, stats: *tail_.Stats) !void {
    const x = r.bfs(most * width, 1);
    const res = r.bfs(most * width, 4);
    const wb = try r.a.alloc(u16, width);
    const w = try r.a.alloc(u16, width);
    for (wb, w, 0..) |*p, *q, i| { // signed, as Gemma 4's norm weights (12B: -143 to 193), one in 97 large
        const big: f32 = if (i % 97 == 3) 30 else 1;
        p.* = rows.bf16((r.rng.float(f32) * 8 - 2) * big);
        q.* = rows.bf16((r.rng.float(f32) * 8 - 2) * big);
    }
    const sc = [8]u16{ bfr.bits(0.004 + 0.9 * r.rng.float(f32)), 0, 0, 0, 0, 0, 0, 0 }; // one bf16, 16 bytes
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dr = try r.dev(std.mem.sliceAsBytes(res));
    const dwb = try r.dev(std.mem.sliceAsBytes(wb));
    const dw = try r.dev(std.mem.sliceAsBytes(w));
    const dsc = try r.dev(std.mem.sliceAsBytes(&sc));
    var hs: [counts.len]u64 = undefined;
    var ys: [counts.len]u64 = undefined;
    var xss: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        hs[c] = try r.out(m * width * 2);
        ys[c] = try r.out(m * width * 2);
        xss[c] = try r.out(m * width / 64 * 4);
        const t: cuda.norms.Tail = .{ .round = f.round, .branch = if (f.branch) dwb else null, .scalar = if (f.scalar) dsc else null };
        try n.addRmsTail(r.s, dx, if (f.residual) dr else null, dw, hs[c], ys[c], xss[c], m, width, eps, t);
    }
    var name: [128]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "norm tail {s}, width {d}", .{ f.name, width });
    const h = try r.back(u16, hs[counts.len - 1], most * width);
    const y = try r.back(u16, ys[counts.len - 1], most * width);
    const xs = try r.back(f32, xss[counts.len - 1], most * width / 64);
    const k: tail_.Case = .{ .rows = most, .width = width, .round = f.round, .x = x, .w = w, .r = if (f.residual) res else null, .wb = if (f.branch) wb else null, .sc = if (f.scalar) sc[0] else null };
    var msg: [200]u8 = undefined;
    if (try tail_.verify(r.gpu.gpa, k, h, y, xs, stats, &msg)) |why| try check.expect(false, "{s}: {s}", .{ label, why });
    for (counts[0 .. counts.len - 1], 0..) |m, c| {
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, ys[c], m * width)), std.mem.sliceAsBytes(y[0 .. m * width]));
        try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(f32, xss[c], m * width / 64)), std.mem.sliceAsBytes(xs[0 .. m * width / 64]));
        if (k.writesH()) try check.sameBytes(label, std.mem.sliceAsBytes(try r.back(u16, hs[c], m * width)), std.mem.sliceAsBytes(h[0 .. m * width]));
    }
}

/// No option set is addRmsH itself: the same launch, the same bytes.
fn unchanged(r: *rows.Rig, n: *const cuda.norms.Norms, width: usize) !void {
    const x = try r.dev(std.mem.sliceAsBytes(r.bfs(most * width, 1)));
    const res = try r.dev(std.mem.sliceAsBytes(r.bfs(most * width, 4)));
    const wv = try r.a.alloc(u16, width);
    for (wv, r.weights(width)) |*o, v| o.* = rows.bf16(v);
    const w = try r.dev(std.mem.sliceAsBytes(wv));
    var outs: [2][3]u64 = undefined;
    for (&outs) |*o| for (o, [_]usize{ 2, 2, 4 }) |*p, b| {
        p.* = try r.out(most * width * b / @as(usize, if (b == 4) 64 else 1));
    };
    try n.addRmsH(r.s, x, res, w, .bf16, outs[0][0], outs[0][1], outs[0][2], most, width, eps);
    try n.addRmsTail(r.s, x, res, w, outs[1][0], outs[1][1], outs[1][2], most, width, eps, .{});
    for (outs[0], outs[1], [_]usize{ 2, 2, 4 }) |p, q, b| {
        const len = most * width / @as(usize, if (b == 4) 64 else 1);
        try check.sameBytes("norm tail without options", std.mem.sliceAsBytes(try r.back(u16, q, len * b / 2)), std.mem.sliceAsBytes(try r.back(u16, p, len * b / 2)));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(4);
    var r: rows.Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
        r.s.deinit();
    }
    var n = try cuda.norms.Norms.load(gpu.d);
    defer n.unload();
    var stats: tail_.Stats = .{};
    for ([_]usize{ 3840, 5376, 1024 }) |width| for (forms) |f| try case(&r, &n, width, f, &stats); // 12B, 31B, the 12B's assistant
    for ([_]usize{ 64, 1984, 2048, 2112, 8192 }) |width| for ([_]Form{ input_norm, dense_tail }) |f| try case(&r, &n, width, f, &stats);
    try unchanged(&r, &n, 3840);
    const sc = try r.out(16);
    const buf = try r.out(2 * 3904);
    try std.testing.expectError(error.RowShape, n.addRmsTail(r.s, buf, null, buf, buf, buf, null, 1, 3848, eps, .{ .round = true }));
    try std.testing.expectError(error.RowShape, n.addRmsTail(r.s, buf, null, buf, buf, buf, null, 1, 3840, eps, .{ .scalar = sc + 1 }));
    try std.testing.expectError(error.RowShape, n.addRmsTail(r.s, buf, null, buf, buf, buf, null, 1, 8256, eps, .{ .round = true }));
    check.pass("norm tail: {d} outputs bit-equal to float64's rounding points, {d} within an fp32 inverse's range, {d} non-finite as float64; {d} group sums equal to the fp32 tree; every row count's bits equal; no options = addRmsH's bytes; bad rows refused", .{ stats.decided, stats.ranged, stats.nonfinite, stats.sums });
}
