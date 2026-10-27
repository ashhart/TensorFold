//! `tf-cuda-test exl3-plans`: every (K splits, warps) and layout of the 27B's EXL3 shapes timed at 1 and 16 rows.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const exl3 = cuda.exl3;
const Gpu = check.Gpu;
const Rig = quant_ref_tests.RigT;

const Shape = struct { name: []const u8, k: usize, n: usize, halves: u32 };

/// Qwen3.8-27B's projections in turboderp's 4.00bpw pack (6-bit head).
const shapes = [_]Shape{
    .{ .name = "gate_up", .k = 5120, .n = 17408, .halves = 8 },
    .{ .name = "down", .k = 17408, .n = 5120, .halves = 8 },
    .{ .name = "qkv", .k = 5120, .n = 10240, .halves = 8 },
    .{ .name = "z", .k = 5120, .n = 6144, .halves = 8 },
    .{ .name = "out_o", .k = 6144, .n = 5120, .halves = 8 },
    .{ .name = "q", .k = 5120, .n = 12288, .halves = 8 },
    .{ .name = "k_v", .k = 5120, .n = 1024, .halves = 8 },
    .{ .name = "head", .k = 5120, .n = 248320, .halves = 12 },
};

const splits = [_]u32{ 1, 2, 4, 5, 8, 10, 16, 20, 32, 40 };
const warps = [_]u32{ 2, 4, 8 };
const row_counts = [_]usize{ 1, 16 };
/// Copies of a weight rotated through, so each call reads DRAM rather than L2 (at least this many bytes in all).
const sweep_bytes = 256 << 20;

pub fn run(gpu: Gpu, only: ?[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(5);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lin = try exl3.Linear.load(gpu.d);
    defer lin.unload();
    for (shapes) |sh| {
        if (only) |o| if (!std.mem.eql(u8, o, sh.name)) continue;
        try shape(&r, &lin, sh);
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }
}

fn shape(r: *Rig, lin: *const exl3.Linear, sh: Shape) !void {
    const words = (sh.k / 16) * (sh.n / 16) * sh.halves * 4;
    const host = try r.a.alloc(u32, words);
    for (host) |*v| v.* = r.rng.int(u32);
    const copies = @min(64, @max(1, sweep_bytes / (words * 4)));
    var trellis: [64]u64 = undefined;
    for (trellis[0..copies]) |*t| t.* = try r.dev(std.mem.sliceAsBytes(host));
    const suh = try r.a.alloc(u16, sh.k);
    const svh = try r.a.alloc(u16, sh.n);
    @memset(suh, 0x3C00);
    @memset(svh, 0x3C00);
    const dsuh = try r.dev(std.mem.sliceAsBytes(suh));
    const dsvh = try r.dev(std.mem.sliceAsBytes(svh));
    const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(16 * sh.k, 1.0)));
    const y = try r.zeros(16 * sh.n * 2);
    const xh = try r.zeros(16 * sh.k * 2);
    const z = try r.zeros(40 * 16 * sh.n * 4);
    const counters = try exl3_tests.zeroed(r, 8 * (sh.n / 128) * 4);
    const kt = sh.k / 16;
    const def = exl3.plan(sh.k, sh.n, 192, 8);
    for (row_counts) |m| {
        var best: f64 = std.math.inf(f64);
        var best_at: [3]u32 = .{ 0, 0, 0 };
        for ([_]bool{ false, true }) |strips| for (splits) |sk| for (warps) |wk| {
            if (kt % (sk * wk) != 0 or kt / (sk * wk) < 2) continue;
            if (sk > 1 and sh.n / 128 * sk > 4096) continue;
            const st = exl3.strides(sh.k, sh.n, sh.halves, strips);
            var w: exl3.Weight = .{ .trellis = 0, .suh = dsuh, .svh = dsvh, .k = @intCast(sh.k), .n = @intCast(sh.n), .halves = sh.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb, .sk = sk, .wk = wk };
            for (0..2) |c| {
                w.trellis = trellis[c % copies];
                try lin.matmul(r.s, w, x, .bf16, m, y, .bf16, xh, z, counters);
            }
            try r.s.synchronize();
            const reps = @max(copies * 4, 24);
            const t0 = check.now(r.gpu.io);
            for (0..reps) |c| {
                w.trellis = trellis[c % copies];
                try lin.matmul(r.s, w, x, .bf16, m, y, .bf16, xh, z, counters);
            }
            try r.s.synchronize();
            const us = @as(f64, @floatFromInt(check.now(r.gpu.io) - t0)) / 1e3 / @as(f64, @floatFromInt(reps));
            const gbs = @as(f64, @floatFromInt(words * 4)) / us / 1e3;
            const mark: []const u8 = if (!strips and sk == def.sk and wk == def.wk) " (plan)" else "";
            std.debug.print("PLAN {s} m={d} {s} sk={d} wk={d}: {d:.1} us {d:.0} GB/s{s}\n", .{ sh.name, m, if (strips) "strips" else "stored", sk, wk, us, gbs, mark });
            if (us < best) {
                best = us;
                best_at = .{ @intFromBool(strips), sk, wk };
            }
        };
        std.debug.print("BEST {s} m={d}: {s} sk={d} wk={d} {d:.1} us\n", .{ sh.name, m, if (best_at[0] == 1) "strips" else "stored", best_at[1], best_at[2], best });
    }
}

/// The family's fixed plans (cuda_weights_exl3.zig) by shape name, for `midm`.
const fixed = [_]struct { name: []const u8, sk: u32, wk: u32 }{
    .{ .name = "gate_up", .sk = 1, .wk = 4 }, .{ .name = "down", .sk = 1, .wk = 8 },  .{ .name = "qkv", .sk = 1, .wk = 4 },
    .{ .name = "z", .sk = 2, .wk = 4 },       .{ .name = "out_o", .sk = 2, .wk = 8 }, .{ .name = "q", .sk = 1, .wk = 8 },
    .{ .name = "k_v", .sk = 5, .wk = 8 },     .{ .name = "head", .sk = 1, .wk = 4 },
};

/// `tf-cuda-test exl3-midm`: the mid-M kernels against the linear at 17-128 rows, byte for byte, and both timed at 128.
pub fn midm(gpu: Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(9);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lin = try exl3.Linear.load(gpu.d);
    defer lin.unload();
    var bad: usize = 0;
    for (shapes, fixed) |sh, f| {
        const words = (sh.k / 16) * (sh.n / 16) * sh.halves * 4;
        const host = try r.a.alloc(u32, words);
        for (host) |*v| v.* = r.rng.int(u32);
        const suh = try r.a.alloc(u16, sh.k);
        const svh = try r.a.alloc(u16, sh.n);
        for (suh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
        for (svh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
        const st = exl3.strides(sh.k, sh.n, sh.halves, false);
        const w: exl3.Weight = .{ .trellis = try r.dev(std.mem.sliceAsBytes(host)), .suh = try r.dev(std.mem.sliceAsBytes(suh)), .svh = try r.dev(std.mem.sliceAsBytes(svh)), .k = @intCast(sh.k), .n = @intCast(sh.n), .halves = sh.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb, .sk = f.sk, .wk = f.wk };
        const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(128 * sh.k, 1.0)));
        const xh = try r.zeros(128 * sh.k * 2);
        const z = try r.zeros(exl3.Linear.zBytes(w, 128));
        const counters = try exl3_tests.zeroed(&r, exl3.Linear.counterBytes(w));
        const ya = try r.zeros(128 * sh.n * 2);
        const yb = try r.zeros(128 * sh.n * 2);
        for ([_]usize{ 17, 24, 32, 33, 48, 49, 64, 100, 128 }) |m| {
            lin.midm = false;
            try lin.matmul(r.s, w, x, .bf16, m, ya, .bf16, xh, z, counters);
            lin.midm = true;
            try lin.matmul(r.s, w, x, .bf16, m, yb, .bf16, xh, z, counters);
            const a = try r.back(ya, m * sh.n * 2);
            const b = try r.back(yb, m * sh.n * 2);
            if (!std.mem.eql(u8, a, b)) {
                bad += 1;
                std.debug.print("FAIL exl3.midm {s} m={d}: the mid-M bits differ\n", .{ sh.name, m });
            }
        }
        // the mid-M kernels alone against the linear too, where linear_wc would take the rows
        lin.use_wc = false;
        for ([_]usize{ 24, 48, 64, 128 }) |m| {
            lin.midm = false;
            try lin.matmul(r.s, w, x, .bf16, m, ya, .bf16, xh, z, counters);
            lin.midm = true;
            try lin.matmul(r.s, w, x, .bf16, m, yb, .bf16, xh, z, counters);
            if (!std.mem.eql(u8, try r.back(ya, m * sh.n * 2), try r.back(yb, m * sh.n * 2))) {
                bad += 1;
                std.debug.print("FAIL exl3.midm {s} m={d}: the mpg bits differ\n", .{ sh.name, m });
            }
        }
        lin.use_wc = true;
        var us: [3]f64 = undefined;
        for (0..3) |mode| {
            lin.midm = mode >= 1;
            lin.use_wc = mode == 2;
            try lin.matmul(r.s, w, x, .bf16, 128, ya, .bf16, xh, z, counters);
            try r.s.synchronize();
            const t0 = check.now(r.gpu.io);
            for (0..10) |_| try lin.matmul(r.s, w, x, .bf16, 128, ya, .bf16, xh, z, counters);
            try r.s.synchronize();
            us[mode] = @as(f64, @floatFromInt(check.now(r.gpu.io) - t0)) / 1e4;
        }
        lin.use_wc = true;
        std.debug.print("RESULT exl3.midm {s} m=128: linear {d:.0} us, mid-M {d:.0} us, wc {d:.0} us\n", .{ sh.name, us[0], us[1], us[2] });
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }
    bad += try groups(&r, &lin);
    try check.expect(bad == 0, "exl3.midm: {d} row counts differ", .{bad});
    check.pass("exl3.midm", .{});
}

/// `tf-cuda-test exl3-shapes [rows]`: each 27B shape once at its fixed plan, 1 and 16 rows, for a profiler to attach.
pub fn profileShapes(gpu: Gpu, rows_arg: ?[]const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(3);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var lin = try exl3.Linear.load(gpu.d);
    defer lin.unload();
    const ms: []const usize = if (rows_arg) |a| &.{try std.fmt.parseInt(usize, a, 10)} else &.{ 1, 16 };
    for (shapes, fixed) |sh, f| {
        const words = (sh.k / 16) * (sh.n / 16) * sh.halves * 4;
        const host = try r.a.alloc(u32, words);
        for (host) |*v| v.* = r.rng.int(u32);
        const suh = try r.a.alloc(u16, sh.k);
        const svh = try r.a.alloc(u16, sh.n);
        @memset(suh, 0x3C00);
        @memset(svh, 0x3C00);
        const st = exl3.strides(sh.k, sh.n, sh.halves, false);
        const w: exl3.Weight = .{ .trellis = try r.dev(std.mem.sliceAsBytes(host)), .suh = try r.dev(std.mem.sliceAsBytes(suh)), .svh = try r.dev(std.mem.sliceAsBytes(svh)), .k = @intCast(sh.k), .n = @intCast(sh.n), .halves = sh.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb, .sk = f.sk, .wk = f.wk };
        const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(128 * sh.k, 1.0)));
        const xh = try r.zeros(128 * sh.k * 2);
        const z = try r.zeros(exl3.Linear.zBytes(w, 128) + 16);
        const counters = try exl3_tests.zeroed(&r, exl3.Linear.counterBytes(w));
        const y = try r.zeros(128 * sh.n * 2);
        for (ms) |m| {
            std.debug.print("SHAPE {s} m={d} sk={d} wk={d}\n", .{ sh.name, m, f.sk, f.wk });
            for (0..2) |_| try lin.matmul(r.s, w, x, .bf16, m, y, .bf16, xh, z, counters);
            try r.s.synchronize();
        }
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }
}

/// The 27B's projection groups (q k v, gate up, qkv z) in one launch against each member's own, byte for byte.
fn groups(r: *Rig, lin: *const exl3.Linear) !usize {
    const sets = [_][]const usize{ &.{ 5, 6, 6 }, &.{ 0, 0 }, &.{ 2, 3 } }; // indices into shapes / fixed
    var bad: usize = 0;
    for (sets) |set| {
        var ws: [3]exl3.Weight = undefined;
        var ya: [3]u64 = undefined;
        var yb: [3]u64 = undefined;
        for (set, 0..) |si, i| {
            const sh = shapes[si];
            const f = fixed[si];
            const words = (sh.k / 16) * (sh.n / 16) * sh.halves * 4;
            const host = try r.a.alloc(u32, words);
            for (host) |*v| v.* = r.rng.int(u32);
            const suh = try r.a.alloc(u16, sh.k);
            const svh = try r.a.alloc(u16, sh.n);
            for (suh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
            for (svh) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 2 - 1)));
            const st = exl3.strides(sh.k, sh.n, sh.halves, false);
            ws[i] = .{ .trellis = try r.dev(std.mem.sliceAsBytes(host)), .suh = try r.dev(std.mem.sliceAsBytes(suh)), .svh = try r.dev(std.mem.sliceAsBytes(svh)), .k = @intCast(sh.k), .n = @intCast(sh.n), .halves = sh.halves, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb, .sk = f.sk, .wk = f.wk };
            ya[i] = try r.zeros(16 * sh.n * 2);
            yb[i] = try r.zeros(16 * sh.n * 2);
        }
        const g = ws[0..set.len];
        const k = g[0].k;
        const x = try r.dev(std.mem.sliceAsBytes(try r.bfs(16 * k, 1.0)));
        const xh = try r.zeros(3 * 16 * k * 2);
        const z = try r.zeros(3 * 40 * 16 * 12288 * 4);
        const counters = try exl3_tests.zeroed(r, 3 * 8 * (248320 / 128) * 4);
        for ([_]usize{ 1, 7, 16 }) |m| {
            for (g, 0..) |w, i| try lin.matmul(r.s, w, x, .bf16, m, ya[i], .bf16, xh, z, counters);
            try lin.matmulGroup(r.s, g, x, .bf16, m, yb[0..set.len], .bf16, xh, z, counters);
            for (g, 0..) |w, i| if (!std.mem.eql(u8, try r.back(ya[i], m * w.n * 2), try r.back(yb[i], m * w.n * 2))) {
                bad += 1;
                std.debug.print("FAIL exl3.group set {d} member {d} m={d}: the group launch's bits differ\n", .{ set[0], i, m });
            };
        }
        for (r.bufs.items) |*b| b.free();
        r.bufs.clearRetainingCapacity();
    }
    if (bad == 0) check.pass("exl3.group", .{});
    return bad;
}
