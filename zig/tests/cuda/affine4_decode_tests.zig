//! Affine-4 decode at the Qwen3.5 shapes: every row count's bits, and the segmented launch's against decode().
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const qlinear = cuda.qlinear;
const Rig = quant_ref.RigT;

/// Projections of one input as the engines launch them: Qwen3.8-27B's, then Qwen3.5-35B-A3B's (n each, one k).
const Set = struct { name: []const u8, k: usize, ns: []const usize };
const sets = [_]Set{
    .{ .name = "27B delta in", .k = 5120, .ns = &.{ 10240, 6144, 48, 48 } },
    .{ .name = "27B delta out", .k = 6144, .ns = &.{5120} },
    .{ .name = "27B attention in", .k = 5120, .ns = &.{ 12288, 1024, 1024 } },
    .{ .name = "27B mlp in", .k = 5120, .ns = &.{ 17408, 17408 } },
    .{ .name = "27B mlp out", .k = 17408, .ns = &.{5120} },
    .{ .name = "27B head", .k = 5120, .ns = &.{248320} },
    .{ .name = "35B delta in", .k = 2048, .ns = &.{ 8192, 4096, 32, 32 } },
    .{ .name = "35B delta out", .k = 4096, .ns = &.{2048} },
    .{ .name = "35B attention in", .k = 2048, .ns = &.{ 8192, 512, 512 } },
    .{ .name = "35B head", .k = 2048, .ns = &.{248320} },
    .{ .name = "short K", .k = 512, .ns = &.{ 2048, 200, 64 } },
};
const row_counts = [_]usize{ 1, 2, 3, 4, 8, 16 };

/// tf_q35_sums's 64-group sums: fp32, in order.
fn sums(r: *Rig, x: []const u16, rows: usize, k: usize) ![]f32 {
    const out = try r.a.alloc(f32, rows * k / 64);
    for (0..rows) |row| for (0..k / 64) |g| {
        var s: f32 = 0;
        for (0..64) |j| s += @floatCast(quant_ref.helpers.bfOf(x[row * k + g * 64 + j]));
        out[row * (k / 64) + g] = s;
    };
    return out;
}

fn weight(r: *Rig, a4: *const qlinear.Affine4Kernels, n: usize, k: usize, words: u64, scales: u64, biases: u64) !qlinear.Affine4 {
    const lay = qlinear.Affine4.layout(n, k);
    const q: qlinear.Affine4 = .{ .w = try r.zeros(lay.words), .s = try r.zeros(lay.scales), .b = try r.zeros(lay.scales), .n = n, .k = k, .npad = lay.npad };
    try a4.pack(r.s, words, scales, biases, q);
    return q;
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(29);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    const kk = cuda.kernels;
    const images = [_][]const u8{ kk.qmm_group, kk.qmm_prefill, kk.lane_gemv, kk.affine4_pack };
    var mods: [images.len]cuda.Module = undefined;
    var loaded: usize = 0;
    defer for (mods[0..loaded]) |*md| md.unload();
    for (images, &mods) |img, *md| {
        md.* = try cuda.Module.load(gpu.d, img);
        loaded += 1;
    }
    const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
    var a4 = try qlinear.Affine4Kernels.resolve(gpu.d, mods[0], mods[1], mods[2], mods[3], sms, false);
    defer a4.deinit();
    var name: [160]u8 = undefined;
    const most = row_counts[row_counts.len - 1];
    const Out = qlinear.Affine4Kernels.Out;
    for (sets) |set| {
        const k = set.k;
        const mark = r.bufs.items.len;
        defer {
            for (r.bufs.items[mark..]) |*b| b.free();
            r.bufs.shrinkRetainingCapacity(mark);
        }
        const x = try r.bfs(most * k, 1.0);
        const dx = try r.dev(std.mem.sliceAsBytes(x));
        const dxs = try r.dev(std.mem.sliceAsBytes(try sums(&r, x, most, k)));
        const qs = try r.a.alloc(qlinear.Affine4, set.ns.len);
        for (set.ns, 0..) |n, p| {
            const words = try r.a.alloc(u32, n * k / 8);
            for (words) |*w| w.* = r.rng.int(u32);
            const scales = try r.a.alloc(u16, n * k / 64);
            const biases = try r.a.alloc(u16, n * k / 64);
            for (scales, biases) |*sv, *bv| {
                sv.* = quant_ref.helpers.bf16Of((r.rng.float(f32) * 0.9 + 0.1) * 0.0025);
                bv.* = quant_ref.helpers.bf16Of(-(r.rng.float(f32) * 0.4 + 0.8) * 0.02);
            }
            const dw = try r.dev(std.mem.sliceAsBytes(words));
            const ds = try r.dev(std.mem.sliceAsBytes(scales));
            const db = try r.dev(std.mem.sliceAsBytes(biases));
            qs[p] = try weight(&r, &a4, n, k, dw, ds, db);
        }
        // decode() of each projection at 16 rows: the bits every other launch and row count must give
        const want = try r.a.alloc([]u8, set.ns.len);
        for (set.ns, qs, want) |n, q, *w| {
            const o = try r.zeros(most * n * 2);
            try a4.decode(r.s, dx, dxs, q, o, most);
            w.* = try r.back(o, most * n * 2);
        }
        const outs = try r.a.alloc(Out, set.ns.len);
        for (outs, qs, set.ns) |*e, q, n| e.* = .{ .q = q, .out = try r.zeros(most * n * 2) };
        for (row_counts) |rows| {
            for (set.ns, qs, want) |n, q, w| {
                const o = try r.zeros(rows * n * 2);
                try a4.decode(r.s, dx, dxs, q, o, rows);
                try check.sameBytes(try std.fmt.bufPrint(&name, "affine-4 [{d}, {d}]: decode() at {d} rows equals its 16-row call", .{ n, k, rows }), try r.back(o, rows * n * 2), w[0 .. rows * n * 2]);
                const one = try r.zeros(rows * n * 2);
                try a4.decodeMany(r.s, dx, dxs, &.{.{ .q = q, .out = one }}, rows);
                try check.sameBytes(try std.fmt.bufPrint(&name, "affine-4 [{d}, {d}]: one segment at {d} rows equals decode()", .{ n, k, rows }), try r.back(one, rows * n * 2), w[0 .. rows * n * 2]);
            }
            for (outs) |o| try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = o.out, .len = most * o.q.n * 2 }).fill8(0xff, r.s.handle);
            try a4.decodeMany(r.s, dx, dxs, outs, rows);
            for (outs, want, set.ns) |o, w, n|
                try check.sameBytes(try std.fmt.bufPrint(&name, "{s}: [{d}, {d}] of {d} segments at {d} rows equals decode()", .{ set.name, n, k, set.ns.len, rows }), try r.back(o.out, rows * n * 2), w[0 .. rows * n * 2]);
        }
    }
    check.pass("affine4-decode: {d} sets by {d} row counts: every row count, one segment and every set's segmented launch byte-equal to decode() at 16 rows", .{ sets.len, row_counts.len });
}
