//! q27-gemm: the wide 4-bit products (tf_q27_gemm_q4w) against the 64x64 ones, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const qwen27 = @import("qwen35");
const check = @import("check.zig");
const rig = @import("q27_back_rig.zig");
const to = qwen27.train_ops;
const Ops = to.Ops;
const Gpu = check.Gpu;
const Rig = rig.Rig;

const Shape = struct { n_out: usize, k_in: usize }; // the forward linear's [out, in]: the product is [m, out] -> [m, in]

const shapes = [_]Shape{
    .{ .n_out = 17408, .k_in = 5120 }, // gate, up
    .{ .n_out = 5120, .k_in = 17408 }, // down
    .{ .n_out = 12288, .k_in = 5120 }, // attention q (q and gate)
    .{ .n_out = 1024, .k_in = 5120 }, // k, v
    .{ .n_out = 5120, .k_in = 6144 }, // o, DeltaNet's out
};
const row_counts = [_]usize{ 1, 61, 130, 984 };

fn product(r: *Rig, o: Ops, wide: bool, a: u64, bf: bool, lda: usize, w: to.Q4, c: u64, m: usize, acc: bool, parts: u64) ![]u32 {
    @constCast(o.k).wide = wide;
    try o.product(a, bf, lda, w, c, w.k, m, acc, parts);
    return r.back(u32, c, m * w.k);
}

pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x9e3_27b);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        r.release();
        r.bufs.deinit(gpu.gpa);
    }
    var k = try to.Kernels.load(gpu.d);
    defer k.deinit();
    const o: Ops = .{ .k = &k, .s = s };
    var cases: usize = 0;
    for (shapes) |sh| {
        const hq = try r.q4(sh.n_out, sh.k_in, 0.05);
        const w = try hq.dev(&r);
        const maxm = row_counts[row_counts.len - 1];
        const a32 = try r.a.alloc(f32, maxm * sh.n_out);
        for (a32) |*x| x.* = @floatCast(r.uni(-1, 1) * @exp(r.uni(-6, 2)));
        const a16 = try r.a.alloc(u16, maxm * sh.n_out);
        for (a16, a32) |*y, x| y.* = rig.bf16(x);
        const da32 = try r.dev(f32, a32);
        const da16 = try r.dev(u16, a16);
        const init = try r.a.alloc(f32, maxm * sh.k_in);
        for (init) |*x| x.* = @floatCast(r.uni(-1, 1));
        const c = try r.dev(f32, init);
        const parts = try r.zeros(64 * maxm * sh.k_in * 4);
        for (row_counts) |m| for ([_]bool{ false, true }) |bf| for ([_]bool{ false, true }) |acc| for ([_]bool{ false, true }) |split| {
            var got: [2][]u32 = undefined;
            for ([_]bool{ false, true }, 0..) |wide, i| {
                const bytes = std.mem.sliceAsBytes(init[0 .. m * sh.k_in]);
                try (cuda.DeviceBuffer{ .d = gpu.d, .ptr = c, .len = bytes.len }).upload(0, bytes);
                got[i] = try product(&r, o, wide, if (bf) da16 else da32, bf, sh.n_out, w, c, m, acc, if (split) parts else 0);
            }
            try check.sameBytes("q27-gemm", std.mem.sliceAsBytes(got[1]), std.mem.sliceAsBytes(got[0]));
            cases += 1;
        };
        // a row list: every third row of 984, against the same rows of the whole product
        {
            const m = maxm;
            var list: std.ArrayList(i32) = .empty;
            var i: usize = 0;
            while (i < m) : (i += 3) try list.append(r.a, @intCast(i));
            const rows = try r.dev(i32, list.items);
            const zero = try r.a.alloc(f32, m * sh.k_in);
            @memset(zero, 0);
            const bytes = std.mem.sliceAsBytes(zero);
            const cb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = c, .len = bytes.len };
            try cb.upload(0, bytes);
            k.wide = true;
            try o.productRows(da32, sh.n_out, w, c, sh.k_in, rows, list.items.len, false);
            const part = try r.back(u32, c, m * sh.k_in);
            try cb.upload(0, bytes);
            try o.product(da32, false, sh.n_out, w, c, sh.k_in, m, false, 0);
            const whole = try r.back(u32, c, m * sh.k_in);
            for (0..m) |row| {
                const got = part[row * sh.k_in ..][0..sh.k_in];
                if (row % 3 == 0) {
                    try check.sameBytes("q27-gemm rows", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(whole[row * sh.k_in ..][0..sh.k_in]));
                } else for (got) |x| try check.expect(x == 0, "q27-gemm rows: row {d} not in the list was written", .{row});
            }
            cases += 1;
        }
        // times at 984 rows, fp32 A, no split
        var times: [2]f64 = undefined;
        for ([_]bool{ false, true }, 0..) |wide, i| {
            k.wide = wide;
            try o.product(da32, false, sh.n_out, w, c, sh.k_in, maxm, false, 0);
            try s.synchronize();
            const t0 = check.now(gpu.io);
            for (0..10) |_| try o.product(da32, false, sh.n_out, w, c, sh.k_in, maxm, false, 0);
            try s.synchronize();
            times[i] = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e10;
        }
        const flops = 2.0 * @as(f64, @floatFromInt(maxm * sh.n_out * sh.k_in));
        std.debug.print("RESULT q27-gemm [{d}, {d}] x 4-bit [{d}, {d}]: 64x64 {d:.2} ms ({d:.1} TFLOPS), wide {d:.2} ms ({d:.1} TFLOPS)\n", .{ maxm, sh.n_out, sh.n_out, sh.k_in, times[0] * 1e3, flops / times[0] / 1e12, times[1] * 1e3, flops / times[1] / 1e12 });
        r.release();
    }
    // the DeltaNet backward's projections: gdn_train's narrow, dequantization and GEMM against the wide product
    const gt = cuda.gdn_train;
    var gk = try gt.Kernels.load(gpu.d);
    defer gk.deinit();
    const go: gt.Ops = .{ .k = &gk, .s = s };
    for ([_]Shape{ .{ .n_out = 16480, .k_in = 5120 }, .{ .n_out = 5120, .k_in = 6144 } }) |sh| {
        const hq = try r.q4(sh.n_out, sh.k_in, 0.05);
        const w = try hq.dev(&r);
        const maxm = row_counts[row_counts.len - 1];
        const a32 = try r.a.alloc(f32, maxm * sh.n_out);
        for (a32) |*x| x.* = @floatCast(r.uni(-1, 1) * @exp(r.uni(-6, 2)));
        const da32 = try r.dev(f32, a32);
        const init = try r.a.alloc(f32, maxm * sh.k_in);
        for (init) |*x| x.* = @floatCast(r.uni(-1, 1));
        const c = try r.dev(f32, init);
        var sc: gt.Scratch = undefined;
        sc.xb = try r.zeros(maxm * @max(sh.n_out, sh.k_in) * 2);
        sc.wt = try r.zeros(sh.n_out * sh.k_in * 2);
        const bytes = std.mem.sliceAsBytes(init);
        const cb: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = c, .len = bytes.len };
        for (row_counts) |m| for ([_]bool{ false, true }) |acc| {
            try cb.upload(0, bytes);
            try go.product(da32, .{ .w = w.w, .s = w.s, .b = w.b, .n = w.n, .k = w.k, .group = w.group }, c, m, sc, acc);
            const want = try r.back(u32, c, m * sh.k_in);
            try cb.upload(0, bytes);
            k.wide = true;
            try o.product(da32, false, sh.n_out, w, c, sh.k_in, m, acc, 0);
            const got = try r.back(u32, c, m * sh.k_in);
            try check.sameBytes("q27-gemm DeltaNet projection", std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want));
            cases += 1;
        };
        r.release();
    }
    check.pass("q27-gemm: {d} cases byte for byte (wide tiles against 64x64 and against the DeltaNet's dequantized GEMM, gathered rows against whole)", .{cases});
}
