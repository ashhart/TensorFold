//! TF_QMMF_HALF_BN's FP4, FP8 and FP8G half tiles against the 64-column tile: bytes at rows 1-16, then times.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const rig = @import("flashnext_rig.zig");

const Mode = cuda.qmmf.Mode;
const half_n = cuda.qmmf_half;

const Shape = struct { n: usize, k: usize };

/// Qwen3.8-27B's decode projections (MLP, GDN, attention, outputs), then column edges and the a/b heads' width.
const shapes = [_]Shape{
    .{ .n = 17408, .k = 5120 }, .{ .n = 5120, .k = 17408 }, .{ .n = 10240, .k = 5120 }, .{ .n = 6144, .k = 5120 },
    .{ .n = 12288, .k = 5120 }, .{ .n = 1024, .k = 5120 },  .{ .n = 5120, .k = 6144 },  .{ .n = 48, .k = 5120 },
    .{ .n = 72, .k = 1024 },    .{ .n = 131, .k = 2048 },
};

/// Tensor scales: one, small, subnormal and infinite.
const scales = [_]f32{ 1.0, 0x1p-12, 1e-40, std.math.inf(f32) };

/// bf16 specials an adversarial row mixes in: zeros, subnormals, the largest finite and infinities.
const specials = [_]u16{ 0x0000, 0x8000, 0x0001, 0x8001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80 };

/// A random weight in `mode`, packed for the lane: raw bytes and scale bits when adversarial, else bounded values.
fn build(r: *rig.Rig, a: std.mem.Allocator, rng: std.Random, mode: Mode, sh: Shape, adversarial: bool) !cuda.qmmf.Weight {
    const n = sh.n;
    const k = sh.k;
    const np = cuda.modelopt.padded(n);
    switch (mode) {
        .fp4 => {
            const codes = try a.alloc(u8, n * k / 2);
            rng.bytes(codes);
            const blocks = try a.alloc(u8, n * k / 16);
            for (blocks) |*s| s.* = if (adversarial) rng.int(u8) else 0x30 + rng.uintLessThan(u8, 16);
            const words = try a.alloc(u32, np / 64 * (k / 64) * 512);
            cuda.nvfp4.packWords(words, codes, n, k);
            const tiles = try a.alloc(u8, np * (k / 16));
            cuda.nvfp4.packScales(tiles, blocks, n, k);
            return cuda.nvfp4.weight(try r.dev(std.mem.sliceAsBytes(words)), try r.dev(tiles), 1.0, n, k);
        },
        .fp8, .fp8g => {
            const codes = try a.alloc(u8, n * k);
            for (codes) |*c| c.* = if (adversarial) rng.int(u8) else rng.int(u8) & 0xF7; // no e4m3 NaN
            const tiled = try a.alloc(u8, cuda.fp8.padded(n) * k);
            cuda.fp8.packCodes(tiled, codes, n, k);
            const codes_dev = try r.dev(tiled);
            if (mode == .fp8) return cuda.modelopt.fp8Weight(codes_dev, 1.0, n, k);
            // kernel tiles [npad/64][k/64][64] of fp32: any bits when adversarial
            const tiles = try a.alloc(f32, cuda.fp8.padded(n) * (k / 64));
            for (tiles) |*s| s.* = if (adversarial) @bitCast(rng.int(u32)) else 0x1p-8 * (1 + rng.float(f32));
            return cuda.fp8.weight(codes_dev, try r.dev(std.mem.sliceAsBytes(tiles)), n, k);
        },
        else => unreachable,
    }
}

fn refill(r: *rig.Rig, ptr: u64, bytes: usize) !void {
    const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = bytes };
    try b.fill8(0xff, r.s.handle);
}

pub fn run(gpu: check.Gpu) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x5b32);
    const rng = prng.random();
    var r = try rig.Rig.init(gpu, a, rng);
    defer r.deinit();
    var lane = try cuda.qmmf.Lane.load(gpu.d, @intCast(try gpu.ctx.capability() / 10));
    defer lane.unload();
    const set = try half_n.load(&lane);
    var cells: usize = 0;
    for (half_n.modes) |mode| for (shapes) |sh| for ([_]bool{ false, true }) |adversarial| {
        const w0 = try build(&r, a, rng, mode, sh, adversarial);
        const xs = r.bfs(16 * sh.k, if (adversarial) 4.0 else 1.0);
        if (adversarial) for (xs) |*v| {
            if (rng.uintLessThan(u8, 8) == 0) v.* = specials[rng.uintLessThan(usize, specials.len)];
        };
        const x = try r.dev(std.mem.sliceAsBytes(xs));
        const part = try r.out(@max(16, lane.partBytes(16, w0)));
        const bytes = 17 * sh.n * 4; // 16 rows of fp32 and a guard row nothing may write
        const ys = [2]u64{ try r.out(bytes), try r.out(bytes) };
        const tensor_scales: []const f32 = if (adversarial) &scales else scales[0..1];
        for (tensor_scales) |scale| for (1..17) |m| inline for (.{ cuda.qmmf.Out.bf16, cuda.qmmf.Out.f32 }) |out| {
            var w = w0;
            w.scale = scale;
            for (ys, 0..) |y, on| {
                try refill(&r, y, bytes);
                lane.half_n = if (on == 1) set else null;
                lane.half_below = if (on == 1) std.math.maxInt(usize) else 0;
                try lane.matmulOut(r.s, x, sh.k, m, w, y, out, part);
            }
            try r.s.synchronize();
            try check.sameBytes("half tiles (FP4, FP8, FP8G) equal the 64-column tile", try r.back(u8, ys[1], bytes), try r.back(u8, ys[0], bytes));
            cells += 1;
        };
        if (!adversarial) try time(&r, gpu, &lane, set, mode, sh, w0, x, ys[0], part);
    };
    lane.half_n = null;
    check.pass("lane-half-n: {d} FP4, FP8 and FP8G calls at rows 1-16, both outputs, adversarial values: bytes equal", .{cells});
}

/// The 64-column tile against the half tile, warm, at 1 and 16 rows (this engine's lanes only).
fn time(r: *rig.Rig, gpu: check.Gpu, lane: *cuda.qmmf.Lane, set: half_n.Set, mode: Mode, sh: Shape, w: cuda.qmmf.Weight, x: u64, y: u64, part: u64) !void {
    for ([_]usize{ 1, 16 }) |m| {
        var us: [2]f64 = undefined;
        const reps = 200;
        for (0..2) |on| {
            lane.half_n = if (on == 1) set else null;
            lane.half_below = if (on == 1) std.math.maxInt(usize) else 0;
            const t0 = check.now(gpu.io);
            for (0..reps) |_| try lane.matmul(r.s, x, sh.k, m, w, y, part);
            try r.s.synchronize();
            us[on] = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e3 / reps;
        }
        const p = cuda.qmmf.planFor(m, w, lane.clusters);
        std.debug.print("lane-half-n {t} rows {d} n={d} k={d} (sk {d}): 64-col {d:.1} us, half {d:.1} us\n", .{ mode, m, sh.n, sh.k, p.sk, us[0], us[1] });
    }
}
