//! Row-exact 4-bit projections before the M5: 8-row simdgroup-matrix tiles, and a scalar twin for 1-2 rows.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// K chunks by output width alone, so a row's reduction tree never depends on the window's row count.
pub fn splits(n: usize) u32 {
    return if (n <= 64) 32 else if (n <= 6144) 16 else 8;
}

/// Chunks for `n`; a split rank passes the whole width as `whole` so it keeps that width's tree (0 keeps `n`).
pub fn chunks(n: usize, whole: usize) u32 {
    return splits(if (whole != 0) whole else n);
}

/// Output tiles a simdgroup within the 16 KB reduction buffer; tiles never change a row's bits.
pub fn tiles(n: usize, rt: u32, s: u32) u32 {
    var nt: u32 = if (n % 32 == 0) 4 else if (n % 16 == 0) 2 else 1;
    while (nt > 1 and !fits(s, nt, rt)) nt /= 2;
    return nt;
}

/// Two adjacent K chunks a simdgroup (tf_sq_mmp): 16 chunks at two row tiles then fit four column tiles, not two.
pub fn pairs(n: usize, rt: u32, s: u32) bool {
    return s == 16 and rt == 2 and n % 32 == 0 and !fits(s, 4, rt) and fits(s / 2, 4, rt);
}

/// Groups of inputs the scalar kernel stages at 1 or 2 rows (0: it can't take them).
pub fn scalarBlock(rows: usize, s: u32) u32 {
    if (rows < 1 or rows > max_scalar_rows) return 0;
    const xb: u32 = if (rows == 1) 32 else @max(s, 16);
    return if (xb % s == 0 and rows * xb * 76 * 4 <= 20480) xb else 0;
}

pub const max_scalar_rows = 2;
const s_values = [_]u32{ 8, 16, 32 };
const nt_values = [_]u32{ 1, 2, 4 };

pub const Args = extern struct { k: i32, n: i32, r: i32, one: f32 = 1.0 };

/// Whether a tile shape's reduction buffer fits 16 KB (the only shapes `tiles` picks, and the only ones built).
fn fits(s: u32, nt: u32, rt: u32) bool {
    return s * rt * nt * 64 * 4 <= 16384;
}

/// Half operands take 8-chunk tiles only: the widths past 6,144 (gate_up, q, q | k | v, z | b | a).
pub const half_chunks = 8;

pub const Pipelines = struct {
    mma: [s_values.len][nt_values.len][2]?mtl.Pipeline,
    /// tf_sq_mmh at 8 chunks: tf_sq_mma's bits through half operands where tf_sq_fits allows.
    half: [nt_values.len][2]mtl.Pipeline,
    fits: mtl.Pipeline,
    /// tf_sq_prep_s and tf_sq_prep_h: X's group sums, and with them its halves, for the tiles that read them.
    prep: [2]mtl.Pipeline,
    scalar: [s_values.len][2][max_scalar_rows]mtl.Pipeline,
    paired: mtl.Pipeline, // tf_sq_mmp_16_4_2_2: the shapes `pairs` takes
    paired_sums: mtl.Pipeline, // tf_sq_mmx_16_4_2_2: the same with tf_sq_prep_s's sums

    pub fn load(device: mtl.Device) !Pipelines {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const lib = try mtl.Library.fromSource(device, sources.core_simd_qmm, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var p: Pipelines = undefined;
        inline for (s_values, 0..) |s, si| {
            inline for (nt_values, 0..) |nt, ti| inline for (.{ 1, 2 }, 0..) |rt, ri| {
                p.mma[si][ti][ri] = if (comptime fits(s, nt, rt)) try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_sq_mma_{d}_{d}_{d}", .{ s, nt, rt }), false) else null;
            };
            inline for (.{ 1, 2 }, 0..) |nr, ni| inline for (.{ 1, 2 }, 0..) |rs, ri| {
                p.scalar[si][ni][ri] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_sq_scalar_{d}_{d}_{d}", .{ s, nr, rs }), false);
            };
        }
        inline for (nt_values, 0..) |nt, ti| inline for (.{ 1, 2 }, 0..) |rt, ri| {
            p.half[ti][ri] = try mtl.Pipeline.init(device, lib, std.fmt.comptimePrint("tf_sq_mmh_{d}_{d}_{d}", .{ half_chunks, nt, rt }), false);
        };
        p.fits = try mtl.Pipeline.init(device, lib, "tf_sq_fits", false);
        p.prep = .{ try mtl.Pipeline.init(device, lib, "tf_sq_prep_s", false), try mtl.Pipeline.init(device, lib, "tf_sq_prep_h", false) };
        p.paired = try mtl.Pipeline.init(device, lib, "tf_sq_mmp_16_4_2_2", false);
        p.paired_sums = try mtl.Pipeline.init(device, lib, "tf_sq_mmx_16_4_2_2", false);
        return p;
    }

    pub fn deinit(p: *Pipelines) void {
        for (p.mma) |a| for (a) |b| for (b) |x| if (x) |y| y.deinit();
        for (p.half) |a| for (a) |x| x.deinit();
        p.fits.deinit();
        for (p.prep) |x| x.deinit();
        for (p.scalar) |a| for (a) |b| for (b) |x| x.deinit();
        p.paired.deinit();
        p.paired_sums.deinit();
    }
};

/// One dispatch: buffers x, W, S, B, args, out at 0-5; `groups` threadgroups of `threads`; `paired`: tf_sq_mmp.
pub const Call = struct { pipeline: mtl.Pipeline, args: Args, groups: mtl.Size, threads: usize, paired: bool = false };

/// y = x W^T, 4-bit groups of 64: the scalar twin at 1-2 rows if `scalar`, chunk pairs if `paired`, else MMA tiles.
pub fn call(p: *const Pipelines, n: usize, k: usize, rows: usize, scalar: bool, paired: bool, whole: usize) !Call {
    if (n == 0 or n % 8 != 0 or n > std.math.maxInt(i32) or k == 0 or k % 64 != 0 or k > std.math.maxInt(i32)) return error.UnsupportedSimdShape;
    if (rows == 0 or rows > std.math.maxInt(i32)) return error.NoRows;
    const s = chunks(n, whole);
    const si = std.mem.indexOfScalar(u32, &s_values, s).?;
    const args = Args{ .k = @intCast(k), .n = @intCast(n), .r = @intCast(rows) };
    if (scalar and scalarBlock(rows, s) != 0) {
        const nr: usize = if (n > 2048) 2 else 1;
        const sgs: usize = if (n > 2048) @max(1, 16 / ((32 / s) * nr)) else 8;
        const per = sgs * (32 / s) * nr;
        const pipeline = p.scalar[si][nr - 1][rows - 1];
        if (pipeline.maxThreads() < sgs * 32) return error.SimdThreads;
        return .{ .pipeline = pipeline, .args = args, .groups = .{ .width = (n + per - 1) / per }, .threads = sgs * 32 };
    }
    const rt: u32 = @min(2, @as(u32, @intCast((rows + 7) / 8)));
    if (paired and pairs(n, rt, s) and p.paired.maxThreads() >= 256)
        return .{ .pipeline = p.paired, .args = args, .groups = .{ .width = n / 32, .height = (rows + 15) / 16 }, .threads = 256, .paired = true };
    const nt = tiles(n, rt, s);
    const pipeline = p.mma[si][std.mem.indexOfScalar(u32, &nt_values, nt).?][rt - 1] orelse return error.UnsupportedSimdShape;
    const sgs: usize = @min(s, 16, pipeline.maxThreads() / 32);
    if (sgs == 0) return error.SimdThreads;
    return .{ .pipeline = pipeline, .args = args, .groups = .{ .width = (n + 8 * nt - 1) / (8 * nt), .height = (rows + 8 * rt - 1) / (8 * rt) }, .threads = sgs * 32 };
}

/// `call`'s tiles through half operands, or null where that call takes another kernel; bind tf_sq_fits's words at 6.
pub fn callHalf(p: *const Pipelines, n: usize, k: usize, rows: usize, whole: usize) !?Call {
    const c = try call(p, n, k, rows, false, false, whole);
    if (chunks(n, whole) != half_chunks) return null;
    const rt: u32 = @min(2, @as(u32, @intCast((rows + 7) / 8)));
    const nt = tiles(n, rt, half_chunks);
    const pipeline = p.half[std.mem.indexOfScalar(u32, &nt_values, nt).?][rt - 1];
    if (pipeline.maxThreads() < c.threads) return null;
    var h = c;
    h.pipeline = pipeline;
    return h;
}

/// `c` reading tf_sq_prep_s's sums (bound at 7, `encodeSums`), or null where `c` isn't the chunk-pair tiles.
pub fn callSums(p: *const Pipelines, c: Call) ?Call {
    if (!c.paired or p.paired_sums.maxThreads() < c.threads) return null;
    var s = c;
    s.pipeline = p.paired_sums;
    return s;
}

/// tf_sq_mmx at any row count, its tiles as at 16 rows (the contract's small shapes); null where `n` takes no pairs.
pub fn callSumsAt(p: *const Pipelines, n: usize, k: usize, rows: usize) !?Call {
    var c = callSums(p, try call(p, n, k, @max(rows, 16), false, true, 0)) orelse return null;
    c.args.r = @intCast(rows);
    c.groups.height = (rows + 15) / 16;
    return c;
}

/// Bytes of tf_sq_prep's output for `rows` rows of `k` inputs: sums (rows padded to pairs, 16 bytes), then halves.
pub fn prepBytes(k: usize, rows: usize, halves: bool) usize {
    const pairs_ = (rows + 1) / 2 * 2;
    return (pairs_ * (k / 64) + 3) / 4 * 16 + if (halves) pairs_ * k * 2 else 0;
}

/// Writes x's group sums by the tiles' tree into `out` and, with `halves`, x as half, two rows interleaved.
pub fn encodePrep(e: mtl.ComputeEncoder, p: *const Pipelines, k: usize, rows: usize, halves: bool, x: mtl.Buffer, x_off: usize, out: mtl.Buffer, out_off: usize) void {
    e.setPipeline(p.prep[@intFromBool(halves)]);
    e.setBuffer(x, x_off, 0);
    e.setValue(Args{ .k = @intCast(k), .n = 0, .r = @intCast(rows) }, 4);
    e.setBuffer(out, out_off, 7);
    e.dispatchGroups(.{ .width = k / 64, .height = (rows + 31) / 32 }, .{ .width = 32 });
}

/// tf_sq_fits's words for `rows` rows of `k` inputs: one u32 a 16-row tile and 64-input group.
pub fn fitsBytes(k: usize, rows: usize) usize {
    return (rows + 15) / 16 * (k / 64) * 4;
}

/// Writes tf_sq_fits's words for x's rows into `flags`; a tile's word is 1 when every input is zero or a normal half.
pub fn encodeFits(e: mtl.ComputeEncoder, p: *const Pipelines, k: usize, rows: usize, x: mtl.Buffer, x_off: usize, flags: mtl.Buffer, flags_off: usize) void {
    e.setPipeline(p.fits);
    e.setBuffer(x, x_off, 0);
    e.setValue(Args{ .k = @intCast(k), .n = 0, .r = @intCast(rows) }, 4);
    e.setBuffer(flags, flags_off, 6);
    e.dispatchGroups(.{ .width = k / 64, .height = (rows + 15) / 16 }, .{ .width = 32 });
}

/// A checkpoint's [n, k] matrix: words [n][k/8], scales and biases [n][k/64] bf16, at byte offsets.
pub const Matrix = struct { w: mtl.Buffer, w_off: usize = 0, scales: mtl.Buffer, s_off: usize = 0, biases: mtl.Buffer, b_off: usize = 0, n: usize, k: usize };

pub fn encode(e: mtl.ComputeEncoder, c: Call, m: Matrix, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize) void {
    e.setPipeline(c.pipeline);
    e.setBuffer(x, x_off, 0);
    e.setBuffer(m.w, m.w_off, 1);
    e.setBuffer(m.scales, m.s_off, 2);
    e.setBuffer(m.biases, m.b_off, 3);
    e.setValue(c.args, 4);
    e.setBuffer(y, y_off, 5);
    e.dispatchGroups(c.groups, .{ .width = c.threads });
}

/// `encode` with tf_sq_fits's words and tf_sq_prep_h's output (sums and halves) bound, for a `callHalf` call.
pub fn encodeHalf(e: mtl.ComputeEncoder, c: Call, m: Matrix, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize, flags: mtl.Buffer, flags_off: usize, prep: mtl.Buffer, prep_off: usize) void {
    e.setBuffer(flags, flags_off, 6);
    e.setBuffer(prep, prep_off, 7);
    encode(e, c, m, x, x_off, y, y_off);
}

/// `encode` with tf_sq_prep_s's sums bound, for a `callSums` call.
pub fn encodeSums(e: mtl.ComputeEncoder, c: Call, m: Matrix, x: mtl.Buffer, x_off: usize, y: mtl.Buffer, y_off: usize, sums: mtl.Buffer, sums_off: usize) void {
    e.setBuffer(sums, sums_off, 7);
    encode(e, c, m, x, x_off, y, y_off);
}

/// Whether half operands give tf_sq_mma's bits for `m` at 16 and 8 rows, inputs half holds and planted ones it doesn't.
pub fn halfExact(p: *const Pipelines, device: mtl.Device, m: Matrix) !bool {
    if ((try callHalf(p, m.n, m.k, 16, 0)) == null) return false;
    const rows = 16;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = try device.buffer(rows * m.k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x51d_0003);
    const xs = x.slice(u16, rows * m.k);
    for (xs) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 2 - 1)) >> 16);
    // planted in groups 1-3: past half's range (2^20), below its normals (2^-20), a half subnormal (2^-15)
    for ([_]u16{ 0x4980, 0x3580, 0x3800 }, 0..) |v, i| xs[(3 + 5 * i) * m.k + 64 * (i + 1) + 7] = v;
    const flags = try device.buffer(fitsBytes(m.k, rows), mtl.ResourceOptions.shared);
    defer flags.deinit();
    const prep = try device.buffer(prepBytes(m.k, rows, true), mtl.ResourceOptions.shared);
    defer prep.deinit();
    const prep8 = try device.buffer(prepBytes(m.k, 8, true), mtl.ResourceOptions.shared);
    defer prep8.deinit();
    var outs: [3]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer for (outs) |o| o.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    encodeFits(e, p, m.k, rows, x, 0, flags, 0);
    encodePrep(e, p, m.k, rows, true, x, 0, prep, 0);
    encodePrep(e, p, m.k, 8, true, x, 0, prep8, 0);
    e.barrier();
    encode(e, try call(p, m.n, m.k, rows, false, false, 0), m, x, 0, outs[0], 0);
    encodeHalf(e, (try callHalf(p, m.n, m.k, rows, 0)).?, m, x, 0, outs[1], 0, flags, 0, prep, 0);
    encodeHalf(e, (try callHalf(p, m.n, m.k, 8, 0)).?, m, x, 0, outs[2], 0, flags, 0, prep8, 0); // 8-row tiles read the 16-row words
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("simd_qmm half check: {s}", .{why});
        return error.SimdCheckFailed;
    }
    const want = outs[0].slice(u16, rows * m.n);
    return std.mem.eql(u16, want, outs[1].slice(u16, rows * m.n)) and std.mem.eql(u16, want[0 .. 8 * m.n], outs[2].slice(u16, 8 * m.n));
}

/// Whether chunk pairs give tf_sq_mma's bits for `m` (16 rows each way); false where `pairs` never takes it.
pub fn pairedExact(p: *const Pipelines, device: mtl.Device, m: Matrix) !bool {
    return pairCheck(p, device, m, false);
}

/// Whether chunk pairs on tf_sq_prep_s's sums give tf_sq_mma's bits for `m` (17 rows: a pad row in the prepass).
pub fn summedExact(p: *const Pipelines, device: mtl.Device, m: Matrix) !bool {
    return pairCheck(p, device, m, true);
}

fn pairCheck(p: *const Pipelines, device: mtl.Device, m: Matrix, sums: bool) !bool {
    if (!pairs(m.n, 2, splits(m.n))) return false;
    const rows: usize = if (sums) 17 else 16;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = try device.buffer(rows * m.k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x51d_0002);
    for (x.slice(u16, rows * m.k)) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 2 - 1)) >> 16);
    var outs: [2]mtl.Buffer = undefined;
    for (&outs) |*o| o.* = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer for (outs) |o| o.deinit();
    const prep = try device.buffer(prepBytes(m.k, rows, false), mtl.ResourceOptions.shared);
    defer prep.deinit();
    const tiles_ = try call(p, m.n, m.k, rows, false, false, 0);
    const c = try call(p, m.n, m.k, rows, false, true, 0);
    const summed = if (sums) callSums(p, c) orelse return false else c;
    const queue = try device.queue();
    defer queue.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    encode(e, tiles_, m, x, 0, outs[0], 0);
    if (sums) {
        encodePrep(e, p, m.k, rows, false, x, 0, prep, 0);
        e.barrier();
        encodeSums(e, summed, m, x, 0, outs[1], 0, prep, 0);
    } else encode(e, c, m, x, 0, outs[1], 0);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("simd_qmm pair check: {s}", .{why});
        return error.SimdCheckFailed;
    }
    return std.mem.eql(u16, outs[0].slice(u16, rows * m.n), outs[1].slice(u16, rows * m.n));
}

/// Whether the scalar twin's 1-2 row results equal the MMA tiles' for `m`; where not, every width takes tiles.
pub fn twinExact(p: *const Pipelines, device: mtl.Device, m: Matrix) !bool {
    const rows = 8;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const x = try device.buffer(rows * m.k * 2, mtl.ResourceOptions.shared);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x51d_0001);
    for (x.slice(u16, rows * m.k)) |*v| v.* = @truncate(@as(u32, @bitCast(prng.random().float(f32) * 2 - 1)) >> 16);
    const tiled = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer tiled.deinit();
    const twin = try device.buffer(rows * m.n * 2, mtl.ResourceOptions.shared);
    defer twin.deinit();
    const twos = try device.buffer(4 * m.n * 2, mtl.ResourceOptions.shared);
    defer twos.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    encode(e, try call(p, m.n, m.k, rows, false, false, 0), m, x, 0, tiled, 0);
    for (0..rows) |r| encode(e, try call(p, m.n, m.k, 1, true, false, 0), m, x, r * m.k * 2, twin, r * m.n * 2);
    for ([_]usize{ 0, rows - 2 }, 0..) |r, i| encode(e, try call(p, m.n, m.k, 2, true, false, 0), m, x, r * m.k * 2, twos, i * 2 * m.n * 2);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("simd_qmm check: {s}", .{why});
        return error.SimdCheckFailed;
    }
    const all = tiled.slice(u16, rows * m.n);
    if (!std.mem.eql(u16, all, twin.slice(u16, rows * m.n))) return false;
    const two = twos.slice(u16, 4 * m.n);
    return std.mem.eql(u16, all[0 .. 2 * m.n], two[0 .. 2 * m.n]) and std.mem.eql(u16, all[(rows - 2) * m.n ..], two[2 * m.n ..]);
}

test "chunks follow output width only and tiles fit the reduction buffer" {
    try std.testing.expectEqual(@as(u32, 32), splits(48));
    try std.testing.expectEqual(@as(u32, 16), splits(5120));
    try std.testing.expectEqual(@as(u32, 8), splits(34816));
    try std.testing.expectEqual(splits(6240), chunks(3168, 6240));
    try std.testing.expectEqual(splits(6240), chunks(6240, 0));
    // a pair's q and q | k | v key their checkpoint width: 12,288 and 10,240 over 6,144 and 5,120 a rank's.
    try std.testing.expectEqual(@as(u32, 8), chunks(6144, 12288));
    try std.testing.expectEqual(@as(u32, 8), chunks(5120, 10240));
    // the rank's window alone would take 16 chunks, the split this key keeps a pair from forking.
    try std.testing.expectEqual(@as(u32, 16), chunks(6144, 0));
    try std.testing.expectEqual(@as(u32, 16), chunks(5120, 0));
    for (s_values) |s| for ([_]u32{ 1, 2 }) |rt| {
        const nt = tiles(34816, rt, s);
        try std.testing.expect(s * rt * nt * 64 * 4 <= 16384);
        try std.testing.expect(std.mem.indexOfScalar(u32, &nt_values, nt) != null);
    };
    try std.testing.expectEqual(@as(u32, 1), tiles(40, 1, 8));
    try std.testing.expectEqual(@as(u32, 2), tiles(48, 1, 8));
}

test "the scalar twin stages its groups within 20 KB at one and two rows only" {
    for (s_values) |s| {
        try std.testing.expectEqual(@as(u32, 32), scalarBlock(1, s));
        try std.testing.expectEqual(@max(s, 16), scalarBlock(2, s));
        try std.testing.expectEqual(@as(u32, 0), scalarBlock(3, s));
        try std.testing.expectEqual(@as(u32, 0), scalarBlock(0, s));
    }
}
