//! x W^T for affine-quantized W on the tensor units, dense or by expert, at the row kernels' arithmetic (exact codes, fp32 group sums).
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("frags.zig");

/// The weights' bits (4 or 8) and group (32 or 64), and the output's type.
pub const Format = struct { bits: u8 = 4, group: u16 = 64, out_f32: bool = false };

/// The kernel source for `f` (nax.h inlined for this GPU).
pub fn source(device: mtl.Device, a: std.mem.Allocator, f: Format) ![]u8 {
    if ((f.bits != 4 and f.bits != 8) or (f.group != 32 and f.group != 64)) return error.UnsupportedFormat;
    const body = try frags.source(device, a, ks.core_affine_mm);
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_OUT_T {s}\n{s}", .{ f.bits, f.group, if (f.out_f32) "float" else "bfloat", body });
}

/// The entry points: row sums, the dense matmul, gathers by tile height, and gathers that put each row's output in another order.
pub const names = [6][:0]const u8{ "tf_affine_row_sums", "tf_affine_mm", "tf_affine_gather_32", "tf_affine_gather_64", "tf_affine_gather_scatter_32", "tf_affine_gather_scatter_64" };

/// A format's pipelines, in `names` order.
pub const Pipes = [6]mtl.Pipeline;

pub const Args = extern struct { rows: i32, n: i32, k: i32, experts: i32 };

/// A dense matmul's pitches and batch offsets, in elements (the sums' in rows); the default is one plain [rows, k] x [k, n] product.
pub const Strides = extern struct { x_row: i32 = 0, y_row: i32 = 0, sums_row: i32 = 1, x_batch: i32 = 0, y_batch: i32 = 0, sums_batch: i32 = 0, w_batch: i32 = 0, pad: i32 = 0 };

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// Each row's sum over each group of x [rows, k] into `sums` (fp32 [rows, k / group]): the bias's operand.
pub fn rowSums(e: mtl.ComputeEncoder, p: Pipes, group: u32, x: anytype, sums: anytype, rows: u32, k: u32) void {
    e.setPipeline(p[0]);
    bind(e, 0, x);
    e.setValue(Args{ .rows = @intCast(rows), .n = 0, .k = @intCast(k), .experts = 0 }, 5);
    bind(e, 6, sums);
    e.dispatchThreads(mtl.Size.of(k / group, rows, 1), mtl.Size.of(@min(k / group, 64), 1, 1));
}

/// y [rows, n] = x [rows, q.k] W^T with x's group sums `sums`; n is q.n rounded up to 64 (the padded weight rows, y's pitch).
pub fn dense(e: mtl.ComputeEncoder, p: Pipes, x: anytype, sums: anytype, q: anytype, y: anytype, rows: u32) void {
    const n = std.mem.alignForward(u32, q.n, 64);
    denseBatch(e, p, x, sums, q, y, rows, 1, .{ .x_row = @intCast(q.k), .y_row = @intCast(n) });
}

/// `batch` dense products at once, each over its own rows of the weights (`s`; q.n a multiple of 64), e.g. a per-head projection.
pub fn denseBatch(e: mtl.ComputeEncoder, p: Pipes, x: anytype, sums: anytype, q: anytype, y: anytype, rows: u32, batch: u32, s: Strides) void {
    const n = std.mem.alignForward(u32, q.n, 64);
    e.setPipeline(p[1]);
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    bind(e, 4, sums);
    e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(n), .k = @intCast(q.k), .experts = 0 }, 5);
    bind(e, 6, y);
    e.setValue(s, 7);
    e.dispatchGroups(mtl.Size.of(n / 64, (rows + 63) / 64, batch), mtl.Size.of(32, 2, 2));
}

/// y [rows, q.n] = x times each row's expert's W^T, rows sorted by expert with `offsets` (each expert's first row); q is [experts, n, k].
pub fn gather(e: mtl.ComputeEncoder, p: Pipes, x: anytype, sums: anytype, q: anytype, offsets: anytype, y: anytype, rows: u32, experts: u32) void {
    gatherTo(e, p, x, sums, q, offsets, y, null, rows, experts);
}

/// `gather` with sorted row i's output at row dst[i] of y (i32 [rows]; null: in sorted order).
pub fn gatherTo(e: mtl.ComputeEncoder, p: Pipes, x: anytype, sums: anytype, q: anytype, offsets: anytype, y: anytype, dst: anytype, rows: u32, experts: u32) void {
    std.debug.assert(q.n % 64 == 0 and q.k % 64 == 0);
    const bm: u32 = if (rows / experts < 64) 32 else 64;
    const scatter = @TypeOf(dst) != @TypeOf(null);
    e.setPipeline(p[@as(usize, if (bm == 64) 3 else 2) + if (scatter) 2 else 0]);
    if (scatter) bind(e, 8, dst);
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    bind(e, 4, sums);
    e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(q.n), .k = @intCast(q.k), .experts = @intCast(experts) }, 5);
    bind(e, 6, y);
    bind(e, 7, offsets);
    const tiles = @min(rows, (rows + bm - 1) / bm + experts - 1); // each expert's last tile may be partial
    e.dispatchGroups(mtl.Size.of(q.n / 64, tiles, 1), mtl.Size.of(32, 2, 2));
}

fn bf16(v: f32) u16 { // round to nearest even
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

test "dense and gathered, each format, against the row kernels' arithmetic on the host" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const E = 3;
    const N = 128;
    const K = 256;
    const counts = [E]u32{ 37, 0, 70 }; // an empty expert and partial tiles; densely, 107 rows (a partial tile)
    const R = counts[0] + counts[1] + counts[2];
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for ([_][2]usize{ .{ 4, 64 }, .{ 4, 32 }, .{ 8, 64 }, .{ 8, 32 } }) |fmt| {
        const bits = fmt[0];
        const group = fmt[1];
        const w_bytes = E * N * K * bits / 8;
        const s_n = E * N * K / group;
        const opts = mtl.ResourceOptions.shared;
        const bufs = [_]mtl.Buffer{ try device.buffer(R * K * 2, opts), try device.buffer(w_bytes, opts), try device.buffer(s_n * 2, opts), try device.buffer(s_n * 2, opts), try device.buffer(E * 4, opts), try device.buffer(5 * R * N * 4, opts), try device.buffer(R * (K / group) * 4, opts), try device.buffer(R * 4, opts) };
        const dst = bufs[7].slice(i32, R); // the scatter's destinations: rows reversed
        for (dst, 0..) |*d, i| d.* = @intCast(R - 1 - i);
        defer for (bufs) |b| b.deinit();
        const x = bufs[0].slice(u16, R * K);
        for (x) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
        const wb = bufs[1].slice(u8, w_bytes);
        rnd.bytes(wb);
        const sc = bufs[2].slice(u16, s_n);
        const bi = bufs[3].slice(u16, s_n);
        for (sc, bi) |*s, *b| {
            s.* = bf16((rnd.float(f32) + 0.5) / 64);
            b.* = bf16(-(rnd.float(f32) + 0.5) * 0.15);
        }
        const offs = bufs[4].slice(i32, E);
        var first: i32 = 0;
        for (offs, counts) |*o, c| {
            o.* = first;
            first += @intCast(c);
        }
        var pipes: [2]Pipes = undefined;
        for (0..2) |o| {
            const src = try source(device, a, .{ .bits = @intCast(bits), .group = @intCast(group), .out_f32 = o == 1 });
            defer a.free(src);
            const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
            defer lib.deinit();
            for (names, 0..) |n, i| pipes[o][i] = try mtl.Pipeline.init(device, lib, n, false);
        }
        defer for (pipes) |pp| for (pp) |p| p.deinit();
        const xr = .{ .buf = bufs[0], .off = @as(usize, 0) };
        const sums = .{ .buf = bufs[6], .off = @as(usize, 0) };
        const q = .{ .w = .{ .buf = bufs[1], .off = @as(usize, 0) }, .s = .{ .buf = bufs[2], .off = @as(usize, 0) }, .b = .{ .buf = bufs[3], .off = @as(usize, 0) }, .n = @as(u32, N), .k = @as(u32, K) };
        const out = [5]struct { buf: mtl.Buffer, off: usize }{ .{ .buf = bufs[5], .off = 0 }, .{ .buf = bufs[5], .off = R * N * 4 }, .{ .buf = bufs[5], .off = 2 * R * N * 4 }, .{ .buf = bufs[5], .off = 3 * R * N * 4 }, .{ .buf = bufs[5], .off = 4 * R * N * 4 } };
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        rowSums(enc, pipes[0], @intCast(group), xr, sums, R, K);
        gather(enc, pipes[0], xr, sums, q, .{ .buf = bufs[4], .off = @as(usize, 0) }, out[0], R, E); // bf16
        gather(enc, pipes[1], xr, sums, q, .{ .buf = bufs[4], .off = @as(usize, 0) }, out[1], R, E); // fp32
        dense(enc, pipes[0], xr, sums, q, out[2], R); // expert 0's weights for every row, bf16
        dense(enc, pipes[1], xr, sums, q, out[3], R); // fp32
        gatherTo(enc, pipes[1], xr, sums, q, .{ .buf = bufs[4], .off = @as(usize, 0) }, out[4], .{ .buf = bufs[7], .off = @as(usize, 0) }, R, E); // fp32, rows reversed
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const all = bufs[5].contents();
        const g16: [*]const u16 = @ptrCast(@alignCast(all));
        const g32: [*]const f32 = @ptrCast(@alignCast(all + R * N * 4));
        const d16: [*]const u16 = @ptrCast(@alignCast(all + 2 * R * N * 4));
        const d32: [*]const f32 = @ptrCast(@alignCast(all + 3 * R * N * 4));
        const s32: [*]const f32 = @ptrCast(@alignCast(all + 4 * R * N * 4));
        for (0..R) |i| for (0..N) |n| try std.testing.expectEqual(g32[i * N + n], s32[@as(usize, @intCast(dst[i])) * N + n]);
        var r: usize = 0;
        for (counts, 0..) |c, ex| for (0..c) |_| {
            for ([2]usize{ ex, 0 }, 0..) |use, pass| for (0..N) |n| {
                const row = use * N + n;
                var sum: f64 = 0;
                var mag: f64 = 0;
                for (0..K / group) |g| {
                    var dot: f64 = 0;
                    var dmag: f64 = 0;
                    var xs: f32 = 0;
                    var i: usize = 0;
                    while (i < group) : (i += if (bits == 4) 4 else 1) {
                        const at = r * K + g * group + i;
                        if (bits == 4) {
                            const chain = f32of(bf16(f32of(bf16(f32of(bf16(f32of(x[at]) + f32of(x[at + 1]))) + f32of(x[at + 2]))) + f32of(x[at + 3])));
                            xs += chain;
                        } else xs += f32of(x[at]);
                    }
                    for (0..group) |j| {
                        const k = g * group + j;
                        const qv: u32 = if (bits == 4) (wb[(row * K + k) / 2] >> @intCast(4 * (k % 2))) & 15 else wb[row * K + k];
                        dot += @as(f64, @floatFromInt(qv)) * f32of(x[r * K + k]);
                        dmag += @abs(@as(f64, @floatFromInt(qv)) * f32of(x[r * K + k]));
                    }
                    const sg = row * (K / group) + g;
                    const term = @as(f64, f32of(sc[sg])) * dot + @as(f64, f32of(bi[sg])) * xs;
                    sum += term;
                    mag += @as(f64, f32of(sc[sg])) * dmag + @abs(@as(f64, f32of(bi[sg])) * xs);
                }
                const got = if (pass == 0) g32[r * N + n] else d32[r * N + n];
                const got16 = if (pass == 0) g16[r * N + n] else d16[r * N + n];
                try std.testing.expect(@abs(got - sum) <= 4e-6 * mag + 1e-30); // fp32 order only
                try std.testing.expectEqual(bf16(got), got16);
            };
            r += 1;
        };
    }
}

test "a batch of dense products with pitches: each batch's rows of x, the weights and y" {
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    const B = 3; // batches, interleaved in x and y like heads in a row: x [R, B, K], y [R, B, N]
    const N = 64;
    const K = 128;
    const R = 70;
    var prng = std.Random.DefaultPrng.init(5);
    const rnd = prng.random();
    const opts = mtl.ResourceOptions.shared;
    const bufs = [_]mtl.Buffer{ try device.buffer(R * B * K * 2, opts), try device.buffer(B * N * K / 2, opts), try device.buffer(B * N * K / 64 * 2, opts), try device.buffer(B * N * K / 64 * 2, opts), try device.buffer(R * B * N * 2, opts), try device.buffer(R * B * (K / 64) * 4, opts) };
    defer for (bufs) |b| b.deinit();
    const x = bufs[0].slice(u16, R * B * K);
    for (x) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
    const wb = bufs[1].slice(u8, B * N * K / 2);
    rnd.bytes(wb);
    const sc = bufs[2].slice(u16, B * N * K / 64);
    const bi = bufs[3].slice(u16, B * N * K / 64);
    for (sc, bi) |*s, *b| {
        s.* = bf16((rnd.float(f32) + 0.5) / 64);
        b.* = bf16(-(rnd.float(f32) + 0.5) * 0.15);
    }
    const src = try source(device, a, .{});
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var pipes: Pipes = undefined;
    for (names, 0..) |n, i| pipes[i] = try mtl.Pipeline.init(device, lib, n, false);
    defer for (pipes) |p| p.deinit();
    const xr = .{ .buf = bufs[0], .off = @as(usize, 0) };
    const sums = .{ .buf = bufs[5], .off = @as(usize, 0) };
    const q = .{ .w = .{ .buf = bufs[1], .off = @as(usize, 0) }, .s = .{ .buf = bufs[2], .off = @as(usize, 0) }, .b = .{ .buf = bufs[3], .off = @as(usize, 0) }, .n = @as(u32, N), .k = @as(u32, K) };
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    rowSums(enc, pipes, 64, xr, sums, R * B, K); // x as [R * B, K]: batch b's row r is row r * B + b
    denseBatch(enc, pipes, xr, sums, q, .{ .buf = bufs[4], .off = @as(usize, 0) }, R, B, .{ .x_row = B * K, .y_row = B * N, .sums_row = B, .x_batch = K, .y_batch = N, .sums_batch = 1, .w_batch = N });
    enc.end();
    cb.commit();
    cb.wait();
    try std.testing.expect(cb.failure() == null);
    const y = bufs[4].slice(u16, R * B * N);
    for (0..R) |r| for (0..B) |b| for (0..N) |n| {
        const row = b * N + n;
        var sum: f64 = 0;
        var mag: f64 = 0;
        for (0..K / 64) |g| {
            var dot: f64 = 0;
            var xs: f32 = 0;
            var i: usize = 0;
            const xrow = (r * B + b) * K;
            while (i < 64) : (i += 4) {
                const at = xrow + g * 64 + i;
                xs += f32of(bf16(f32of(bf16(f32of(bf16(f32of(x[at]) + f32of(x[at + 1]))) + f32of(x[at + 2]))) + f32of(x[at + 3])));
            }
            for (0..64) |j| {
                const k = g * 64 + j;
                const qv: u32 = (wb[(row * K + k) / 2] >> @intCast(4 * (k % 2))) & 15;
                dot += @as(f64, @floatFromInt(qv)) * f32of(x[xrow + k]);
            }
            const sg = row * (K / 64) + g;
            sum += @as(f64, f32of(sc[sg])) * dot + @as(f64, f32of(bi[sg])) * xs;
            mag += @abs(@as(f64, f32of(sc[sg])) * dot) + @abs(@as(f64, f32of(bi[sg])) * xs);
        }
        const got: f64 = f32of(y[(r * B + b) * N + n]);
        try std.testing.expect(@abs(got - sum) <= @abs(sum) / 128.0 + 4e-6 * mag); // one bf16 rounding of the fp32 sum
    };
}
