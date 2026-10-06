//! Rows sorted by expert times their expert's quantized W^T on the tensor units (kernels/metal/core/expert_gather.metal):
//! MLX-layout affine weights at a format's bits and group size, bf16 out or fp32 partial sums for another pass to add.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("frags.zig");

/// The weights' bits (4 or 8) and group (32 or 64), and the output's type.
pub const Format = struct { bits: u8 = 4, group: u16 = 64, out_f32: bool = false };

/// The kernel source for `f` (nax.h inlined for this GPU).
pub fn source(device: mtl.Device, a: std.mem.Allocator, f: Format) ![]u8 {
    if ((f.bits != 4 and f.bits != 8) or (f.group != 32 and f.group != 64)) return error.UnsupportedFormat;
    const body = try frags.source(device, a, ks.core_expert_gather);
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_OUT_T {s}\n{s}", .{ f.bits, f.group, if (f.out_f32) "float" else "bfloat", body });
}

/// The two tile heights' entry points: 32 rows (few rows an expert) and 64.
pub const names = [2][:0]const u8{ "tf_expert_gather_32", "tf_expert_gather_64" };

pub const Args = extern struct { rows: i32, n: i32, k: i32, experts: i32 };

fn bind(e: mtl.ComputeEncoder, i: usize, r: anytype) void {
    e.setBuffer(r.buf, r.off, i);
}

/// y [rows, q.n] = x [rows, q.k] times each row's expert's W^T, the rows sorted by expert with `offsets` [experts]
/// (the first row of each expert); `q` is [experts, n, k] (w, s, b, n, k) and n a multiple of 64.
pub fn run(e: mtl.ComputeEncoder, pipes: [2]mtl.Pipeline, x: anytype, q: anytype, offsets: anytype, y: anytype, rows: u32, experts: u32) void {
    std.debug.assert(q.n % 64 == 0 and q.k % 64 == 0);
    const bm: u32 = if (rows / experts < 64) 32 else 64;
    e.setPipeline(pipes[@intFromBool(bm == 64)]);
    bind(e, 0, x);
    bind(e, 1, q.w);
    bind(e, 2, q.s);
    bind(e, 3, q.b);
    bind(e, 4, offsets);
    e.setValue(Args{ .rows = @intCast(rows), .n = @intCast(q.n), .k = @intCast(q.k), .experts = @intCast(experts) }, 5);
    bind(e, 6, y);
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

test "each format against the host's sums; fp32 out rounds to bf16 out bit for bit" {
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
    const counts = [E]u32{ 37, 0, 70 }; // an empty expert and partial tiles
    const R = counts[0] + counts[1] + counts[2];
    var prng = std.Random.DefaultPrng.init(7);
    const rnd = prng.random();
    for ([_][2]u16{ .{ 4, 64 }, .{ 4, 32 }, .{ 8, 64 }, .{ 8, 32 } }) |fmt| {
        const bits: usize = fmt[0];
        const group: usize = fmt[1];
        const w_bytes = E * N * K * bits / 8;
        const s_n = E * N * K / group;
        const opts = mtl.ResourceOptions.shared;
        const bufs = [_]mtl.Buffer{ try device.buffer(R * K * 2, opts), try device.buffer(w_bytes, opts), try device.buffer(s_n * 2, opts), try device.buffer(s_n * 2, opts), try device.buffer(E * 4, opts), try device.buffer(R * N * 2, opts), try device.buffer(R * N * 4, opts) };
        defer for (bufs) |b| b.deinit();
        const x = bufs[0].slice(u16, R * K);
        for (x) |*v| v.* = bf16(rnd.float(f32) * 2 - 1);
        const wb = bufs[1].slice(u8, w_bytes);
        rnd.bytes(wb);
        const sc = bufs[2].slice(u16, s_n);
        const bi = bufs[3].slice(u16, s_n);
        for (sc, bi) |*s, *b| {
            s.* = bf16((rnd.float(f32) + 0.5) / 64);
            b.* = bf16(rnd.float(f32) * 0.2 - 0.1);
        }
        const offs = bufs[4].slice(i32, E);
        var first: i32 = 0;
        for (offs, counts) |*o, c| {
            o.* = first;
            first += @intCast(c);
        }
        var pipes: [2][2]mtl.Pipeline = undefined;
        for (0..2) |o| {
            const src = try source(device, a, .{ .bits = @intCast(bits), .group = @intCast(group), .out_f32 = o == 1 });
            defer a.free(src);
            const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
            defer lib.deinit();
            for (names, 0..) |n, i| pipes[o][i] = try mtl.Pipeline.init(device, lib, n, false);
        }
        defer for (pipes) |pp| for (pp) |p| p.deinit();
        const q = .{ .w = .{ .buf = bufs[1], .off = 0 }, .s = .{ .buf = bufs[2], .off = 0 }, .b = .{ .buf = bufs[3], .off = 0 }, .n = @as(u32, N), .k = @as(u32, K) };
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        run(enc, pipes[0], .{ .buf = bufs[0], .off = 0 }, q, .{ .buf = bufs[4], .off = 0 }, .{ .buf = bufs[5], .off = 0 }, R, E);
        run(enc, pipes[1], .{ .buf = bufs[0], .off = 0 }, q, .{ .buf = bufs[4], .off = 0 }, .{ .buf = bufs[6], .off = 0 }, R, E);
        enc.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        const y16 = bufs[5].slice(u16, R * N);
        const y32 = bufs[6].slice(f32, R * N);
        var r: usize = 0;
        for (counts, 0..) |c, ex| for (0..c) |_| {
            for (0..N) |n| {
                const row = ex * N + n;
                var sum: f64 = 0;
                var mag: f64 = 0;
                for (0..K) |k| {
                    const g = (row * K + k) / group;
                    const qv: u32 = if (bits == 4) (wb[(row * K + k) / 2] >> @intCast(4 * (k % 2))) & 15 else wb[row * K + k];
                    const wv = f32of(bf16(f32of(sc[g]) * @as(f32, @floatFromInt(qv)) + f32of(bi[g])));
                    const p = @as(f64, wv) * f32of(x[r * K + k]);
                    sum += p;
                    mag += @abs(p);
                }
                const got = y32[r * N + n];
                try std.testing.expect(@abs(got - sum) <= 1e-4 * mag + 1e-30); // the order of the sums, a weight's bf16 tie
                try std.testing.expectEqual(bf16(got), y16[r * N + n]);
            }
            r += 1;
        };
    }
}
