//! The paged prompt tile's exact output bits on fixed integer-derived inputs, pinned by a digest per architecture.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const kernels = @import("../kernels.zig");

const page = 64;
const heads = 24;
const kv_heads = 4;
const d = 256;

/// The digest of every case's output on an architecture whose bits were recorded.
const pins = [_]struct { arch: []const u8, digest: u64 }{
    .{ .arch = "gfx1151", .digest = 0x30a7911a6c016c2b },
};

/// A prompt chunk: `qlen` queries from position `q_pos0`.
const Case = struct { qlen: usize, q_pos0: usize };

/// Partial last pages, a short last row block, diagonal tiles and, from a later chunk, whole tiles before them.
const cases = [_]Case{
    .{ .qlen = 200, .q_pos0 = 0 },
    .{ .qlen = 150, .q_pos0 = 320 },
    .{ .qlen = 96, .q_pos0 = 100 },
};

fn mix(x: u32) u32 {
    var h = x;
    h ^= h >> 16;
    h *%= 0x7feb352d;
    h ^= h >> 15;
    h *%= 0x846ca68b;
    h ^= h >> 16;
    return h;
}

/// A bf16 of any sign and mantissa with magnitude in [2^-4, 2^2).
fn cacheBits(i: usize, salt: u32) u16 {
    const h = mix(@as(u32, @truncate(i)) *% 0x9E3779B1 +% salt);
    const sign: u16 = @intCast(h >> 31);
    const exp: u16 = @intCast(123 + (h >> 8) % 6);
    return sign << 15 | exp << 7 | @as(u16, @intCast(h & 0x7f));
}

/// An fp32 query element with a full mantissa, so its rounding to the cache type matters.
fn queryValue(i: usize) f32 {
    const h = mix(@as(u32, @truncate(i)) *% 0x85EBCA77 +% 0x51);
    const sign: u32 = h >> 31;
    const exp: u32 = 123 + (h >> 23) % 6;
    const mant: u32 = mix(h) & 0x7fffff;
    return @bitCast(sign << 31 | exp << 23 | mant);
}

/// FNV-1a over `bytes`, continuing from `h`.
fn fnv(h: u64, bytes: []const u8) u64 {
    var x = h;
    for (bytes) |b| {
        x ^= b;
        x *%= 0x100000001b3;
    }
    return x;
}

/// One case's output through the paged tile, folded into `h`; slots past the span hold NaN so a read of them shows.
fn run(t: *Rig, c: Case, h: u64) !u64 {
    const span = c.q_pos0 + c.qlen;
    const count = (span + page - 1) / page;
    const pool = kv_heads * count * page * d;
    const k = try gpa.alloc(u16, pool);
    defer gpa.free(k);
    const v = try gpa.alloc(u16, pool);
    defer gpa.free(v);
    const table = try gpa.alloc(u32, count);
    defer gpa.free(table);
    // a permutation, so a walk that ignored the table would read the wrong keys
    for (table, 0..) |*p, i| p.* = @intCast((i * 5 + 3) % count);
    @memset(k, 0x7FC0);
    @memset(v, 0x7FC0);
    for (0..kv_heads) |kh| for (0..span) |pos| {
        const to = ((kh * count + table[pos / page]) * page + pos % page) * d;
        for (0..d) |j| {
            const i = (kh * span + pos) * d + j;
            k[to + j] = cacheBits(i, 0x4B);
            v[to + j] = cacheBits(i, 0x56);
        }
    };
    const hq = try gpa.alloc(f32, heads * c.qlen * d);
    defer gpa.free(hq);
    for (hq, 0..) |*x, i| x.* = queryValue(i + c.q_pos0 * heads * d);
    var q = try t.upload(hq);
    defer q.free();
    var pool_k = try t.upload(k);
    defer pool_k.free();
    var pool_v = try t.upload(v);
    defer pool_v.free();
    var dev_table = try t.upload(table);
    defer dev_table.free();
    var out = try t.alloc(heads * c.qlen * d * 4);
    defer out.free();
    const scale: f32 = 1 / @sqrt(@as(f32, d));
    try t.on.pagedCausal(at(q), at(pool_k), at(pool_v), at(dev_table), at(out), c.qlen, span, heads, kv_heads, d, scale, c.q_pos0, count, 1, t.stream.handle);
    try t.stream.synchronize();
    const got = try rig.download(f32, out);
    defer gpa.free(got);
    for (got) |x| try std.testing.expect(std.math.isFinite(x));
    return fnv(h, std.mem.sliceAsBytes(got));
}

test "the paged prompt tile writes its pinned bits for 24 heads over 4 kv heads at 256 dims, partial pages included" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16) return error.SkipZigTest;
    const want = for (pins) |p| {
        if (std.mem.eql(u8, p.arch, kernels.arch)) break p.digest;
    } else return error.SkipZigTest;
    var h: u64 = 0xcbf29ce484222325;
    for (cases) |c| h = try run(t, c, h);
    if (h != want) std.debug.print("paged tile digest on {s}: 0x{x:0>16}\n", .{ kernels.arch, h });
    try std.testing.expectEqual(want, h);
}
