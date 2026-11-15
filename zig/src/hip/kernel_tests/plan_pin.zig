//! The planned decode walk's exact output bits on NaN- and inf-poisoned keys and queries, pinned by a digest per arch.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const kernels = @import("../kernels.zig");
const plan = @import("../launches/plan.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const page = 64;
const d = 256;

/// The digest of every case's output on an architecture whose bits were recorded.
const pins = [_]struct { arch: []const u8, digest: u64 }{
    .{ .arch = "gfx1151", .digest = 0x0adb33378a143b95 },
};

/// A round of rows at `pos` over one slot's `span` keys, q fp32 or, with `act`, in the cache's kind.
const Case = struct { heads: usize, kv_heads: usize, span: usize, pos: []const i32, act: bool };

/// The long-span score (four keys a warp) and the short one (two), each with q fp32 and in the cache's kind.
const cases = [_]Case{
    .{ .heads = 16, .kv_heads = 2, .span = 16384, .pos = &.{ 16383, 9001 }, .act = false },
    .{ .heads = 16, .kv_heads = 2, .span = 16384, .pos = &.{ 16383, 9001 }, .act = true },
    .{ .heads = 24, .kv_heads = 4, .span = 1024, .pos = &.{ 1023, 300, 37 }, .act = false },
    .{ .heads = 24, .kv_heads = 4, .span = 1024, .pos = &.{ 1023, 300, 37 }, .act = true },
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

/// A finite bf16 of any sign and mantissa with magnitude in [2^-4, 2^2).
fn finite(h: u32) u16 {
    const sign: u16 = @intCast(h >> 31);
    const exp: u16 = @intCast(123 + (h >> 8) % 6);
    return sign << 15 | exp << 7 | @as(u16, @intCast(h & 0x7f));
}

/// Element e (lane e % 32) of a key hashed h: payload NaNs, +inf and -inf, or signaling NaNs, each one key in ten.
fn keyBits(h: u32, e: usize) u16 {
    const g = mix(h ^ @as(u32, @intCast(e)) *% 0x9E3779B1);
    const lane = e % 32;
    const sign: u16 = @intCast((g >> 31) << 15);
    return switch (h % 10) {
        0 => sign | 0x7FC0 | @as(u16, @intCast(g & 0x3F)),
        1 => if (e < 32) (if (lane < 16) @as(u16, 0x7F80) else 0xFF80) else finite(g),
        2 => if (lane % 2 == 0) sign | 0x7F80 | @as(u16, @intCast(1 + (g >> 8) % 0x3F)) else finite(g),
        else => finite(g),
    };
}

/// Element e of a (row, head) query hashed h, as an fp32 pattern: quiet NaNs of distinct payloads in one of four.
fn queryBits(h: u32, e: usize) u32 {
    const g = mix(h ^ @as(u32, @intCast(e)) *% 0x85EBCA77);
    if (h % 4 == 0) return (g >> 31) << 31 | 0x7FC00000 | (mix(g) & 0x3FFFFF);
    const exp: u32 = 123 + (g >> 23) % 6;
    return (g >> 31) << 31 | exp << 23 | (mix(g) & 0x7fffff);
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

/// A device buffer of `bytes`, every byte 0xff, so a word the walk does not write shows in the digest.
fn poisoned(t: *Rig, bytes: usize) !DeviceBuffer {
    var b = try t.alloc(bytes);
    errdefer b.free();
    try b.fill8(0xff);
    return b;
}

/// One case's scores, stats, partials and output folded into `h`; `patterns` gathers the scores' NaN bit patterns.
fn run(t: *Rig, c: Case, h: u64, patterns: *std.AutoHashMap(u32, void)) !u64 {
    const count = c.span / page;
    const rows = c.pos.len;
    const k = try gpa.alloc(u16, c.kv_heads * c.span * d);
    defer gpa.free(k);
    const v = try gpa.alloc(u16, k.len);
    defer gpa.free(v);
    const table = try gpa.alloc(u32, count);
    defer gpa.free(table);
    // a permutation, so a walk that ignored the table would read the wrong keys
    for (table, 0..) |*p, i| p.* = @intCast((i * 5 + 3) % count);
    for (0..c.kv_heads) |kh| for (0..c.span) |pos| {
        const to = ((kh * count + table[pos / page]) * page + pos % page) * d;
        const hk = mix(@as(u32, @intCast(kh * c.span + pos)) *% 0x27D4EB2F +% 0x4B);
        for (0..d) |e| {
            k[to + e] = keyBits(hk, e);
            v[to + e] = finite(mix(hk ^ @as(u32, @intCast(e)) +% 0x56));
        }
    };
    const q32 = try gpa.alloc(u32, rows * c.heads * d);
    defer gpa.free(q32);
    for (q32, 0..) |*x, i| x.* = queryBits(mix(@as(u32, @intCast(i / d)) +% 0x51), i % d);
    // q in the cache's kind: the upper half of each pattern, a quiet NaN keeping six payload bits
    const q16 = try gpa.alloc(u16, q32.len);
    defer gpa.free(q16);
    for (q16, q32) |*y, x| y.* = if (x & 0x7FC00000 == 0x7FC00000) @intCast(((x >> 16) & 0xFFC0) | (x & 0x3F)) else @intCast(x >> 16);
    var q = if (c.act) try t.upload(q16) else try t.upload(q32);
    defer q.free();
    var pool_k = try t.upload(k);
    defer pool_k.free();
    var pool_v = try t.upload(v);
    defer pool_v.free();
    // one slot: descriptor [cached, kept, k pool, v pool], then its page table
    const words = 4;
    const desc = try gpa.alloc(u64, words + (count + 1) / 2);
    defer gpa.free(desc);
    @memset(desc, 0);
    desc[2] = at(pool_k);
    desc[3] = at(pool_v);
    @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. count * 4], std.mem.sliceAsBytes(table));
    var dev_desc = try t.upload(desc);
    defer dev_desc.free();
    const desc_ptr = [_]u64{at(dev_desc)};
    var descs = try t.upload(&desc_ptr);
    defer descs.free();
    var dev_pos = try t.upload(c.pos);
    defer dev_pos.free();
    const zeros = [_]i32{ 0, 0, 0, 0 };
    var slot = try t.upload(zeros[0..rows]);
    defer slot.free();
    var dev_first = try t.upload(zeros[0..1]);
    defer dev_first.free();
    const n = [_]i32{@intCast(rows)};
    var dev_count = try t.upload(&n);
    defer dev_count.free();
    var scores = try poisoned(t, rows * c.heads * c.span * 4);
    defer scores.free();
    var stats = try poisoned(t, rows * c.heads * 2 * 4);
    defer stats.free();
    var partials = try poisoned(t, rows * c.heads * (c.span / 128) * d * 4);
    defer partials.free();
    var out = try poisoned(t, rows * c.heads * d * @as(usize, if (c.act) 2 else 4));
    defer out.free();
    const p: plan.PlanRef = .{ .rows = rows, .slots = 1, .args = .{ .pos = at(dev_pos), .slot = at(slot), .first = at(dev_first), .count = at(dev_count), .desc = at(descs), .pages = words, .pool = @intCast(count) } };
    const scale: f32 = 1 / @sqrt(@as(f32, d));
    try t.on.planCausalAs(at(q), at(out), at(scores), at(stats), at(partials), p, 0, c.heads, c.kv_heads, d, c.span, scale, t.kind(), c.act, t.stream.handle);
    try t.stream.synchronize();
    var x = h;
    for ([_]DeviceBuffer{ scores, stats, partials, out }, 0..) |buf, i| {
        const got = try rig.download(u32, buf);
        defer gpa.free(got);
        if (i == 0) for (got) |w| if (w & 0x7F800000 == 0x7F800000 and w & 0x7FFFFF != 0) try patterns.put(w, {});
        x = fnv(x, std.mem.sliceAsBytes(got));
    }
    return x;
}

test "the planned decode walk writes its pinned bits on NaN- and inf-poisoned keys and queries, long and short spans" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    if (!t.bf16) return error.SkipZigTest;
    const want = for (pins) |p| {
        if (std.mem.eql(u8, p.arch, kernels.arch)) break p.digest;
    } else return error.SkipZigTest;
    var patterns = std.AutoHashMap(u32, void).init(gpa);
    defer patterns.deinit();
    var h: u64 = 0xcbf29ce484222325;
    for (cases) |c| h = try run(t, c, h, &patterns);
    // NaNs of many payloads reach the scores, so the digest shows which payload each fold keeps
    try std.testing.expect(patterns.count() >= 64);
    if (h != want) std.debug.print("planned walk NaN digest on {s}: 0x{x:0>16} ({d} NaN patterns in the scores)\n", .{ kernels.arch, h, patterns.count() });
    try std.testing.expectEqual(want, h);
}
