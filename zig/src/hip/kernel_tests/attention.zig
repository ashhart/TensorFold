//! Attention: planned decode over pages and the paged prompt tile against their flat forms (same bits) and float64.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const plan = @import("../launches/plan.zig");
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

const page = 64;

/// A cache of `kv_heads` heads and `span` positions in the activation type, flat (kv_heads, span, d) and paged.
const Cache = struct {
    k: []u16,
    v: []u16,
    /// Position block i lives in page table[i].
    table: []u32,

    fn make(t: *Rig, rng: *rig.Rng, kv_heads: usize, span: usize, d: usize) !Cache {
        const c: Cache = .{ .k = try gpa.alloc(u16, kv_heads * span * d), .v = try gpa.alloc(u16, kv_heads * span * d), .table = try gpa.alloc(u32, span / page) };
        for (c.k) |*x| x.* = t.bits(rng.unit());
        for (c.v) |*x| x.* = t.bits(rng.unit());
        // a permutation, so a walk that ignored the table would read the wrong keys
        for (c.table, 0..) |*p, i| p.* = @intCast((i * 5 + 3) % c.table.len);
        return c;
    }

    fn free(c: Cache) void {
        gpa.free(c.k);
        gpa.free(c.v);
        gpa.free(c.table);
    }

    /// The pool layout of decode/pages.hpp: ((h * count + page) * 64 + slot) * d.
    fn pool(c: Cache, src: []const u16, kv_heads: usize, span: usize, d: usize) ![]u16 {
        const out = try gpa.alloc(u16, src.len);
        const count = span / page;
        for (0..kv_heads) |h| for (0..span) |pos| {
            const to = ((h * count + c.table[pos / page]) * page + pos % page) * d;
            @memcpy(out[to..][0..d], src[(h * span + pos) * d ..][0..d]);
        };
        return out;
    }
};

/// softmax(q . k * scale) v in float64 for one query of `head` seeing positions [0, visible).
fn reference(t: *const Rig, c: Cache, q: []const f32, head: usize, group: usize, span: usize, d: usize, visible: usize, scale: f64, out: []f64) void {
    const kv = head / group;
    var top: f64 = -std.math.inf(f64);
    var scores: [4096]f64 = undefined;
    for (0..visible) |key| {
        var dot: f64 = 0;
        for (0..d) |j| dot += @as(f64, q[j]) * t.value(c.k[(kv * span + key) * d + j]);
        scores[key] = dot * scale;
        top = @max(top, scores[key]);
    }
    var sum: f64 = 0;
    for (scores[0..visible]) |*s| {
        s.* = @exp(s.* - top);
        sum += s.*;
    }
    @memset(out[0..d], 0);
    for (0..visible) |key| for (0..d) |j| {
        out[j] += scores[key] / sum * t.value(c.v[(kv * span + key) * d + j]);
    };
}

fn worst(got: []const f32, want: []const f64) f64 {
    var scale: f64 = 0;
    var max: f64 = 0;
    for (got, want) |g, w| {
        max = @max(max, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return max / scale;
}

/// Rows at positions `pos` over one slot's cache: the planned walk (grouped heads, pages) and the flat walk agree.
fn decode(t: *Rig, rng: *rig.Rng, heads: usize, kv_heads: usize, d: usize, pos: []const i32) !void {
    const span: usize = 1024;
    const rows = pos.len;
    const group = heads / kv_heads;
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(d)));
    const c = try Cache.make(t, rng, kv_heads, span, d);
    defer c.free();
    const hq = try gpa.alloc(f32, rows * heads * d);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit();
    var q = try t.upload(hq);
    defer q.free();
    var k = try t.upload(c.k);
    defer k.free();
    var v = try t.upload(c.v);
    defer v.free();
    const pk = try c.pool(c.k, kv_heads, span, d);
    defer gpa.free(pk);
    const pv = try c.pool(c.v, kv_heads, span, d);
    defer gpa.free(pv);
    var pool_k = try t.upload(pk);
    defer pool_k.free();
    var pool_v = try t.upload(pv);
    defer pool_v.free();
    // one slot: descriptor [cached, kept, k pool, v pool], then its page table
    const words = 4;
    const desc = try gpa.alloc(u64, words + (c.table.len + 1) / 2);
    defer gpa.free(desc);
    @memset(desc, 0);
    desc[2] = at(pool_k);
    desc[3] = at(pool_v);
    @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. c.table.len * 4], std.mem.sliceAsBytes(c.table));
    var dev_desc = try t.upload(desc);
    defer dev_desc.free();
    const desc_ptr = [_]u64{at(dev_desc)};
    var descs = try t.upload(&desc_ptr);
    defer descs.free();
    var dev_pos = try t.upload(pos);
    defer dev_pos.free();
    const zeros = try gpa.alloc(i32, rows);
    defer gpa.free(zeros);
    @memset(zeros, 0);
    var slot = try t.upload(zeros);
    defer slot.free();
    const first = [_]i32{0};
    const count = [_]i32{@intCast(rows)};
    var dev_first = try t.upload(&first);
    defer dev_first.free();
    var dev_count = try t.upload(&count);
    defer dev_count.free();
    const tiles = span / 128;
    var scores = try t.alloc(rows * heads * span * 4);
    defer scores.free();
    var stats = try t.alloc(rows * heads * 2 * 4);
    defer stats.free();
    var partials = try t.alloc(rows * heads * tiles * d * 4);
    defer partials.free();
    var outs: [2]DeviceBuffer = .{ try t.alloc(rows * heads * d * 4), try t.alloc(rows * heads * d * 4) };
    defer for (&outs) |*o| o.free();
    const s = t.stream.handle;
    try t.on.tf_causal(@ptrFromInt(at(q)), @ptrFromInt(at(k)), @ptrFromInt(at(v)), @ptrFromInt(at(outs[0])), @intCast(rows), 1, @intCast(span), @intCast(heads), @intCast(kv_heads), @intCast(d), scale, 0, 0, @intCast(span * d), @intCast(d), 0, @intCast(span * d), @intCast(d), if (t.bf16) 1 else 0, @ptrFromInt(at(scores)), @ptrFromInt(at(stats)), @ptrFromInt(at(partials)), s, @ptrFromInt(at(dev_pos)));
    const p: plan.PlanRef = .{ .rows = rows, .slots = 1, .args = .{ .pos = at(dev_pos), .slot = at(slot), .first = at(dev_first), .count = at(dev_count), .desc = at(descs), .pages = words, .pool = @intCast(c.table.len) } };
    try t.on.planCausal(at(q), at(outs[1]), at(scores), at(stats), at(partials), p, 0, heads, kv_heads, d, span, scale, t.kind(), s);
    try t.stream.synchronize();
    const flat = try rig.download(f32, outs[0]);
    defer gpa.free(flat);
    const planned = try rig.download(f32, outs[1]);
    defer gpa.free(planned);
    try std.testing.expectEqualSlices(u32, @ptrCast(flat), @ptrCast(planned));
    const want = try gpa.alloc(f64, rows * heads * d);
    defer gpa.free(want);
    for (0..rows) |r| for (0..heads) |h| {
        const at_q = (r * heads + h) * d;
        reference(t, c, hq[at_q..][0..d], h, group, span, d, @intCast(pos[r] + 1), scale, want[at_q..][0..d]);
    };
    try std.testing.expect(worst(planned, want) < 1e-5);
}

test "the planned decode walk over pages writes the flat walk's bits for grouped heads, within 1e-5 of float64" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xA77E57 };
    const pos = [_]i32{ 0, 37, 300, 1023 };
    try decode(t, &rng, 16, 2, 256, &pos); // 8 query heads a KV head, Qwen3.6-35B-A3B's attention
    try decode(t, &rng, 24, 4, 256, &pos); // 6, Qwen3.8-27B
    try decode(t, &rng, 16, 4, 128, &pos); // 4
    try decode(t, &rng, 4, 4, 128, &pos); // no sharing
}

/// A prompt of `qlen` queries from `q_pos0` over `span` positions: the paged 64-row tile writes the flat tile's bits.
fn prompt(t: *Rig, rng: *rig.Rng, heads: usize, kv_heads: usize, d: usize, qlen: usize, q_pos0: usize) !void {
    const span = q_pos0 + qlen;
    const group = heads / kv_heads;
    const scale: f32 = 1 / @sqrt(@as(f32, @floatFromInt(d)));
    const c = try Cache.make(t, rng, kv_heads, span, d);
    defer c.free();
    const hq = try gpa.alloc(f32, heads * qlen * d);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit();
    var q = try t.upload(hq);
    defer q.free();
    var k = try t.upload(c.k);
    defer k.free();
    var v = try t.upload(c.v);
    defer v.free();
    const pk = try c.pool(c.k, kv_heads, span, d);
    defer gpa.free(pk);
    const pv = try c.pool(c.v, kv_heads, span, d);
    defer gpa.free(pv);
    var pool_k = try t.upload(pk);
    defer pool_k.free();
    var pool_v = try t.upload(pv);
    defer pool_v.free();
    var table = try t.upload(c.table);
    defer table.free();
    var outs: [2]DeviceBuffer = .{ try t.alloc(heads * qlen * d * 4), try t.alloc(heads * qlen * d * 4) };
    defer for (&outs) |*o| o.free();
    const s = t.stream.handle;
    const kind: c_int = if (t.bf16) 1 else 0;
    try t.on.tf_causal(@ptrFromInt(at(q)), @ptrFromInt(at(k)), @ptrFromInt(at(v)), @ptrFromInt(at(outs[0])), 1, @intCast(qlen), @intCast(span), @intCast(heads), @intCast(kv_heads), @intCast(d), scale, @intCast(q_pos0), 0, @intCast(span * d), @intCast(d), 0, @intCast(span * d), @intCast(d), kind, null, null, null, s, null);
    try t.on.pagedCausal(at(q), at(pool_k), at(pool_v), at(table), at(outs[1]), qlen, span, heads, kv_heads, d, scale, q_pos0, c.table.len, kind, s);
    try t.stream.synchronize();
    const flat = try rig.download(f32, outs[0]);
    defer gpa.free(flat);
    const paged = try rig.download(f32, outs[1]);
    defer gpa.free(paged);
    try std.testing.expectEqualSlices(u32, @ptrCast(flat), @ptrCast(paged));
    // the tile rounds q to the cache's type once, so it sits within that type's rounding of float64
    const want = try gpa.alloc(f64, heads * qlen * d);
    defer gpa.free(want);
    for (0..heads) |h| for (0..qlen) |i| {
        const at_q = (h * qlen + i) * d;
        reference(t, c, hq[at_q..][0..d], h, group, span, d, q_pos0 + i + 1, scale, want[at_q..][0..d]);
    };
    const bound: f64 = if (t.bf16) 2e-2 else 4e-3;
    try std.testing.expect(worst(paged, want) < bound);
}

test "the paged prompt tile writes the flat tile's bits, within the cache type's rounding of float64" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9A6ED };
    try prompt(t, &rng, 16, 2, 256, 128, 0);
    try prompt(t, &rng, 16, 2, 256, 64, 192);
    try prompt(t, &rng, 24, 4, 256, 192, 64);
    try prompt(t, &rng, 16, 4, 128, 128, 128);
}
