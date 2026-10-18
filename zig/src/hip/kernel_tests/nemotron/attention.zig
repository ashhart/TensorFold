//! Nemotron-H decode attention against float64: the planned walk and the paged prompt tile at one query, long spans.

const std = @import("std");
const rig = @import("../rig.zig");
const plan = @import("../../launches/plan.zig");
const DeviceBuffer = @import("../../memory.zig").DeviceBuffer;

const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;

const page = 64;
const heads = 32;
const kv_heads = 2;
const d = 128;
const span = 65536;

/// Visible lengths of the queries: page and tile edges, then spread up to the longest context.
const lengths = [_]usize{ 1, 2, 63, 64, 65, 127, 128, 129, 1000, 4097, 16385, 65536 };

/// softmax(q . k * scale) v in float64 for one head of a query over keys [0, visible) of the flat (kv, span, d) cache.
fn reference(
    t: *const Rig,
    k: []const u16,
    v: []const u16,
    q: []const f64,
    head: usize,
    visible: usize,
    scale: f64,
    scores: []f64,
    out: []f64,
) void {
    const kv = head / (heads / kv_heads);
    var top: f64 = -std.math.inf(f64);
    for (0..visible) |key| {
        var dot: f64 = 0;
        for (0..d) |j| dot += q[j] * t.value(k[(kv * span + key) * d + j]);
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
        out[j] += scores[key] / sum * t.value(v[(kv * span + key) * d + j]);
    };
}

/// The largest error of a query's heads over the largest magnitude of its float64 output.
fn relative(got: []const f32, want: []const f64) f64 {
    var scale: f64 = 0;
    var max: f64 = 0;
    for (got, want) |g, w| {
        max = @max(max, @abs(@as(f64, g) - w));
        scale = @max(scale, @abs(w));
    }
    return max / scale;
}

/// Pools of the decode/pages.hpp layout, ((h * count + page) * 64 + slot) * d, under a permuted page table.
fn pooled(src: []const u16, table: []const u32) ![]u16 {
    const out = try gpa.alloc(u16, src.len);
    for (0..kv_heads) |h| for (0..span) |pos| {
        const to = ((h * table.len + table[pos / page]) * page + pos % page) * d;
        @memcpy(out[to..][0..d], src[(h * span + pos) * d ..][0..d]);
    };
    return out;
}

/// The cache: keys and values flat for the reference, pooled on the device for the kernels, and the page table.
const Cache = struct { k: []const u16, v: []const u16, pool_k: u64, pool_v: u64, table: []const u32 };

/// Worst error over the float64 scale of the planned walk and the one-row tile, q at `gain` and in the activation.
fn sweep(t: *Rig, rng: *rig.Rng, c: Cache, gain: f32) ![2]f64 {
    const rows = lengths.len;
    const scale: f32 = 1 / @sqrt(@as(f32, d));
    const hq = try gpa.alloc(f32, rows * heads * d);
    defer gpa.free(hq);
    const wq = try gpa.alloc(f64, hq.len);
    defer gpa.free(wq);
    for (hq, wq) |*x, *w| {
        w.* = t.value(t.bits(rng.unit() * gain));
        x.* = @floatCast(w.*);
    }
    var q = try t.upload(hq);
    defer q.free();
    // one slot: descriptor [cached, kept, k pool, v pool], then its page table
    const words = 4;
    const desc = try gpa.alloc(u64, words + (c.table.len + 1) / 2);
    defer gpa.free(desc);
    @memset(desc, 0);
    desc[2] = c.pool_k;
    desc[3] = c.pool_v;
    @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. c.table.len * 4], std.mem.sliceAsBytes(c.table));
    var dev_desc = try t.upload(desc);
    defer dev_desc.free();
    var descs = try t.upload(&[_]u64{at(dev_desc)});
    defer descs.free();
    var pos: [rows]i32 = undefined;
    for (&pos, lengths) |*p, n| p.* = @intCast(n - 1);
    var dev_pos = try t.upload(&pos);
    defer dev_pos.free();
    const zeros: [rows]i32 = @splat(0);
    var slot = try t.upload(&zeros);
    defer slot.free();
    var first = try t.upload(&[_]i32{0});
    defer first.free();
    var count = try t.upload(&[_]i32{rows});
    defer count.free();
    var dev_table = try t.upload(c.table);
    defer dev_table.free();
    var scores = try t.alloc(rows * heads * span * 4);
    defer scores.free();
    var stats = try t.alloc(rows * heads * 2 * 4);
    defer stats.free();
    var partials = try t.alloc(rows * heads * (span / 128) * d * 4);
    defer partials.free();
    var outs: [2]DeviceBuffer = .{ try t.alloc(rows * heads * d * 4), try t.alloc(rows * heads * d * 4) };
    defer for (&outs) |*o| o.free();
    const s = t.stream.handle;
    const p: plan.PlanRef = .{
        .rows = rows,
        .slots = 1,
        .args = .{
            .pos = at(dev_pos),
            .slot = at(slot),
            .first = at(first),
            .count = at(count),
            .desc = at(descs),
            .pages = words,
            .pool = @intCast(c.table.len),
        },
    };
    try t.on.planCausal(
        at(q),
        at(outs[0]),
        at(scores),
        at(stats),
        at(partials),
        p,
        0,
        heads,
        kv_heads,
        d,
        span,
        scale,
        t.kind(),
        s,
    );
    const kind: c_int = if (t.bf16) 1 else 0;
    for (lengths, 0..) |n, r| {
        const off = r * heads * d * 4;
        try t.on.pagedCausal(
            at(q) + off,
            c.pool_k,
            c.pool_v,
            at(dev_table),
            at(outs[1]) + off,
            1,
            n,
            heads,
            kv_heads,
            d,
            scale,
            n - 1,
            c.table.len,
            kind,
            s,
        );
    }
    try t.stream.synchronize();
    const planned = try rig.download(f32, outs[0]);
    defer gpa.free(planned);
    const tile = try rig.download(f32, outs[1]);
    defer gpa.free(tile);
    const want = try gpa.alloc(f64, heads * d);
    defer gpa.free(want);
    const sc = try gpa.alloc(f64, span);
    defer gpa.free(sc);
    var worst: [2]f64 = .{ 0, 0 };
    for (lengths, 0..) |n, r| {
        const base = r * heads * d;
        for (0..heads) |h| reference(t, c.k, c.v, wq[base + h * d ..][0..d], h, n, scale, sc, want[h * d ..][0..d]);
        const e: [2]f64 = .{
            relative(planned[base..][0 .. heads * d], want),
            relative(tile[base..][0 .. heads * d], want),
        };
        std.debug.print(
            "nemotron attention q x{d}: {d} keys: planned {e:.2}, paged tile {e:.2} of the float64 scale\n",
            .{ gain, n, e[0], e[1] },
        );
        worst = .{ @max(worst[0], e[0]), @max(worst[1], e[1]) };
    }
    return worst;
}

test "Nemotron's decode attention: the planned walk within 1e-5 of float64 and no worse than the paged tile" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4E3A7 };
    const k = try gpa.alloc(u16, kv_heads * span * d);
    defer gpa.free(k);
    const v = try gpa.alloc(u16, kv_heads * span * d);
    defer gpa.free(v);
    for (k) |*x| x.* = t.bits(rng.unit());
    for (v) |*x| x.* = t.bits(rng.unit());
    const table = try gpa.alloc(u32, span / page);
    defer gpa.free(table);
    for (table, 0..) |*p, i| p.* = @intCast((i * 5 + 3) % table.len);
    const pk = try pooled(k, table);
    defer gpa.free(pk);
    const pv = try pooled(v, table);
    defer gpa.free(pv);
    var pool_k = try t.upload(pk);
    defer pool_k.free();
    var pool_v = try t.upload(pv);
    defer pool_v.free();
    const c: Cache = .{ .k = k, .v = v, .pool_k = at(pool_k), .pool_v = at(pool_v), .table = table };
    // unit q spreads a query over many keys; sixteen times it puts most of the weight on a few
    for ([_]f32{ 1, 16 }) |gain| {
        const w = try sweep(t, &rng, c, gain);
        try rig.expect(w[0] < 1e-5 and w[0] <= w[1], "q x{d}: planned {e} against the tile {e}", .{ gain, w[0], w[1] });
    }
}

/// The planned walk over q in the activation type into rows of it, against q cast to fp32, the walk and the cast back.
fn actWalk(t: *Rig, rng: *rig.Rng, c: Cache) !bool {
    const rows = lengths.len;
    const scale: f32 = 1 / @sqrt(@as(f32, d));
    const hq = try gpa.alloc(u16, rows * heads * d);
    defer gpa.free(hq);
    for (hq) |*x| x.* = t.bits(rng.unit());
    var q = try t.upload(hq);
    defer q.free();
    const words = 4;
    const desc = try gpa.alloc(u64, words + (c.table.len + 1) / 2);
    defer gpa.free(desc);
    @memset(desc, 0);
    desc[2] = c.pool_k;
    desc[3] = c.pool_v;
    @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. c.table.len * 4], std.mem.sliceAsBytes(c.table));
    var dev_desc = try t.upload(desc);
    defer dev_desc.free();
    var descs = try t.upload(&[_]u64{at(dev_desc)});
    defer descs.free();
    var pos: [rows]i32 = undefined;
    for (&pos, lengths) |*p, n| p.* = @intCast(n - 1);
    var dev_pos = try t.upload(&pos);
    defer dev_pos.free();
    const zeros: [rows]i32 = @splat(0);
    var slot = try t.upload(&zeros);
    defer slot.free();
    var first = try t.upload(&[_]i32{0});
    defer first.free();
    var count = try t.upload(&[_]i32{rows});
    defer count.free();
    var scores = try t.alloc(rows * heads * span * 4);
    defer scores.free();
    var stats = try t.alloc(rows * heads * 2 * 4);
    defer stats.free();
    var partials = try t.alloc(rows * heads * (span / 128) * d * 4);
    defer partials.free();
    var q32 = try t.alloc(rows * heads * d * 4);
    defer q32.free();
    var o32 = try t.alloc(rows * heads * d * 4);
    defer o32.free();
    var outs: [2]DeviceBuffer = .{ try t.alloc(rows * heads * d * 2), try t.alloc(rows * heads * d * 2) };
    defer for (&outs) |*o| o.free();
    const s = t.stream.handle;
    const p: plan.PlanRef = .{
        .rows = rows,
        .slots = 1,
        .args = .{
            .pos = at(dev_pos),
            .slot = at(slot),
            .first = at(first),
            .count = at(count),
            .desc = at(descs),
            .pages = words,
            .pool = @intCast(c.table.len),
        },
    };
    const n: c_longlong = @intCast(rows * heads * d);
    try t.on.tf_cast(@ptrFromInt(at(q)), t.kind(), @ptrFromInt(at(q32)), 0, n, s);
    try t.on.planCausal(
        at(q32),
        at(o32),
        at(scores),
        at(stats),
        at(partials),
        p,
        0,
        heads,
        kv_heads,
        d,
        span,
        scale,
        t.kind(),
        s,
    );
    try t.on.tf_cast(@ptrFromInt(at(o32)), 0, @ptrFromInt(at(outs[0])), t.kind(), n, s);
    try t.on.planCausalAs(
        at(q),
        at(outs[1]),
        at(scores),
        at(stats),
        at(partials),
        p,
        0,
        heads,
        kv_heads,
        d,
        span,
        scale,
        t.kind(),
        true,
        s,
    );
    try t.stream.synchronize();
    const a = try rig.download(u16, outs[0]);
    defer gpa.free(a);
    const b = try rig.download(u16, outs[1]);
    defer gpa.free(b);
    return std.mem.eql(u16, a, b);
}

test "the planned walk over q and out in the activation type writes the bytes of the casts around the fp32 walk" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0xAC7 };
    const hk = try gpa.alloc(u16, kv_heads * span * d);
    defer gpa.free(hk);
    const hv = try gpa.alloc(u16, kv_heads * span * d);
    defer gpa.free(hv);
    for (hk) |*x| x.* = t.bits(rng.unit());
    for (hv) |*x| x.* = t.bits(rng.unit());
    const table = try gpa.alloc(u32, span / page);
    defer gpa.free(table);
    for (table, 0..) |*e, i| e.* = @intCast((i * 7919) % table.len);
    const pk = try pooled(hk, table);
    defer gpa.free(pk);
    const pv = try pooled(hv, table);
    defer gpa.free(pv);
    var dk = try t.upload(pk);
    defer dk.free();
    var dv = try t.upload(pv);
    defer dv.free();
    const c: Cache = .{ .k = hk, .v = hv, .pool_k = at(dk), .pool_v = at(dv), .table = table };
    try rig.expect(try actWalk(t, &rng, c), "planned walk in the activation type: the bytes differ", .{});
    std.debug.print(
        "  nemotron planned walk in the activation type: {d} rows, the bytes of the casts around it\n",
        .{lengths.len},
    );
}

test "the planned key write from the activation type writes the bytes of the cast keys' write" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x6E1 };
    const pages = 8;
    const rows = 5;
    const table = [_]u32{ 3, 0, 6, 1, 7, 2, 5, 4 };
    const words = 4;
    var pools: [2][2]DeviceBuffer = undefined;
    for (&pools) |*pp| for (pp) |*b| {
        b.* = try t.alloc(kv_heads * pages * page * d * 2);
        try b.fill8(0);
    };
    defer for (&pools) |*pp| for (pp) |*b| b.free();
    var descs: [2]DeviceBuffer = undefined;
    for (0..2) |v| {
        var desc: [words + pages / 2]u64 = @splat(0);
        desc[2] = at(pools[v][0]);
        desc[3] = at(pools[v][1]);
        @memcpy(std.mem.sliceAsBytes(desc[words..])[0 .. pages * 4], std.mem.sliceAsBytes(&table));
        descs[v] = try t.upload(&desc);
    }
    defer for (&descs) |*b| b.free();
    const hk = try gpa.alloc(u16, rows * kv_heads * d);
    defer gpa.free(hk);
    const hv = try gpa.alloc(u16, rows * kv_heads * d);
    defer gpa.free(hv);
    for (hk) |*x| x.* = t.bits(rng.unit() * 3);
    for (hv) |*x| x.* = t.bits(rng.unit());
    var k = try t.upload(hk);
    defer k.free();
    var vals = try t.upload(hv);
    defer vals.free();
    var k32 = try t.alloc(rows * kv_heads * d * 4);
    defer k32.free();
    const pos = [rows]i32{ 0, 63, 64, 200, 511 };
    var dev_pos = try t.upload(&pos);
    defer dev_pos.free();
    const zeros: [rows]i32 = @splat(0);
    var slot = try t.upload(&zeros);
    defer slot.free();
    const s = t.stream.handle;
    for (0..2) |v| {
        var list = try t.upload(&[_]u64{at(descs[v])});
        defer list.free();
        const args: plan.PlanArgs = .{
            .pos = at(dev_pos),
            .slot = at(slot),
            .desc = at(list),
            .pages = words,
            .pool = pages,
        };
        const p: plan.PlanRef = .{ .rows = rows, .slots = 1, .args = args };
        if (v == 0) {
            try t.on.tf_cast(@ptrFromInt(at(k)), t.kind(), @ptrFromInt(at(k32)), 0, @intCast(rows * kv_heads * d), s);
            try t.on.planKvWrite(at(k32), at(vals), t.kind(), p, 0, kv_heads, d, s);
        } else try t.on.planKvWriteAct(at(k), at(vals), t.kind(), p, 0, kv_heads, d, s);
        try t.stream.synchronize();
    }
    for (0..2) |w| {
        const a = try rig.download(u8, pools[0][w]);
        defer gpa.free(a);
        const b = try rig.download(u8, pools[1][w]);
        defer gpa.free(b);
        try rig.expect(std.mem.eql(u8, a, b), "planned key write from the activation type: pool {d} differs", .{w});
    }
    std.debug.print("  nemotron planned key write from the activation type: the cast keys' bytes\n", .{});
}
