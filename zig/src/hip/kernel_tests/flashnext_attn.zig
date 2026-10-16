//! Flash Next's indexer and sparse attention against host references: selections exact, values near fp64's.
const std = @import("std");
const rig = @import("rig.zig");
const fx = @import("flashnext.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const ptr = fx.ptr;
const rnd = fx.rnd;
const value = fx.value;

test "pooled block keys: the mean of a block's raw keys, normed and rotated at its first position" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9001 };
    const dims = 128;
    const ratio = 4;
    const rotary = 64;
    const first = 3;
    const blocks = 9;
    const theta: f32 = 10_000_000;
    const eps: f32 = 1e-6;
    const hraw = try fx.acts(t, &rng, (first + blocks) * ratio * dims, 2.0);
    defer gpa.free(hraw);
    var scale: [dims]f32 = undefined;
    for (&scale) |*x| x.* = 1.0 + rng.unit() * 0.5;
    var raw = try t.upload(hraw);
    defer raw.free();
    var sc = try t.upload(&scale);
    defer sc.free();
    var pooled = try t.alloc(blocks * dims * 2);
    defer pooled.free();
    try t.on.fn_idx_pool(
        ptr(?*const anyopaque, raw),
        ptr(?*anyopaque, pooled),
        ptr(?[*]const f32, sc),
        t.kind(),
        first,
        blocks,
        dims,
        ratio,
        rotary,
        theta,
        eps,
        t.stream.handle,
    );
    try t.stream.synchronize();
    const got = try rig.download(u16, pooled);
    defer gpa.free(got);
    var x: [dims]f32 = undefined;
    var n: [dims]f32 = undefined;
    for (0..blocks) |j| {
        const b = first + j;
        var sum: f32 = 0;
        for (0..dims) |d| {
            var m: f32 = 0;
            for (0..ratio) |k| m += value(t, hraw[(b * ratio + k) * dims + d]);
            x[d] = rnd(t, m / ratio);
            sum += x[d] * x[d];
        }
        const inv = 1.0 / @sqrt(sum / dims + eps);
        for (0..dims) |d| n[d] = rnd(t, x[d] * inv * scale[d]);
        for (0..dims) |d| {
            var want = n[d];
            if (d < rotary) {
                const half = rotary / 2;
                const f = d % half;
                const freq: f64 = 1.0 / std.math.pow(f64, theta, @as(f64, @floatFromInt(f)) / half);
                const ang: f64 = @as(f64, @floatFromInt(b * ratio)) * freq;
                const c: f64 = @cos(ang);
                const s: f64 = @sin(ang);
                const y: f64 = if (d < half) n[f] * c - n[f + half] * s else n[f] * s + n[f + half] * c;
                want = rnd(t, @floatCast(y));
            }
            const step: f32 = if (t.bf16) 1.0 / 128.0 else 1.0 / 1024.0;
            try rig.expect(
                @abs(value(t, got[j * dims + d]) - want) <= 2 * step * @max(@abs(want), 0.25),
                "pool block {d} dim {d}",
                .{ b, d },
            );
        }
    }
}

test "pooled block keys at device positions: the blocks a window's rows complete, each as a pool from block 0" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x9002 };
    const dims = 128;
    const ratio = 4;
    const blocks = 5;
    const hraw = try fx.acts(t, &rng, blocks * ratio * dims, 2.0);
    defer gpa.free(hraw);
    var scale: [dims]f32 = undefined;
    for (&scale) |*x| x.* = 1.0 + rng.unit() * 0.5;
    var raw = try t.upload(hraw);
    defer raw.free();
    var sc = try t.upload(&scale);
    defer sc.free();
    var whole = try t.alloc(blocks * dims * 2);
    defer whole.free();
    try t.on.fn_idx_pool(
        ptr(?*const anyopaque, raw),
        ptr(?*anyopaque, whole),
        ptr(?[*]const f32, sc),
        t.kind(),
        0,
        blocks,
        dims,
        ratio,
        64,
        1e7,
        1e-6,
        t.stream.handle,
    );
    var mark: [blocks * dims]u16 = undefined;
    @memset(&mark, 0xffff);
    var slots = try t.upload(&mark);
    defer slots.free();
    // rows at 10 .. 15: the rows at 11 and 15 complete blocks 2 and 3
    const pos = [_]i32{ 10, 11, 12, 13, 14, 15 };
    var dp = try t.upload(&pos);
    defer dp.free();
    try t.on.fn_idx_pool_at(
        ptr(?*const anyopaque, raw),
        ptr(?*anyopaque, slots),
        ptr(?[*]const f32, sc),
        t.kind(),
        ptr(?[*]const i32, dp),
        pos.len,
        dims,
        ratio,
        64,
        1e7,
        1e-6,
        t.stream.handle,
    );
    try t.stream.synchronize();
    const want = try rig.download(u16, whole);
    defer gpa.free(want);
    const got = try rig.download(u16, slots);
    defer gpa.free(got);
    for (0..blocks) |b| {
        const done = b == 2 or b == 3;
        for (0..dims) |d| try rig.expect(
            got[b * dims + d] == if (done) want[b * dims + d] else 0xffff,
            "block {d} dim {d}",
            .{ b, d },
        );
    }
}

/// A row's keys as the reference picks them: the top best blocks (ties: lower id) in block order, then the tail.
fn selectRef(scores: []const f32, nb: usize, end: usize, top: usize, ratio: usize, out: *std.ArrayList(i32)) !void {
    const order = try gpa.alloc(usize, nb);
    defer gpa.free(order);
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, scores, struct {
        fn lt(s: []const f32, a: usize, b: usize) bool {
            return s[a] > s[b] or (s[a] == s[b] and a < b);
        }
    }.lt);
    const keep = try gpa.alloc(bool, nb);
    defer gpa.free(keep);
    @memset(keep, false);
    for (order[0..top]) |b| keep[b] = true;
    for (0..nb) |b| if (keep[b]) for (0..ratio) |j| try out.append(gpa, @intCast(b * ratio + j));
    for (nb * ratio..end) |k| try out.append(gpa, @intCast(k));
}

test "block scores and selection: relu-summed heads, the best blocks with ties to the lower id, dense rows untouched" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x5E1 };
    const heads = 4;
    const dims = 128;
    const ratio = 4;
    const top = 32;
    const blocks = 3000;
    const ends = [_]i32{ 100, 4 * 3000 + 3, 4 * 1500, 4 * 33 + 1, 4 * 2999 + 2 };
    const rows = ends.len;
    var complete: [rows]i32 = undefined;
    for (&complete, ends) |*c, e| c.* = @divFloor(e, ratio);
    const hq = try gpa.alloc(f32, rows * heads * dims);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit();
    const hp = try fx.acts(t, &rng, blocks * dims, 1.0);
    defer gpa.free(hp);
    var q = try t.upload(hq);
    defer q.free();
    var pooled = try t.upload(hp);
    defer pooled.free();
    var dc = try t.upload(&complete);
    defer dc.free();
    var de = try t.upload(&ends);
    defer de.free();
    var scores = try t.alloc(rows * blocks * 4);
    defer scores.free();
    try scores.fill8(0);
    const st = t.stream.handle;
    try t.on.fn_idx_scores(
        ptr(?[*]const f32, q),
        ptr(?*const anyopaque, pooled),
        ptr(?[*]const i32, dc),
        ptr(?[*]f32, scores),
        t.kind(),
        rows,
        heads,
        dims,
        blocks,
        blocks,
        top,
        st,
    );
    try t.stream.synchronize();
    const gs = try rig.download(f32, scores);
    defer gpa.free(gs);
    for (0..rows) |r| for (0..blocks) |b| {
        const live = complete[r] > top and b < complete[r];
        var want: f64 = 0;
        if (live) for (0..heads) |h| {
            var dot: f64 = 0;
            for (0..dims) |d| dot += hq[(r * heads + h) * dims + d] * t.value(hp[b * dims + d]);
            want += @max(dot, 0);
        };
        want /= @sqrt(@as(f64, dims));
        try rig.expect(
            @abs(gs[r * blocks + b] - want) <= 1e-4 * @max(@abs(want), 1),
            "score row {d} block {d}",
            .{ r, b },
        );
    };
    // the selection runs on scores with ties written in: the GPU's, then a few equal to the cut
    for (0..rows) |r| for (0..blocks) |b| if (b % 50 == 5) {
        gs[r * blocks + b] = 3.0;
    };
    var sel_in = try t.upload(gs);
    defer sel_in.free();
    const key_stride = top * ratio + ratio;
    var keys = try t.alloc(rows * key_stride * 4);
    defer keys.free();
    try keys.fill8(0xff);
    var count = try t.alloc(rows * 4);
    defer count.free();
    var sparse = try t.alloc(rows * 4);
    defer sparse.free();
    try t.on.fn_idx_select(
        ptr(?[*]const f32, sel_in),
        blocks,
        ptr(?[*]const i32, dc),
        ptr(?[*]const i32, de),
        ptr(?[*]i32, keys),
        key_stride,
        ptr(?[*]i32, count),
        ptr(?[*]i32, sparse),
        rows,
        top,
        ratio,
        st,
    );
    try t.stream.synchronize();
    const gk = try rig.download(i32, keys);
    defer gpa.free(gk);
    const gc = try rig.download(i32, count);
    defer gpa.free(gc);
    const gsp = try rig.download(i32, sparse);
    defer gpa.free(gsp);
    for (0..rows) |r| {
        const nb: usize = @intCast(complete[r]);
        if (nb <= top) {
            try std.testing.expectEqual(@as(i32, 0), gsp[r]);
            try std.testing.expectEqual(ends[r], gc[r]);
            continue;
        }
        var want: std.ArrayList(i32) = .empty;
        defer want.deinit(gpa);
        try selectRef(gs[r * blocks ..][0..nb], nb, @intCast(ends[r]), top, ratio, &want);
        try std.testing.expectEqual(@as(i32, 1), gsp[r]);
        try std.testing.expectEqual(@as(i32, @intCast(want.items.len)), gc[r]);
        try std.testing.expectEqualSlices(i32, want.items, gk[r * key_stride ..][0..want.items.len]);
    }
}

test "sparse attention: listed keys and dense rows near fp64's softmax, a row's parts alike at any row count" {
    const t = try Rig.open(1 << 22);
    defer t.close();
    // the whole model's 12 query heads a kv head, and a tp-4 rank's 6
    try sparseCase(t, 24, 2);
    try sparseCase(t, 6, 1);
}

fn sparseCase(t: *Rig, comptime heads: usize, comptime kv_heads: usize) !void {
    var rng: rig.Rng = .{ .state = 0xA77 + heads };
    const dims = 256;
    const cap = 700;
    const parts = 16;
    const rows = 3;
    const stride = 300;
    const scale: f32 = 0.0625;
    const hk = try fx.acts(t, &rng, kv_heads * cap * dims, 1.0);
    defer gpa.free(hk);
    const hv = try fx.acts(t, &rng, kv_heads * cap * dims, 1.0);
    defer gpa.free(hv);
    const hq = try gpa.alloc(f32, rows * heads * dims);
    defer gpa.free(hq);
    for (hq) |*x| x.* = rng.unit() * 4.0;
    var ids: [rows * stride]i32 = @splat(0);
    const count = [rows]i32{ 260, 650, 7 };
    const sparse = [rows]i32{ 1, 0, 1 };
    for (0..rows) |r| for (0..stride) |j| {
        ids[r * stride + j] = @intCast((j * 37 + r * 11) % cap);
    };
    var dk = try t.upload(hk);
    defer dk.free();
    var dv = try t.upload(hv);
    defer dv.free();
    var dq = try t.upload(hq);
    defer dq.free();
    var di = try t.upload(&ids);
    defer di.free();
    var dc = try t.upload(&count);
    defer dc.free();
    var ds = try t.upload(&sparse);
    defer ds.free();
    var po = try t.alloc(rows * heads * parts * dims * 4);
    defer po.free();
    var pm = try t.alloc(rows * heads * parts * 2 * 4);
    defer pm.free();
    var out = try t.alloc(rows * heads * dims * 4);
    defer out.free();
    var one = try t.alloc(heads * dims * 4);
    defer one.free();
    const ck: c_int = if (t.bf16) 2 else 1;
    const st = t.stream.handle;
    try t.on.fn_sparse_attention(
        ptr(?[*]const f32, dq),
        ptr(?*const anyopaque, dk),
        ptr(?*const anyopaque, dv),
        ck,
        ptr(?[*]const i32, di),
        stride,
        ptr(?[*]const i32, dc),
        ptr(?[*]const i32, ds),
        ptr(?[*]f32, po),
        ptr(?[*]f32, pm),
        ptr(?[*]f32, out),
        rows,
        heads,
        kv_heads,
        dims,
        cap,
        scale,
        parts,
        st,
    );
    try t.on.fn_sparse_attention(
        ptr(?[*]const f32, dq),
        ptr(?*const anyopaque, dk),
        ptr(?*const anyopaque, dv),
        ck,
        ptr(?[*]const i32, di),
        stride,
        ptr(?[*]const i32, dc),
        ptr(?[*]const i32, ds),
        ptr(?[*]f32, po),
        ptr(?[*]f32, pm),
        ptr(?[*]f32, one),
        1,
        heads,
        kv_heads,
        dims,
        cap,
        scale,
        parts,
        st,
    );
    try t.stream.synchronize();
    const got = try rig.download(f32, out);
    defer gpa.free(got);
    const solo = try rig.download(f32, one);
    defer gpa.free(solo);
    try std.testing.expectEqualSlices(f32, got[0 .. heads * dims], solo);
    var w: [cap]f64 = undefined;
    for (0..rows) |r| for (0..heads) |h| {
        const kvh = h / (heads / kv_heads);
        const n: usize = @intCast(count[r]);
        var top: f64 = -std.math.inf(f64);
        for (0..n) |j| {
            const key: usize = if (sparse[r] != 0) @intCast(ids[r * stride + j]) else j;
            var s: f64 = 0;
            for (0..dims) |d| s += @as(
                f64,
                hq[(r * heads + h) * dims + d] * scale,
            ) * t.value(hk[(kvh * cap + key) * dims + d]);
            w[j] = s;
            top = @max(top, s);
        }
        var total: f64 = 0;
        for (w[0..n]) |*x| {
            x.* = @exp(x.* - top);
            total += x.*;
        }
        for (0..dims) |d| {
            var acc: f64 = 0;
            for (0..n) |j| {
                const key: usize = if (sparse[r] != 0) @intCast(ids[r * stride + j]) else j;
                acc += w[j] * t.value(hv[(kvh * cap + key) * dims + d]);
            }
            const want = acc / total;
            try rig.expect(
                @abs(got[(r * heads + h) * dims + d] - want) <= 1e-4 * @max(@abs(want), 0.1),
                "attention row {d} head {d} dim {d}",
                .{ r, h, d },
            );
        }
    };
}
