//! Flash Next's stream kernels and n-gram hashes against host references: hashes exact, values within an ulp or two.
const std = @import("std");
const rig = @import("rig.zig");
const Rig = rig.Rig;
const gpa = rig.gpa;
const at = rig.at;
const DeviceBuffer = @import("../memory.zig").DeviceBuffer;

pub fn ptr(comptime T: type, b: DeviceBuffer) T {
    return @ptrFromInt(at(b));
}

/// The activation's round-to-nearest-even of `v`, widened back.
pub fn rnd(t: *const Rig, v: f32) f32 {
    if (!t.bf16) return @floatCast(rig.f16Value(rig.f16Bits(v)));
    const u: u32 = @bitCast(v);
    const b: u32 = (u + 0x7fff + ((u >> 16) & 1)) >> 16;
    return @bitCast(b << 16);
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// Within `ulps` steps of the activation type of `want`'s size (or of 1e-3 near zero).
pub fn close(t: *const Rig, got: f32, want: f32, ulps: f32) bool {
    const step: f32 = if (t.bf16) 1.0 / 128.0 else 1.0 / 1024.0;
    return @abs(got - want) <= ulps * step * @max(@abs(want), 1e-3);
}

/// Random activations exact in the activation type, scaled.
pub fn acts(t: *const Rig, rng: *rig.Rng, n: usize, scale: f32) ![]u16 {
    const out = try gpa.alloc(u16, n);
    for (out) |*x| x.* = t.bits(rng.unit() * scale);
    return out;
}

pub fn value(t: *const Rig, b: u16) f32 {
    return @floatCast(t.value(b));
}

fn rmsInv(xs: []const f32, eps: f32) f32 {
    var sum: f32 = 0;
    for (xs) |x| sum += x * x;
    return 1.0 / @sqrt(sum / @as(f32, @floatFromInt(xs.len)) + eps);
}

test "hyper-connection norm, gates and mix: the reference's roundings, each row alone" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x4C };
    const rows = 3;
    const s = 4;
    const d = 2560;
    const w = s * d;
    const low = 320;
    const eps: f32 = 1e-6;
    const hh = try acts(t, &rng, rows * w, 2.0);
    defer gpa.free(hh);
    const hb = try acts(t, &rng, rows * d, 1.0);
    defer gpa.free(hb);
    const hi = try acts(t, &rng, rows * s, 1.0);
    defer gpa.free(hi);
    const scale = try gpa.alloc(f32, w);
    defer gpa.free(scale);
    for (scale) |*x| x.* = 1.0 + rng.unit() * 0.5;
    var h = try t.upload(hh);
    defer h.free();
    var branch = try t.upload(hb);
    defer branch.free();
    var inject = try t.upload(hi);
    defer inject.free();
    var sc = try t.upload(scale);
    defer sc.free();
    var h_out = try t.alloc(rows * w * 2);
    defer h_out.free();
    var x_out = try t.alloc(rows * w * 2);
    defer x_out.free();
    var x_one = try t.alloc(w * 2);
    defer x_one.free();
    const k = t.kind();
    const st = t.stream.handle;
    try t.on.fn_hc_norm(
        ptr(?*const anyopaque, h),
        ptr(?*const anyopaque, branch),
        ptr(?*const anyopaque, inject),
        ptr(?*anyopaque, h_out),
        ptr(?*anyopaque, x_out),
        ptr(?[*]const f32, sc),
        k,
        rows,
        s,
        d,
        eps,
        st,
    );
    try t.on.fn_hc_norm(
        ptr(?*const anyopaque, h),
        ptr(?*const anyopaque, branch),
        ptr(?*const anyopaque, inject),
        null,
        ptr(?*anyopaque, x_one),
        ptr(?[*]const f32, sc),
        k,
        1,
        s,
        d,
        eps,
        st,
    );
    // the gates and the mix over random fp32 sums
    const zh = try gpa.alloc(f32, rows * (low + s));
    defer gpa.free(zh);
    for (zh) |*x| x.* = rng.unit() * 6.0;
    const gh = try gpa.alloc(f32, rows * w);
    defer gpa.free(gh);
    for (gh) |*x| x.* = rng.unit() * 6.0;
    var z = try t.upload(zh);
    defer z.free();
    var g = try t.upload(gh);
    defer g.free();
    var act = try t.alloc(rows * low * 2);
    defer act.free();
    var inj = try t.alloc(rows * s * 2);
    defer inj.free();
    var mixed = try t.alloc(rows * d * 2);
    defer mixed.free();
    try t.on.fn_hc_gates(ptr(?[*]const f32, z), ptr(?*anyopaque, act), ptr(?*anyopaque, inj), k, rows, low, s, st);
    try t.on.fn_hc_mix(
        ptr(?[*]const f32, g),
        ptr(?*const anyopaque, x_out),
        ptr(?*anyopaque, mixed),
        k,
        rows,
        s,
        d,
        st,
    );
    try t.stream.synchronize();
    const gho = try rig.download(u16, h_out);
    defer gpa.free(gho);
    const gx = try rig.download(u16, x_out);
    defer gpa.free(gx);
    const gx1 = try rig.download(u16, x_one);
    defer gpa.free(gx1);
    try std.testing.expectEqualSlices(u16, gx[0..w], gx1);
    // the branch as two fp32 shares (a tp rank's gather): summed in order, then as the whole branch
    const shares = try gpa.alloc(f32, 2 * rows * d);
    defer gpa.free(shares);
    for (shares) |*v| v.* = rng.unit() * 0.5;
    var dev_shares = try t.upload(shares);
    defer dev_shares.free();
    var h_shared = try t.upload(hh);
    defer h_shared.free();
    var x_shared = try t.alloc(rows * w * 2);
    defer x_shared.free();
    try t.on.fn_hc_norm_shares(
        ptr(?*const anyopaque, h_shared),
        ptr(?*const anyopaque, dev_shares),
        ptr(?*const anyopaque, inject),
        ptr(?*anyopaque, h_shared),
        ptr(?*anyopaque, x_shared),
        ptr(?[*]const f32, sc),
        k,
        rows,
        s,
        d,
        eps,
        2,
        rows * d,
        st,
    );
    try t.stream.synchronize();
    const ghs = try rig.download(u16, h_shared);
    defer gpa.free(ghs);
    for (0..rows) |r| for (0..s) |si| for (0..d) |j| {
        const b = rnd(t, shares[r * d + j] + shares[rows * d + r * d + j]);
        const want = rnd(t, value(t, hh[(r * s + si) * d + j]) + rnd(t, b * value(t, hi[r * s + si])));
        try std.testing.expectEqual(t.bits(want), ghs[(r * s + si) * d + j]);
    };
    const want_h = try gpa.alloc(f32, d);
    defer gpa.free(want_h);
    for (0..rows) |r| for (0..s) |si| {
        for (0..d) |j| {
            const add = rnd(t, value(t, hb[r * d + j]) * value(t, hi[r * s + si]));
            want_h[j] = rnd(t, value(t, hh[(r * s + si) * d + j]) + add);
            try std.testing.expectEqual(t.bits(want_h[j]), gho[(r * s + si) * d + j]);
        }
        const inv = rmsInv(want_h, eps);
        for (0..d) |j| {
            const want = rnd(t, want_h[j] * inv * scale[si * d + j]);
            try rig.expect(
                close(t, value(t, gx[(r * s + si) * d + j]), want, 1),
                "hc norm row {d} stream {d} dim {d}",
                .{ r, si, j },
            );
        }
    };
    const ga = try rig.download(u16, act);
    defer gpa.free(ga);
    const gi = try rig.download(u16, inj);
    defer gpa.free(gi);
    for (0..rows) |r| for (0..low + s) |c| {
        const v = rnd(t, rnd(t, zh[r * (low + s) + c]) / 4.0);
        if (c < low) {
            try rig.expect(close(t, value(t, ga[r * low + c]), rnd(t, v * sigmoid(v)), 1), "hc act {d} {d}", .{ r, c });
        } else try rig.expect(
            close(t, value(t, gi[r * s + c - low]), 2 * rnd(t, sigmoid(v)), 1),
            "hc inject {d} {d}",
            .{ r, c },
        );
    };
    const gm = try rig.download(u16, mixed);
    defer gpa.free(gm);
    for (0..rows) |r| for (0..d) |j| {
        var total: f32 = 0;
        for (0..s) |si| {
            const e = (r * s + si) * d + j;
            total += rnd(t, rnd(t, sigmoid(rnd(t, gh[e]))) * value(t, gx[e]));
        }
        try rig.expect(close(t, value(t, gm[r * d + j]), rnd(t, total / 4.0), 2), "hc mix {d} {d}", .{ r, j });
    };
}

test "the wide norm over a row of every stream, and the gates without inject gates" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x51DE };
    const rows = 3;
    const w = 10240;
    const low = 320;
    const eps: f32 = 1e-6;
    const hx = try acts(t, &rng, rows * w, 2.0);
    defer gpa.free(hx);
    const scale = try gpa.alloc(f32, w);
    defer gpa.free(scale);
    for (scale) |*x| x.* = 1.0 + rng.unit() * 0.5;
    var x = try t.upload(hx);
    defer x.free();
    var sc = try t.upload(scale);
    defer sc.free();
    var y = try t.alloc(rows * w * 2);
    defer y.free();
    const zh = try gpa.alloc(f32, rows * low);
    defer gpa.free(zh);
    for (zh) |*v| v.* = rng.unit() * 6.0;
    var z = try t.upload(zh);
    defer z.free();
    var act = try t.alloc(rows * low * 2);
    defer act.free();
    const st = t.stream.handle;
    try t.on.fn_rms_wide(
        ptr(?*const anyopaque, x),
        ptr(?[*]const f32, sc),
        ptr(?*anyopaque, y),
        t.kind(),
        rows,
        w,
        eps,
        st,
    );
    try t.on.fn_hc_gates(ptr(?[*]const f32, z), ptr(?*anyopaque, act), null, t.kind(), rows, low, 4, st);
    try t.stream.synchronize();
    const gy = try rig.download(u16, y);
    defer gpa.free(gy);
    const xs = try gpa.alloc(f32, w);
    defer gpa.free(xs);
    for (0..rows) |r| {
        for (xs, 0..) |*v, j| v.* = value(t, hx[r * w + j]);
        const inv = rmsInv(xs, eps);
        for (0..w) |j| try rig.expect(
            close(t, value(t, gy[r * w + j]), rnd(t, xs[j] * inv * scale[j]), 1),
            "wide norm row {d} dim {d}",
            .{ r, j },
        );
    }
    const ga = try rig.download(u16, act);
    defer gpa.free(ga);
    for (0..rows * low) |i| {
        const v = rnd(t, rnd(t, zh[i]) / 4.0);
        try rig.expect(close(t, value(t, ga[i]), rnd(t, v * sigmoid(v)), 1), "gate {d}", .{i});
    }
}

test "the n-gram gate and its dilated conv: gate, norm, conv and residual sums as the reference rounds them" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    var rng: rig.Rng = .{ .state = 0x91E };
    const rows = 5;
    const s = 4;
    const d = 2560;
    const w = s * d;
    const taps = 4;
    const dil = 3;
    const tail = dil * (taps - 1);
    const eps: f32 = 1e-6;
    const hkv = try acts(t, &rng, rows * (w + d), 2.0);
    defer gpa.free(hkv);
    const hh = try acts(t, &rng, rows * w, 2.0);
    defer gpa.free(hh);
    const htail = try acts(t, &rng, tail * w, 1.0);
    defer gpa.free(htail);
    const scales = try gpa.alloc(f32, 3 * w + w * taps);
    defer gpa.free(scales);
    for (scales) |*x| x.* = 1.0 + rng.unit() * 0.5;
    var kv = try t.upload(hkv);
    defer kv.free();
    var h = try t.upload(hh);
    defer h.free();
    var sc = try t.upload(scales);
    defer sc.free();
    var cin = try t.alloc((tail + rows) * w * 2);
    defer cin.free();
    try cin.upload(0, std.mem.sliceAsBytes(htail));
    var gated = try t.alloc(rows * w * 2);
    defer gated.free();
    var out = try t.alloc(rows * w * 2);
    defer out.free();
    const k = t.kind();
    const st = t.stream.handle;
    const base: usize = @intCast(at(sc));
    const f32p = struct {
        fn at(b: usize, i: usize) ?[*]const f32 {
            return @ptrFromInt(b + i * 4);
        }
    };
    try t.on.fn_ple_gate(
        ptr(?*const anyopaque, kv),
        ptr(?*const anyopaque, h),
        f32p.at(base, 0),
        f32p.at(base, w),
        f32p.at(base, 2 * w),
        ptr(?*anyopaque, gated),
        @ptrFromInt(at(cin) + tail * w * 2),
        k,
        rows,
        s,
        d,
        eps,
        st,
    );
    try t.on.fn_ple_conv(
        ptr(?*const anyopaque, cin),
        f32p.at(base, 3 * w),
        ptr(?*const anyopaque, gated),
        ptr(?*const anyopaque, h),
        ptr(?*anyopaque, out),
        k,
        rows,
        w,
        taps,
        dil,
        st,
    );
    try t.stream.synchronize();
    const gg = try rig.download(u16, gated);
    defer gpa.free(gg);
    const gc = try rig.download(u16, cin);
    defer gpa.free(gc);
    const go = try rig.download(u16, out);
    defer gpa.free(go);
    const kn = try gpa.alloc(f32, d);
    defer gpa.free(kn);
    const qn = try gpa.alloc(f32, d);
    defer gpa.free(qn);
    for (0..rows) |r| for (0..s) |si| {
        for (0..d) |j| {
            kn[j] = value(t, hkv[r * (w + d) + si * d + j]);
            qn[j] = value(t, hh[r * w + si * d + j]);
        }
        const rk = rmsInv(kn, eps);
        const rq = rmsInv(qn, eps);
        var dot: f32 = 0;
        for (0..d) |j| dot += rnd(
            t,
            rnd(t, kn[j] * rk * scales[si * d + j]) * rnd(t, qn[j] * rq * scales[w + si * d + j]),
        );
        const g1 = rnd(t, rnd(t, dot) / rnd(t, @sqrt(@as(f32, d))));
        const root = rnd(t, @sqrt(@max(@abs(g1), 1e-6)));
        const gate = rnd(t, sigmoid(rnd(t, std.math.sign(g1) * root)));
        for (0..d) |j| {
            kn[j] = rnd(t, gate * value(t, hkv[r * (w + d) + w + j]));
            try rig.expect(
                close(t, value(t, gg[r * w + si * d + j]), kn[j], 2),
                "ple gate row {d} stream {d} dim {d}",
                .{ r, si, j },
            );
        }
        const rc = rmsInv(kn, eps);
        for (0..d) |j| {
            const want = rnd(t, kn[j] * rc * scales[2 * w + si * d + j]);
            try rig.expect(
                close(t, value(t, gc[(tail + r) * w + si * d + j]), want, 2),
                "ple norm row {d} stream {d} dim {d}",
                .{ r, si, j },
            );
        }
    };
    for (0..rows) |r| for (0..w) |c| {
        var y: f32 = 0;
        for (0..taps) |j| y = @mulAdd(f32, scales[3 * w + c * taps + j], value(t, gc[(r + dil * j) * w + c]), y);
        const yb = rnd(t, y);
        const silu = rnd(t, yb * rnd(t, sigmoid(yb)));
        const want = rnd(t, value(t, hh[r * w + c]) + rnd(t, value(t, gg[r * w + c]) + silu));
        try rig.expect(close(t, value(t, go[r * w + c]), want, 1), "ple conv row {d} channel {d}", .{ r, c });
    };
}

/// The reference's hash of row `row`'s n-gram for head `head`: history then tokens, eos for those past a segment.
pub fn ngramId(
    history: [2]i64,
    tokens: []const i32,
    row: usize,
    head: usize,
    eos: i64,
    mult: []const i64,
    size: i64,
    offset: i64,
    per: usize,
) u32 {
    var seq: [66]i64 = undefined;
    seq[0], seq[1] = history;
    for (tokens[0 .. row + 1], 0..) |x, i| seq[2 + i] = x;
    const here = 2 + row;
    var before: ?usize = null;
    for (0..here) |q| if (seq[q] == eos) {
        before = q;
    };
    const in_seg = here - if (before) |b| b + 1 else 0;
    var mixed: u64 = 0;
    for (0..2 + head / per) |q| {
        const tok = if (in_seg >= q) seq[here - q] else eos;
        const term = @as(u64, @bitCast(tok)) *% @as(u64, @bitCast(mult[q]));
        mixed = if (q == 0) term else mixed ^ term;
    }
    var m = @rem(@as(i64, @bitCast(mixed)), size);
    if (m < 0) m += size;
    return @intCast(m + offset);
}

test "n-gram rows: every head's hash of the window, segments cut at eos, exact" {
    const t = try Rig.open(1 << 20);
    defer t.close();
    const eos: i64 = 248044;
    const mult = [3]i64{ 23703573157769, 20109073645365, 8052911324071 };
    var sizes: [16]i64 = undefined;
    var offsets: [16]i64 = undefined;
    const primes = [16]i64{
        20000003,
        20000023,
        20000033,
        20000047,
        20000059,
        20000063,
        20000069,
        20000077,
        20000081,
        20000093,
        20000107,
        20000147,
        20000153,
        20000159,
        20000161,
        20000171,
    };
    var off: i64 = 0;
    for (primes, 0..) |p, i| {
        sizes[i] = p;
        offsets[i] = off;
        off += p;
    }
    const cases = [_]struct { history: [2]i64, tokens: []const i32 }{
        .{ .history = .{ 11, 248044 }, .tokens = &.{ 5, 248044, 77, 90000, 248044, 248044, 3, 4, 248319, 0, 12 } },
        .{ .history = .{ 248044, 248044 }, .tokens = &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } },
        .{ .history = .{ 70000, 248044 }, .tokens = &.{248044} },
    };
    for (cases) |c| {
        var meta: [3 + 3 + 32]i64 = undefined;
        meta[0], meta[1], meta[2] = .{ c.history[0], c.history[1], eos };
        @memcpy(meta[3..6], &mult);
        @memcpy(meta[6..22], &sizes);
        @memcpy(meta[22..38], &offsets);
        var dm = try t.upload(&meta);
        defer dm.free();
        var dt = try t.upload(c.tokens);
        defer dt.free();
        const rows: c_int = @intCast(c.tokens.len);
        var ids = try t.alloc(c.tokens.len * 16 * 4);
        defer ids.free();
        try t.on.fn_ngram_ids(
            ptr(?[*]const i32, dt),
            ptr(?*const anyopaque, dm),
            ptr(?*anyopaque, ids),
            rows,
            16,
            8,
            3,
            t.stream.handle,
        );
        try t.stream.synchronize();
        const got = try rig.download(u32, ids);
        defer gpa.free(got);
        for (0..c.tokens.len) |r| for (0..16) |hd| {
            try std.testing.expectEqual(
                ngramId(c.history, c.tokens, r, hd, eos, &mult, sizes[hd], offsets[hd], 8),
                got[r * 16 + hd],
            );
        };
    }
}
