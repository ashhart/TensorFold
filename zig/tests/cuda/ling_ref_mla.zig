//! Ling's MLA on GLM's kernels vs float64: 128-wide absorb and expand, rope, head gate, the 576-wide roped attention.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const glm = @import("glm_cuda");
const Gpu = check.Gpu;
const Rig = qr.RigT;
const bf = qr.helpers.bfOf;
const latent = glm.latent;

const heads = 64;
const lw = 512;
const rw = latent.rope_width;
const kw = lw + rw;
const nope = 128;
const qk = nope + rw;
const dv = 128;
const u = std.math.pow(f64, 2, -24);

/// Worst |got - want| / tol: `want` is `width` of every `stride` bf16 elements of `raw` from `off`.
fn within(r: *Rig, name: []const u8, raw: []const u8, stride: usize, off: usize, width: usize, want: []const f64, tol: []const f64) !void {
    var most: f64 = 0;
    for (want, tol, 0..) |w, t, i| {
        const at = (i / width) * stride + off + i % width;
        const q = @abs(bf(std.mem.bytesToValue(u16, raw[2 * at ..][0..2])) - w) / t;
        most = if (std.math.isNan(q)) std.math.inf(f64) else @max(most, q);
    }
    r.worst = @max(r.worst, most);
    std.debug.print("RESULT {s}: worst {e:.3} of its float64 tolerance\n", .{ name, most });
    try check.expect(most <= 1, "{s} within its float64 tolerance", .{name});
}

fn f64s(r: *Rig, x: []const u16) ![]f64 {
    const out = try r.a.alloc(f64, x.len);
    for (out, x) |*o, v| o.* = bf(v);
    return out;
}

/// The bytes of rows lo .. hi of a [rows][n] bf16 output equal `got`'s.
fn sameSpan(name: []const u8, all: []const u8, got: []const u8, lo: usize, n: usize) !void {
    try check.sameBytes(name, got, all[lo * n * 2 ..][0..got.len]);
}

/// absorbQStrided (192-wide q heads into 576-wide qa rows) and expandV at dv 128 vs float64; row counts' bits.
fn projections(r: *Rig, set: glm.glue.Set) !void {
    const rows = 70;
    const q = try r.bfs(rows * heads * qk, 1.0);
    const wk = try r.bfs(heads * nope * lw, 0.05);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const dwk = try r.dev(std.mem.sliceAsBytes(wk));
    const qa = try r.zeros(rows * heads * kw * 2);
    try latent.absorbQStrided(set, r.s, dq, qk, dwk, qa, kw, rows, heads, nope, lw);
    var want = try r.a.alloc(f64, rows * heads * lw);
    var tol = try r.a.alloc(f64, rows * heads * lw);
    const qd = try f64s(r, q);
    const wkd = try f64s(r, wk);
    for (0..rows) |row| for (0..heads) |h| for (0..lw) |c| {
        var s: f64 = 0;
        var abs: f64 = 0;
        for (0..nope) |k| {
            const p = qd[(row * heads + h) * qk + k] * wkd[(h * nope + k) * lw + c];
            s += p;
            abs += @abs(p);
        }
        want[(row * heads + h) * lw + c] = s;
        tol[(row * heads + h) * lw + c] = @abs(s) * std.math.pow(f64, 2, -8) + abs * (nope + 8) * u + 1e-30;
    };
    const all = try r.back(qa, rows * heads * kw * 2);
    try within(r, "absorb_q 128 (Ling): 70 rows x 64 heads, q stride 192, qa stride 576", all, kw, 0, lw, want, tol);
    for ([_][2]usize{ .{ 0, 1 }, .{ 37, 42 }, .{ 3, 67 } }) |span| {
        const n = span[1] - span[0];
        const o = try r.zeros(n * heads * kw * 2);
        try latent.absorbQStrided(set, r.s, dq + span[0] * heads * qk * 2, qk, dwk, o, kw, n, heads, nope, lw);
        const got = try r.back(o, n * heads * kw * 2);
        for (0..n * heads) |i| try check.sameBytes("absorb_q 128: rows of a narrower call equal the 70-row call's", got[i * kw * 2 ..][0 .. lw * 2], all[(span[0] * heads + i) * kw * 2 ..][0 .. lw * 2]);
    }
    const ol = try r.bfs(rows * heads * lw, 1.0);
    const wv = try r.bfs(heads * dv * lw, 0.05);
    const dol = try r.dev(std.mem.sliceAsBytes(ol));
    const dwv = try r.dev(std.mem.sliceAsBytes(wv));
    const vn = try r.zeros(rows * heads * dv * 2);
    try latent.expandV(set, r.s, dol, dwv, vn, rows, heads, dv, lw);
    want = try r.a.alloc(f64, rows * heads * dv);
    tol = try r.a.alloc(f64, rows * heads * dv);
    const old = try f64s(r, ol);
    const wvd = try f64s(r, wv);
    for (0..rows) |row| for (0..heads) |h| for (0..dv) |n| {
        var s: f64 = 0;
        var abs: f64 = 0;
        for (old[(row * heads + h) * lw ..][0..lw], wvd[(h * dv + n) * lw ..][0..lw]) |a, b| {
            s += a * b;
            abs += @abs(a * b);
        }
        want[(row * heads + h) * dv + n] = s;
        tol[(row * heads + h) * dv + n] = @abs(s) * std.math.pow(f64, 2, -8) + abs * (lw + 8) * u + 1e-30;
    };
    const vall = try r.back(vn, rows * heads * dv * 2);
    try within(r, "expand_v 128 (Ling): 70 rows x 64 heads", vall, dv, 0, dv, want, tol);
    for ([_][2]usize{ .{ 0, 1 }, .{ 37, 42 }, .{ 3, 67 } }) |span| {
        const n = span[1] - span[0];
        const o = try r.zeros(n * heads * dv * 2);
        try latent.expandV(set, r.s, dol + span[0] * heads * lw * 2, dwv, o, n, heads, dv, lw);
        try sameSpan("expand_v 128: rows of a narrower call equal the 70-row call's", vall, try r.back(o, n * heads * dv * 2), span[0], heads * dv);
    }
    check.pass("absorb_q / expand_v at 128-wide heads: within float64 tolerance, every row count's bits equal", .{});
}

/// rope (q heads into qa, cache rows in place) at two positions vs float64 on fp32 angles; head gate vs float64.
fn ropeGate(r: *Rig, set: glm.glue.Set) !void {
    const rows = 40;
    const freqs = latent.ropeFreqs(6000000);
    const dfreq = try r.dev(std.mem.asBytes(&freqs));
    const q = try r.bfs(rows * heads * qk, 1.0);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const qd = try f64s(r, q);
    var name: [128]u8 = undefined;
    for ([_]i32{ 1000, 261000 }) |p0| {
        const at = try r.dev(std.mem.asBytes(&p0));
        const qa = try r.zeros(rows * heads * kw * 2);
        const v: latent.RopeRows = .{ .x = dq + nope * 2, .x_row = heads * qk, .x_head = qk, .out = qa + lw * 2, .o_row = heads * kw, .o_head = kw };
        try latent.rope(set, r.s, v, rows, heads, at, dfreq);
        const want = try r.a.alloc(f64, rows * heads * rw);
        const tol = try r.a.alloc(f64, rows * heads * rw);
        for (0..rows) |row| for (0..heads) |h| for (0..rw / 2) |i| {
            const ang: f64 = freqs[i] * @as(f32, @floatFromInt(p0 + @as(i32, @intCast(row))));
            const x0 = qd[(row * heads + h) * qk + nope + 2 * i];
            const x1 = qd[(row * heads + h) * qk + nope + 2 * i + 1];
            const c = @cos(ang);
            const s = @sin(ang);
            const at2 = (row * heads + h) * rw + 2 * i;
            want[at2] = x0 * c - x1 * s;
            want[at2 + 1] = x1 * c + x0 * s;
            for (0..2) |e| tol[at2 + e] = @abs(want[at2 + e]) * std.math.pow(f64, 2, -8) + (@abs(x0) + @abs(x1)) * 8 * u + 1e-30;
        };
        const all = try r.back(qa, rows * heads * kw * 2);
        try within(r, try std.fmt.bufPrint(&name, "rope: 40 rows x 64 heads at {d} (GPT-J pairs, fp32 angles)", .{p0}), all, kw, lw, rw, want, tol);
        const p1: i32 = p0 + 23;
        const one = try r.zeros(heads * kw * 2);
        const v1: latent.RopeRows = .{ .x = dq + (23 * heads * qk + nope) * 2, .x_row = heads * qk, .x_head = qk, .out = one + lw * 2, .o_row = heads * kw, .o_head = kw };
        try latent.rope(set, r.s, v1, 1, heads, try r.dev(std.mem.asBytes(&p1)), dfreq);
        const got = try r.back(one, heads * kw * 2);
        for (0..heads) |h| try check.sameBytes("rope: row 23 alone equals its row", got[(h * kw + lw) * 2 ..][0 .. rw * 2], all[((23 * heads + h) * kw + lw) * 2 ..][0 .. rw * 2]);
        // a cache row's k_rope in place equals q's rope of the same values at the same position
        const k = try r.dev(all[(23 * heads * kw) * 2 ..][0 .. kw * 2]);
        @memcpy(all[(23 * heads * kw + lw) * 2 ..][0 .. rw * 2], std.mem.sliceAsBytes(q[(23 * heads) * qk + nope ..][0..rw]));
        const kin = try r.dev(all[(23 * heads * kw) * 2 ..][0 .. kw * 2]);
        try latent.rope(set, r.s, .{ .x = kin + lw * 2, .x_row = kw, .x_head = 0, .out = kin + lw * 2, .o_row = kw, .o_head = 0 }, 1, 1, try r.dev(std.mem.asBytes(&p1)), dfreq);
        try check.sameBytes("rope: a cache row in place equals the q rope", try r.back(kin, kw * 2), try r.back(k, kw * 2));
    }
    const o = try r.bfs(rows * heads * dv, 1.0);
    const g = try r.a.alloc(f32, rows * (heads + 3));
    for (g) |*x| x.* = (r.rng.float(f32) * 2 - 1) * 12;
    const dov = try r.dev(std.mem.sliceAsBytes(o));
    try latent.headGate(set, r.s, dov, try r.dev(std.mem.sliceAsBytes(g)), heads + 3, rows, heads, dv);
    const want = try r.a.alloc(f64, rows * heads * dv);
    const tol = try r.a.alloc(f64, rows * heads * dv);
    for (want, tol, 0..) |*w, *t, i| {
        const sg = 1 / (1 + @exp(-@as(f64, g[(i / (heads * dv)) * (heads + 3) + (i / dv) % heads])));
        w.* = bf(o[i]) * sg;
        t.* = @abs(w.*) * std.math.pow(f64, 2, -8) + @abs(bf(o[i])) * 8 * u + 1e-30;
    }
    try within(r, "head gate: 40 rows x 64 heads x 128, g stride 67", try r.back(dov, rows * heads * dv * 2), dv, 0, dv, want, tol);
    check.pass("rope and head gate: within float64 tolerance, a row alone and a cache row in place equal", .{});
}

/// float64 attention over 576-wide rows (scores) and their first 512 (values); bound as glm_ref_attn's, 576 + 8 terms.
fn reference(r: *Rig, q: []const f64, cd: []const f64, pos: usize, rows: usize, scale: f64) !struct { want: []f64, tol: []f64 } {
    const want = try r.a.alloc(f64, rows * heads * lw);
    const tol = try r.a.alloc(f64, rows * heads * lw);
    const s = try r.a.alloc(f64, pos + rows);
    const ds = try r.a.alloc(f64, pos + rows);
    const o = try r.a.alloc(f64, lw);
    const spread = try r.a.alloc(f64, lw);
    for (0..rows) |row| for (0..heads) |h| {
        const qr_ = q[(row * heads + h) * kw ..][0..kw];
        const n = pos + row + 1;
        var most = -std.math.inf(f64);
        for (0..n) |key| {
            var dot: f64 = 0;
            var abs: f64 = 0;
            for (qr_, cd[key * kw ..][0..kw]) |a, b| {
                dot += a * b;
                abs += @abs(a * b);
            }
            s[key] = dot * scale;
            ds[key] = abs * scale * (kw + 8) * u;
            most = @max(most, s[key]);
        }
        @memset(o, 0);
        @memset(spread, 0);
        var l: f64 = 0;
        for (0..n) |key| {
            const p = @exp(s[key] - most);
            l += p;
            for (o, spread, cd[key * kw ..][0..lw]) |*od, *sd, v| {
                od.* += p * v;
                sd.* += p * @abs(v) * (std.math.pow(f64, 2, -8) + ds[key]);
            }
        }
        const base = (row * heads + h) * lw;
        for (0..lw) |d| {
            want[base + d] = o[d] / l;
            tol[base + d] = @abs(o[d] / l) * std.math.pow(f64, 2, -8) + spread[d] / l + 1e-30;
        }
    };
    return .{ .want = want, .tol = tol };
}

/// attentionRoped of `rows` rows at `pos` (rows of qa from `first`) into a fresh [rows][heads][lw] output.
fn attend(r: *Rig, set: glm.glue.Set, qa: u64, cache: u64, pos: usize, first: usize, rows: usize, scale: f32) ![]u8 {
    const nch = latent.chunksFor(pos + rows);
    const out = try r.zeros(rows * heads * lw * 2);
    const p0: i32 = @intCast(pos);
    const p: latent.Partials = .{ .po = try r.zeros(nch * rows * heads * lw * 4), .pm = try r.zeros(nch * rows * heads * 4), .pl = try r.zeros(nch * rows * heads * 4) };
    try latent.attentionRoped(set, r.s, qa + first * heads * kw * 2, cache, try r.dev(std.mem.asBytes(&p0)), p, out, try r.zeros(4), rows, heads, lw, nch, scale);
    return r.back(out, rows * heads * lw * 2);
}

/// Roped attention: 60 rows at 1,500 (keys cross 1,536) vs float64; one-row decode, a verify window and a cut prompt.
fn attention(r: *Rig, set: glm.glue.Set) !void {
    const pos = 1500;
    const rows = 60;
    const scale: f32 = @floatCast(1 / @sqrt(@as(f64, qk)));
    const q = try r.bfs(rows * heads * kw, 1.0);
    const c = try r.bfs((pos + rows) * kw, 1.0);
    const dq = try r.dev(std.mem.sliceAsBytes(q));
    const dc = try r.dev(std.mem.sliceAsBytes(c));
    const all = try attend(r, set, dq, dc, pos, 0, rows, scale);
    const ref64 = try reference(r, try f64s(r, q), try f64s(r, c), pos, rows, scale);
    try within(r, "roped attention (Ling): 60 rows at 1500, 64 heads, 576-wide keys, 16-head blocks", all, lw, 0, lw, ref64.want, ref64.tol);
    const n = heads * lw;
    var name: [128]u8 = undefined;
    for ([_]usize{ 0, 35, 36, 59 }) |row| {
        const one = try attend(r, set, dq, dc, pos + row, row, 1, scale);
        try sameSpan(try std.fmt.bufPrint(&name, "roped attention: row {d} alone (decode) equals its row", .{row}), all, one, row, n);
    }
    const win = try attend(r, set, dq, dc, pos + 30, 30, 8, scale);
    try sameSpan("roped attention: an 8-row verify window across 1,536 equals serial rows", all, win, 30, n);
    const head = try attend(r, set, dq, dc, pos, 0, 36, scale);
    const tail = try attend(r, set, dq, dc, pos + 36, 36, rows - 36, scale);
    try sameSpan("roped attention: a prompt cut at 36 (key 1,536): first part equals uncut", all, head, 0, n);
    try sameSpan("roped attention: a prompt cut at 36 (key 1,536): second part equals uncut", all, tail, 36, n);
    check.pass("roped attention over {d} keys: within float64 tolerance; decode rows, verify window, cut prompt equal", .{pos + rows});
}

/// `ling-mla`: the checks above on synthetic inputs (seeded).
pub fn run(gpu: Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(731);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var mod = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer mod.unload();
    const ks = try glm.glue.Kernels.load(mod);
    const set: glm.glue.Set = .{ .glue = &ks };
    try projections(&r, set);
    try ropeGate(&r, set);
    try attention(&r, set);
    check.pass("ling-mla: every check within {e:.3} of its float64 tolerance", .{r.worst});
}
