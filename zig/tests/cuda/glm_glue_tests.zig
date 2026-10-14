//! GLM glue kernels: each against a float64 reference and the Python engine's fixture bytes (glue.py, qmm.py).
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3_tests = @import("exl3_tests.zig");
const ref = @import("glm_kda_ref.zig");
const gaps = @import("glm_gaps_tests.zig");
const glue = cuda.glm_glue;

const Run = exl3_tests.Run;
const Rig = quant_ref_tests.RigT;
const Failed = check.Failed;

pub fn run(r: *Rig, state: *Run, dir: []const u8) !void {
    embedCheck(r, state, dir) catch |e| state.fail(e, "glm.glue.embed");
    rmsnorm(r, state, dir) catch |e| state.fail(e, "glm.glue.rmsnorm");
    try rmsnormGaps(r, state, dir);
    hcPre(r, state, dir) catch |e| state.fail(e, "glm.glue.hc_pre");
    hcPost(r, state, dir) catch |e| state.fail(e, "glm.glue.hc_post");
    residual(r, state, dir) catch |e| state.fail(e, "glm.glue.residual");
    swiglu(r, state, dir) catch |e| state.fail(e, "glm.glue.swiglu");
    router(r, state, dir) catch |e| state.fail(e, "glm.glue.router");
    select(r, state, dir) catch |e| state.fail(e, "glm.glue.select");
    combine(r, state, dir) catch |e| state.fail(e, "glm.glue.combine");
    try combineGaps(r, state, dir);
    bmm(r, state, dir) catch |e| state.fail(e, "glm.bmm");
    greedy(r, state, dir) catch |e| state.fail(e, "glm.sample.greedy");
}

fn skip(id: []const u8) void {
    std.debug.print("SKIP {s}\n", .{id});
}

/// The fixture family's manifest object, or null (SKIP printed) when there is no fixture dir or family.
fn section(r: *Rig, dir: []const u8, id: []const u8) !?std.json.ObjectMap {
    if (std.mem.eql(u8, dir, "none")) {
        skip(id);
        return null;
    }
    var man = try ref.Manifest.open(r.a, r.gpu.io, dir);
    const g = ref.Manifest.root(&man).get("glue") orelse {
        skip(id);
        return null;
    };
    return g.object;
}

fn up(r: *Rig, dir: []const u8, name: []const u8) !u64 {
    return r.dev(try ref.readFile(r.a, r.gpu.io, dir, name));
}

fn f32s(bytes: []const u8) []const f32 {
    return ref.f32s(bytes);
}

/// bf16 bytes as f64 values (exact).
fn bf4(a: std.mem.Allocator, bytes: []const u8) ![]f64 {
    const u = std.mem.bytesAsSlice(u16, bytes);
    const out = try a.alloc(f64, u.len);
    for (u, 0..) |v, i| out[i] = ref.bf(v);
    return out;
}

fn sigmoid(x: f64) f64 {
    return 1.0 / (1.0 + @exp(-x));
}

pub const Tol = struct {
    worst: f64 = 0,

    pub fn f64v(t: *Tol, got: f64, want: f64, scale: f64, floor: f64) void {
        const q = @abs(got - want) / (@abs(want) * scale + floor);
        if (q > t.worst) t.worst = q;
    }

    pub fn f32v(t: *Tol, arr: []const f32, i: usize, want: f64, scale: f64, floor: f64) void {
        t.f64v(arr[i], want, scale, floor);
    }

    pub fn done(t: *Tol, name: []const u8) Failed!void {
        if (t.worst > 1.0) {
            std.debug.print("FAIL {s}: f64 worst {d:.4} of tolerance\n", .{ name, t.worst });
            return error.TestFailed;
        }
        std.debug.print("RESULT {s}: f64 worst {d:.4} of tolerance\n", .{ name, t.worst });
    }
};

// -- glm.glue.embed -----------------------------------------------------------------------------
fn embedCheck(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.embed")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const ids = try up(r, dir, ref.str(o, "ids"));
    const tab = try up(r, dir, ref.str(o, "e_tab"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "e_out"));
    const out = try r.zeros(want.len);
    try k.embedB16(r.s, ids, tab, out, rows, d, 4);
    try check.sameBytes("glm.glue.embed", try r.back(out, want.len), want);
    // f64: the rows are exact copies of the table's bf16 values.
    const idv = std.mem.bytesAsSlice(u32, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "ids")));
    const tabv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "e_tab")));
    const gotv = try bf4(r.a, try r.back(out, want.len));
    var t = Tol{};
    for (0..rows) |i| for (0..4) |c| for (0..d) |j| {
        t.f64v(gotv[(i * 4 + c) * d + j], tabv[@as(usize, @intCast(idv[i])) * d + j], 0.0, 1e-6);
    };
    try t.done("glm.glue.embed");
    check.pass("glm.glue.embed", .{});
}

// -- glm.glue.rmsnorm ---------------------------------------------------------------------------
fn rmsnorm(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.rmsnorm")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const eps: f32 = @floatCast(ref.float(o, "eps"));
    const x = try up(r, dir, ref.str(o, "r_x"));
    const w = try up(r, dir, ref.str(o, "nw"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "r_out"));
    const want_sums = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "r_sums"));
    const out = try r.zeros(want.len);
    const sums = try r.zeros(want_sums.len);
    try k.rmsnorm(r.s, x, @intCast(d), w, out, @intCast(d), sums, eps, rows, d);
    try check.sameBytes("glm.glue.rmsnorm", try r.back(out, want.len), want);
    try check.sameBytes("glm.glue.rmsnorm sums", try r.back(sums, want_sums.len), want_sums);
    const xv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "r_x")));
    const wv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "nw")));
    const gotv = try bf4(r.a, try r.back(out, want.len));
    var t = Tol{};
    for (0..rows) |i| {
        var ss: f64 = 0;
        for (0..d) |j| ss += xv[i * d + j] * xv[i * d + j];
        const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(d)) + eps);
        for (0..d) |j| t.f64v(gotv[i * d + j], wv[j] * xv[i * d + j] * rinv, 1.0 / 64.0, 3e-3);
    }
    try t.done("glm.glue.rmsnorm");
    check.pass("glm.glue.rmsnorm", .{});
}

// -- glm.glue.hc_pre ----------------------------------------------------------------------------
fn hcPre(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.hc_pre")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const eps: f32 = @floatCast(ref.float(o, "eps"));
    const hc_eps: f32 = @floatCast(ref.float(o, "hc_eps"));
    const iters: u32 = @intCast(ref.int(o, "iters"));
    const x = try up(r, dir, ref.str(o, "h_x"));
    const fn_w = try up(r, dir, ref.str(o, "fn"));
    const base = try up(r, dir, ref.str(o, "base"));
    const scale = try up(r, dir, ref.str(o, "hc_scale"));
    const nw = try up(r, dir, ref.str(o, "nw"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_out"));
    const want_xs = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_xs"));
    const want_post = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_post"));
    const want_comb = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_comb"));
    const part = try r.zeros(rows * 16 * 32 * 4);
    const out = try r.zeros(want.len);
    const xs = try r.zeros(want_xs.len);
    const post = try r.zeros(want_post.len);
    const comb = try r.zeros(want_comb.len);
    try k.hcPartial(r.s, x, fn_w, part, rows);
    try k.hcFinish(r.s, x, part, base, scale, nw, out, xs, post, comb, eps, hc_eps, iters, rows, d);
    try check.sameBytes("glm.glue.hc_pre", try r.back(out, want.len), want);
    try check.sameBytes("glm.glue.hc_pre xs", try r.back(xs, want_xs.len), want_xs);
    try check.sameBytes("glm.glue.hc_pre post", try r.back(post, want_post.len), want_post);
    try check.sameBytes("glm.glue.hc_pre part", try r.back(part, rows * 16 * 32 * 4), try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_part")));
    try check.sameBytes("glm.glue.hc_pre comb", try r.back(comb, want_comb.len), want_comb);
    // f64: the dots, gates, sinkhorn and collapsed row in double precision.
    const xv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_x")));
    const fnv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "fn")));
    const basev = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "base")));
    const scv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "hc_scale")));
    const nwv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "nw")));
    const wide = 4 * d;
    const gotv = try bf4(r.a, try r.back(out, want.len));
    const post_got = f32s(try r.back(post, want_post.len));
    const comb_got = f32s(try r.back(comb, want_comb.len));
    var t = Tol{};
    for (0..rows) |i| {
        var mix: [24]f64 = @splat(0);
        var ss: f64 = 0;
        for (0..wide) |j| {
            const xj: f64 = xv[i * wide + j];
            ss += xj * xj;
            for (0..24) |m| mix[m] += fnv[m * wide + j] * xj;
        }
        const rinv = 1.0 / @sqrt(ss / @as(f64, @floatFromInt(wide)) + eps);
        var p: [4]f64 = undefined;
        for (0..4) |sv| {
            p[sv] = sigmoid(mix[sv] * rinv * scv[0] + basev[sv]) + hc_eps;
            t.f32v(post_got, i * 4 + sv, 2.0 * sigmoid(mix[4 + sv] * rinv * scv[1] + basev[4 + sv]), 1.0 / 1024.0, 1e-6);
        }
        var ce: [4][4]f64 = undefined;
        for (0..4) |a| {
            var row: [4]f64 = undefined;
            var cmax: f64 = -std.math.inf(f64);
            for (0..4) |b| {
                row[b] = mix[8 + a * 4 + b] * rinv * scv[2] + basev[8 + a * 4 + b];
                cmax = @max(cmax, row[b]);
            }
            for (0..4) |b| ce[a][b] = @exp(row[b] - cmax);
        }
        for (0..4) |a| {
            var rs: f64 = 0;
            for (0..4) |b| rs += ce[a][b];
            for (0..4) |b| ce[a][b] = ce[a][b] / rs + hc_eps;
        }
        for (0..4) |b| {
            var cs: f64 = 0;
            for (0..4) |a| cs += ce[a][b];
            for (0..4) |a| ce[a][b] = ce[a][b] / (cs + hc_eps);
        }
        for (1..iters) |_| {
            for (0..4) |a| {
                var rs: f64 = 0;
                for (0..4) |b| rs += ce[a][b];
                for (0..4) |b| ce[a][b] = ce[a][b] / (rs + hc_eps);
            }
            for (0..4) |b| {
                var cs: f64 = 0;
                for (0..4) |a| cs += ce[a][b];
                for (0..4) |a| ce[a][b] = ce[a][b] / (cs + hc_eps);
            }
        }
        for (0..4) |a| for (0..4) |b| t.f32v(comb_got, i * 16 + a * 4 + b, ce[a][b], 1.0 / 512.0, 1e-6);
        // the collapsed row and its norm
        const c = try r.a.alloc(f64, d);
        for (0..d) |j| c[j] = p[0] * xv[i * wide + j] + p[1] * xv[i * wide + d + j] + p[2] * xv[i * wide + 2 * d + j] + p[3] * xv[i * wide + 3 * d + j];
        var cs2: f64 = 0;
        for (0..d) |j| cs2 += c[j] * c[j];
        const rinv2 = 1.0 / @sqrt(cs2 / @as(f64, @floatFromInt(d)) + eps);
        for (0..d) |j| t.f64v(gotv[i * d + j], nwv[j] * c[j] * rinv2, 1.0 / 64.0, 3e-3);
    }
    try t.done("glm.glue.hc_pre");
    check.pass("glm.glue.hc_pre", .{});
}

// -- glm.glue.hc_post ---------------------------------------------------------------------------
fn hcPost(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.hc_post")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const x = try up(r, dir, ref.str(o, "p_x"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "p_out"));
    const want_pair = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "p_pair_out"));
    const post = try up(r, dir, ref.str(o, "h_post"));
    const comb = try up(r, dir, ref.str(o, "h_comb"));
    const g_all = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "p_g"));
    const xout = try r.zeros(want.len);
    try k.hcPost(r.s, x, xout, try r.dev(g_all), post, comb, @intCast(rows * d), 2, rows, d, 4);
    try check.sameBytes("glm.glue.hc_post", try r.back(xout, want.len), want);
    // the pair form: the two ranks' partials held apart, rank 1 read at its element distance
    const row_bytes = g_all.len / 2;
    const g0 = try r.dev(g_all[0..row_bytes]);
    const g1 = try r.dev(g_all[row_bytes..]);
    const pout = try r.zeros(want_pair.len);
    try k.hcPost(r.s, x, pout, g0, post, comb, @intCast((g1 - g0) / 4), 2, rows, d, 4);
    try check.sameBytes("glm.glue.hc_post pair", try r.back(pout, want_pair.len), want_pair);
    // f64: the branch sum, the gate mix
    const xv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "p_x")));
    const gv = f32s(g_all);
    const postv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_post")));
    const combv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "h_comb")));
    const gotv = try bf4(r.a, try r.back(xout, want.len));
    var t = Tol{};
    for (0..rows) |i| for (0..4) |sv| for (0..d) |j| {
        const branch: f64 = bf16f64(gv[i * d + j] + gv[(rows + i) * d + j]);
        var mixed: f64 = combv[i * 16 + 0 * 4 + sv] * xv[(i * 4 + 0) * d + j];
        mixed += combv[i * 16 + 1 * 4 + sv] * xv[(i * 4 + 1) * d + j];
        mixed += combv[i * 16 + 2 * 4 + sv] * xv[(i * 4 + 2) * d + j];
        mixed += combv[i * 16 + 3 * 4 + sv] * xv[(i * 4 + 3) * d + j];
        t.f64v(gotv[(i * 4 + sv) * d + j], postv[i * 4 + sv] * branch + mixed, 1.0 / 64.0, 3e-3);
    };
    try t.done("glm.glue.hc_post");
    check.pass("glm.glue.hc_post", .{});
}

// -- glm.glue.residual (residual_add and stream_mean) --------------------------------------------
fn residual(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.residual")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const x = try up(r, dir, ref.str(o, "res_x"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "res_out"));
    const g_all = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "p_g"));
    const xout = try r.zeros(want.len);
    try k.residualAdd(r.s, x, xout, try r.dev(g_all), @intCast(rows * d), 2, rows, d);
    try check.sameBytes("glm.glue.residual", try r.back(xout, want.len), want);
    const smx = try up(r, dir, ref.str(o, "sm_x"));
    const want_sm = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "sm_out"));
    const smout = try r.zeros(want_sm.len);
    try k.streamMean(r.s, smx, smout, rows, d, 4);
    try check.sameBytes("glm.glue.residual mean", try r.back(smout, want_sm.len), want_sm);
    // f64
    const xv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "res_x")));
    const gv = f32s(g_all);
    const gotv = try bf4(r.a, try r.back(xout, want.len));
    var t = Tol{};
    for (0..rows * d) |i| t.f64v(gotv[i], bf16f64(xv[i] + bf16f64(gv[i] + gv[rows * d + i])), 1.0 / 64.0, 3e-3);
    const smv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "sm_x")));
    const got_sm = try bf4(r.a, try r.back(smout, want_sm.len));
    for (0..rows) |i| for (0..d) |j| {
        var acc: f64 = 0;
        for (0..4) |sv| acc += smv[(i * 4 + sv) * d + j];
        t.f64v(got_sm[i * d + j], acc / 4.0, 1.0 / 64.0, 3e-3);
    };
    try t.done("glm.glue.residual");
    check.pass("glm.glue.residual", .{});
}

// -- glm.glue.swiglu -----------------------------------------------------------------------------
fn swiglu(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.swiglu")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const w: usize = 6144;
    const lim: f32 = @floatCast(ref.float(o, "lim"));
    const gu = try up(r, dir, ref.str(o, "s_gu"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "s_out"));
    const want_xs = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "s_xs"));
    const out = try r.zeros(want.len);
    const xs = try r.zeros(want_xs.len);
    try k.swiglu(r.s, gu, out, xs, lim, rows, w);
    try check.sameBytes("glm.glue.swiglu", try r.back(out, want.len), want);
    try check.sameBytes("glm.glue.swiglu xs", try r.back(xs, want_xs.len), want_xs);
    const guv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "s_gu")));
    const gotv = try bf4(r.a, try r.back(out, want.len));
    var t = Tol{};
    for (0..rows) |i| for (0..w) |j| {
        const g = @min(guv[i * 2 * w + j], lim);
        const u = @min(@max(guv[i * 2 * w + w + j], -lim), lim);
        t.f64v(gotv[i * w + j], bf16f64(bf16f64(g / (1.0 + @exp(-g))) * u), 1.0 / 64.0, 3e-3);
    };
    try t.done("glm.glue.swiglu");
    check.pass("glm.glue.swiglu", .{});
}

// -- glm.glue.router -----------------------------------------------------------------------------
fn router(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.router")) orelse return;
    _ = state;
    var k = try cuda.glm_bmm.Bmm.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const ne: usize = @intCast(ref.int(o, "ne"));
    const x = try up(r, dir, ref.str(o, "router_x"));
    const w = try up(r, dir, ref.str(o, "router_w"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "router_out"));
    const part = try r.zeros(8 * rows * ne * 4);
    const out = try r.zeros(want.len);
    try k.router(r.s, x, @intCast(d), w, rows, ne, d, out, part);
    try check.sameBytes("glm.glue.router", try r.back(out, want.len), want);
    const xv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "router_x")));
    const wv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "router_w")));
    const gotv = f32s(try r.back(out, want.len));
    var t = Tol{};
    for (0..rows) |i| for (0..ne) |e| {
        var acc: f64 = 0;
        for (0..d) |j| acc += xv[i * d + j] * wv[e * d + j];
        t.f32v(gotv, i * ne + e, acc, 1.0 / 1024.0, 1e-5);
    };
    try t.done("glm.glue.router");
    check.pass("glm.glue.router", .{});
}

// -- glm.glue.select -----------------------------------------------------------------------------
fn select(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.select")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const ne: usize = @intCast(ref.int(o, "ne"));
    const topk: usize = @intCast(ref.int(o, "topk"));
    const scale: f32 = @floatCast(ref.float(o, "scale"));
    const lg = try up(r, dir, ref.str(o, "sel_logits"));
    const bias = try up(r, dir, ref.str(o, "sel_bias"));
    const want_pick = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "sel_pick"));
    const want_wts = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "sel_wts"));
    const pick = try r.zeros(want_pick.len);
    const wts = try r.zeros(want_wts.len);
    try k.topk(r.s, lg, bias, pick, wts, scale, rows, ne, topk);
    try check.sameBytes("glm.glue.select", try r.back(pick, want_pick.len), want_pick);
    try check.sameBytes("glm.glue.select wts", try r.back(wts, want_wts.len), want_wts);
    // f64: the same picks by biased score, lowest id on ties, then the normalized weights
    const lgv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "sel_logits")));
    const bv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "sel_bias")));
    const got_wts = f32s(try r.back(wts, want_wts.len));
    var t = Tol{};
    for (0..rows) |i| {
        const score = try r.a.alloc(f64, ne);
        const used = try r.a.alloc(bool, ne);
        var total: f64 = 0;
        for (0..ne) |e| {
            score[e] = sigmoid(lgv[i * ne + e]);
            used[e] = false;
        }
        const picks = try r.a.alloc(usize, topk);
        for (0..topk) |kk| {
            var best: f64 = -std.math.inf(f64);
            var idx: usize = ne;
            for (0..ne) |e| if (!used[e] and score[e] + bv[e] > best) {
                best = score[e] + bv[e];
                idx = e;
            };
            used[idx] = true;
            picks[kk] = idx;
            total += score[idx];
        }
        for (0..topk) |kk| t.f32v(got_wts, i * (topk + 1) + kk, score[picks[kk]] / (total + 1e-20) * scale, 1.0 / 1024.0, 1e-6);
    }
    try t.done("glm.glue.select");
    check.pass("glm.glue.select", .{});
}

fn bf16f64(v: f64) f64 {
    const u: u32 = @bitCast(@as(f32, @floatCast(v)));
    const ru: u32 = u +% (0x7FFF + ((u >> 16) & 1));
    return @as(f64, @as(f32, @bitCast(ru & 0xFFFF0000)));
}

// -- glm.glue.combine ----------------------------------------------------------------------------
fn combine(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.glue.combine")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const y = try up(r, dir, ref.str(o, "c_y"));
    const wts = try up(r, dir, ref.str(o, "c_wts"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "c_out"));
    const out = try r.zeros(want.len);
    try k.combine(r.s, y, wts, out, rows, d, 9);
    try check.sameBytes("glm.glue.combine", try r.back(out, want.len), want);
    std.debug.print("DBG c_y={s} c_wts={s} rows={d} d={d}\n", .{ ref.str(o, "c_y"), ref.str(o, "c_wts"), rows, d });
    const yv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "c_y")));
    std.debug.print("DBG yv len={d}\n", .{yv.len});
    const wv = f32s(try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "c_wts")));
    std.debug.print("DBG wv len={d} ptr={*}\n", .{ wv.len, wv.ptr });
    const gotv = f32s(try r.back(out, want.len));
    var t = Tol{};
    for (0..rows) |i| for (0..d) |j| {
        var acc: f64 = 0;
        for (0..9) |kk| acc += wv[i * 9 + kk] * yv[(i * 9 + kk) * d + j];
        t.f32v(gotv, i * d + j, acc, 1.0 / 1024.0, 1e-5);
    };
    try t.done("glm.glue.combine");
    check.pass("glm.glue.combine", .{});
}

// -- glm.bmm -------------------------------------------------------------------------------------
fn bmm(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.bmm")) orelse return;
    _ = state;
    var k = try cuda.glm_bmm.Bmm.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = @intCast(ref.int(o, "d"));
    const x = try up(r, dir, ref.str(o, "bmm_x"));
    const w = try up(r, dir, ref.str(o, "bmm_w"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "bmm_out"));
    const w16 = cuda.glm_bmm.B16{ .w = w, .n = 4096, .k = @intCast(d) };
    const part = try r.zeros(cuda.glm_bmm.Bmm.partBytes(w16, rows));
    const out = try r.zeros(want.len);
    try k.matmul(r.s, w16, x, @intCast(d), rows, out, false, part);
    try check.sameBytes("glm.bmm", try r.back(out, want.len), want);
    const xv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "bmm_x")));
    const wv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "bmm_w")));
    const gotv = try bf4(r.a, try r.back(out, want.len));
    var t = Tol{};
    for (0..rows) |i| for (0..4096) |e| {
        var acc: f64 = 0;
        for (0..d) |j| acc += xv[i * d + j] * wv[e * d + j];
        t.f64v(gotv[i * 4096 + e], acc, 1.0 / 64.0, 3e-3);
    };
    try t.done("glm.bmm");
    check.pass("glm.bmm", .{});
}

// -- glm.sample.greedy ---------------------------------------------------------------------------
fn greedy(r: *Rig, state: *Run, dir: []const u8) !void {
    const o = (try section(r, dir, "glm.sample.greedy")) orelse return;
    _ = state;
    var k = try glue.Glue.load(r.gpu.d);
    defer k.unload();
    const rows: usize = @intCast(ref.int(o, "rows"));
    const d: usize = 4096;
    const lg = try up(r, dir, ref.str(o, "g_logits"));
    const want = try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "g_tok"));
    const tok = try r.zeros(want.len);
    try k.greedy(r.s, lg, tok, rows, d);
    try check.sameBytes("glm.sample.greedy", try r.back(tok, want.len), want);
    // f64: the highest logit, lowest id on ties
    const lgv = try bf4(r.a, try ref.readFile(r.a, r.gpu.io, dir, ref.str(o, "g_logits")));
    const got = std.mem.bytesAsSlice(i32, try r.back(tok, want.len));
    for (0..rows) |i| {
        var best: f64 = lgv[i * d];
        var bi: usize = 0;
        for (0..d) |j| if (lgv[i * d + j] > best) {
            best = lgv[i * d + j];
            bi = j;
        };
        if (got[i] != @as(i32, @intCast(bi))) return error.TestFailed;
    }
    check.pass("glm.sample.greedy", .{});
}

// -- the gap fixtures (glm_oracle_gaps.py's "gaps"): folded into the rmsnorm and combine ids above ---

fn rmsnormGaps(r: *Rig, state: *Run, dir: []const u8) !void {
    gaps.rmsnorm(r, dir) catch |e| state.fail(e, "glm.glue.rmsnorm");
}

fn combineGaps(r: *Rig, state: *Run, dir: []const u8) !void {
    gaps.combine(r, dir) catch |e| state.fail(e, "glm.glue.combine");
    gaps.combineMoe(r, dir) catch |e| state.fail(e, "glm.glue.combine");
}
