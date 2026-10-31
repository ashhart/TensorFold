//! Grouped EXL3 SwiGLU-OAI (act_mode 2) against float64; every row's bits at any row count and either grouping.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const exl3fmt = @import("core").exl3_format;
const f64t = @import("exl3_experts_f64_tests.zig");
const exl3t = @import("exl3_tests.zig");
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;

const E = f64t.E;
const D = f64t.D;
const I = f64t.I;
const SLOTS = f64t.SLOTS;
const LIMIT: f32 = 7.0; // MiniMax-M3.1-Flash's swiglu_limit and swiglu_alpha
const ALPHA: f32 = 1.702;
const ROWS = 128;
const rows = [_]usize{ 1, 3, 8, 16, 17, 64 }; // each against the same rows of a 128-row call, bit for bit

const Kit = struct {
    mod: cuda.Module,
    router: cuda.grouped.Router,
    exp: cuda.exl3_experts.Experts,
    g: cuda.exl3_grouped.Grouped,
};

pub fn run(r: *Rig, state: *exl3t.Run) !void {
    var k: Kit = .{ .mod = try cuda.Module.load(r.gpu.d, cuda.kernels.experts), .router = undefined, .exp = undefined, .g = undefined };
    defer k.mod.unload();
    k.router = try cuda.grouped.Router.resolve(k.mod);
    k.exp = try cuda.exl3_experts.Experts.load(r.gpu.d);
    defer k.exp.unload();
    k.g = .{ .d = r.gpu.d, .kernels = &k.exp };
    defer k.g.deinit();
    // the builder's small scales keep the gate near 0; svh x 64 drives it past the clamp and into the sigmoid's tails
    for ([_]f32{ 1, 64 }, [_][]const u8{ "exl3.grouped.oai.f64", "exl3.grouped.oai.f64.clamped" }) |scale, id| {
        f64Up(r, &k, scale) catch |e| {
            state.fail(e, id);
            continue;
        };
        check.pass("{s}", .{id});
    }
    rowsUp(r, &k) catch |e| {
        state.fail(e, "exl3.grouped.oai.rows");
        return;
    };
    check.pass("exl3.grouped.oai.rows", .{});
}

/// The builder's layer as SwiGLU-OAI, its output scales times `scale`.
fn oaiLayer(r: *Rig, scale: f32) !struct { b: f64t.Built, l: cuda.experts.Layer } {
    var b = try f64t.buildLayer(r, &.{ 3, 4, 5, 6, 3, 4, 5, 6, 3 });
    for ([_][]u16{ b.svh_g, b.svh_u }) |sv| for (sv) |*v| {
        v.* = @bitCast(@as(f16, @floatCast(@as(f32, @floatCast(@as(f16, @bitCast(v.*)))) * scale)));
    };
    b.layer.svh_g = try r.dev(std.mem.sliceAsBytes(b.svh_g));
    b.layer.svh_u = try r.dev(std.mem.sliceAsBytes(b.svh_u));
    b.layer.limit = LIMIT;
    b.layer.alpha = ALPHA;
    b.layer.act_mode = 2;
    const l: cuda.experts.Layer = .{ .format = .exl3, .exl3 = b.layer, .up = 0, .down = 0, .width = I, .dims = D, .experts = E, .limit = LIMIT, .swiglu = true };
    return .{ .b = b, .l = l };
}

/// The grouped up of the first `m` rows, pairs grouped `tile` at a time: bf16 [m * SLOTS][I], each pair's activation.
fn up(r: *Rig, k: *Kit, l: cuda.experts.Layer, x: u64, pick: u64, m: usize, tile: usize) ![]const u8 {
    const pairs = m * SLOTS;
    const items = cuda.grouped.maxItems(pairs, E, tile);
    const plan: cuda.grouped.Plan = .{
        .members = try r.zeros(pairs * 4 + 16),
        .items = try r.zeros(items * 3 * 4 + 16),
        .counts = try r.zeros(16),
        .rank = try r.zeros(pairs * 4 + 16),
        .hist = try r.zeros((E + 1) * 4 * ((pairs + 1023) / 1024 + 1)),
    };
    try k.router.route(r.s, pick, pairs, E, tile, plan);
    const out = try r.zeros(pairs * I * 2);
    try k.g.run(r.s, .up, .{ .x = x, .stride = D, .slots = SLOTS }, l, plan, out, items, -1, false);
    return r.back(out, pairs * I * 2);
}

fn f64Up(r: *Rig, k: *Kit, scale: f32) !void {
    const o = try oaiLayer(r, scale);
    const b = &o.b;
    const got = try up(r, k, o.l, b.dx, b.dpick, ROWS, 16);
    const p = ROWS * SLOTS;
    // the rotated inputs, gate's and up's own: fp16((x * suh) @ H), each within its fp16 and rotation roundings
    var xq: [2][]f16 = undefined;
    var xe: [2][]f64 = undefined;
    for ([_][]const u16{ b.suh_g, b.suh_u }, 0..) |suh, mat| {
        const xs = try r.a.alloc(f64, p * D);
        for (0..p) |s0| {
            const ex: usize = @intCast(b.pick[s0]);
            for (0..D) |i| xs[s0 * D + i] = bf(b.x[(s0 / SLOTS) * D + i]) * f16v(suh[ex * D + i]);
        }
        exl3fmt.rotate(xs, p, D, 1);
        xq[mat] = try r.a.alloc(f16, p * D);
        for (xq[mat], xs) |*q, v| q.* = @floatCast(v);
        xe[mat] = try r.a.alloc(f64, p * D);
        for (xe[mat], xs) |*e, v| e.* = @abs(v);
        f64t.absRotateCols(xe[mat], p, D);
        for (xe[mat]) |*v| v.* *= 1.0 / 2048.0 + 8.0 / 16777216.0;
    }
    // the GEMMs into [n][p], their |t||w| sums and propagated input errors, then the output rotation's
    const wq = try r.a.alloc(f16, D * I);
    var z: [2][]f64 = undefined;
    var ez: [2][]f64 = undefined;
    for ([_][E][]i16{ b.gate, b.up }, 0..) |mats, mat| {
        z[mat] = try r.a.alloc(f64, I * p);
        ez[mat] = try r.a.alloc(f64, I * p);
        const s1 = try r.a.alloc(f64, I * p);
        @memset(z[mat], 0);
        @memset(ez[mat], 0);
        @memset(s1, 0);
        for (0..E) |ex| {
            exl3fmt.unpack(f64t.widthOf(b.bits[ex]), .mul1, mats[ex], D / 16, I / 16, wq);
            for (0..p) |s0| {
                if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
                f64t.gemmK(xq[mat][s0 * D ..][0..D], wq, z[mat][s0..], s1[s0..], ez[mat][s0..], xe[mat][s0 * D ..][0..D], D, I, p);
            }
        }
        for (ez[mat], s1) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
        exl3fmt.rotate(z[mat], I, p, 0);
        f64t.absRotate(ez[mat], I, p);
        const zr = try r.a.alloc(f64, I * p);
        for (zr, z[mat]) |*e, v| e.* = @abs(v);
        f64t.absRotate(zr, I, p);
        for (ez[mat], zr) |*e, v| e.* += v * 8.0 / 16777216.0;
    }
    // the kernel's bf16 steps, each step's error carried: a bf16 flip |value| / 128, exp 2^-20, clamps 1-Lipschitz
    var most: f64 = 0;
    var at: usize = 0;
    // controls (no + 1, alpha 1): summed error at least 10 times the true chain's (the bound alone is too wide)
    var sums: [3]f64 = .{ 0, 0, 0 };
    const L: f64 = LIMIT;
    const A: f64 = ALPHA;
    for (0..p) |s0| {
        const ex: usize = @intCast(b.pick[s0]);
        for (0..I) |n| {
            const sg = f16v(b.svh_g[ex * I + n]);
            const su = f16v(b.svh_u[ex * I + n]);
            const vg = z[0][n * p + s0] * sg;
            const vu = z[1][n * p + s0] * su;
            const g = @min(bf16r(vg), L);
            const u = @min(@max(bf16r(vu), -L), L);
            const ga = bf16r(g * A);
            const sig = bf16r(1.0 / (1.0 + std.math.exp(-ga)));
            const glu = bf16r(g * sig);
            const up1 = bf16r(u + 1);
            const act = up1 * glu;
            const eg = ez[0][n * p + s0] * @abs(sg) + @abs(vg) / 128;
            const eu = ez[1][n * p + s0] * @abs(su) + @abs(vu) / 128;
            const ega = A * eg + @abs(ga) / 128;
            const es = ega / 4 + @abs(sig) / 128 + @abs(sig) / 1048576;
            const eglu = @abs(sig) * eg + (@abs(g) + eg) * es + @abs(glu) / 128;
            const eup = eu + @abs(up1) / 128;
            const tol = @abs(up1) * eglu + @abs(glu) * eup + eglu * eup + @abs(act) / 128 + 1e-30;
            const v: f64 = bf(std.mem.readInt(u16, got[2 * (s0 * I + n) ..][0..2], .little));
            const q = @abs(v - act) / tol;
            if (!(q <= most)) {
                most = q;
                at = s0 * I + n;
            }
            const plain = bf16r(u) * glu;
            const one = up1 * bf16r(g * bf16r(1.0 / (1.0 + std.math.exp(-g))));
            sums[0] += @abs(v - act);
            sums[1] += @abs(v - plain);
            sums[2] += @abs(v - one);
        }
    }
    std.debug.print("RESULT exl3.grouped.oai.f64 svh x{d}: largest difference {d:.3} of its tolerance (element {d})\n", .{ scale, most, at });
    const r1 = sums[1] / @max(sums[0], 1e-300);
    const r2 = sums[2] / @max(sums[0], 1e-300);
    std.debug.print("RESULT exl3.grouped.oai.f64 svh x{d} controls: no + 1 {d:.1}x, alpha 1 {d:.1}x the true chain's summed error\n", .{ scale, r1, r2 });
    if (!(most <= 1)) return error.TestFailed;
    if (!(r1 >= 10 and r2 >= 10)) {
        std.debug.print("FAIL exl3.grouped.oai.f64: a control is within 10 times the true chain's error\n", .{});
        return error.TestFailed;
    }
}

fn rowsUp(r: *Rig, k: *Kit) !void {
    const o = try oaiLayer(r, 64);
    const whole = try up(r, k, o.l, o.b.dx, o.b.dpick, ROWS, 16);
    for (rows) |m| {
        const part = try up(r, k, o.l, o.b.dx, o.b.dpick, m, 16);
        if (!std.mem.eql(u8, part, whole[0..part.len])) {
            std.debug.print("FAIL exl3.grouped.oai.rows: {d} rows differ from the same rows of {d}\n", .{ m, ROWS });
            return error.TestFailed;
        }
    }
    // a prompt chunk groups its pairs 64 at a time, a decode window 16: the same bits
    if (!std.mem.eql(u8, try up(r, k, o.l, o.b.dx, o.b.dpick, ROWS, 64), whole)) {
        std.debug.print("FAIL exl3.grouped.oai.rows: pairs grouped by 64 differ from pairs grouped by 16\n", .{});
        return error.TestFailed;
    }
}

const f16v = f64t.f16v;
const bf16r = f64t.bf16r;
