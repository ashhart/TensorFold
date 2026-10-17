//! The EXL3 routed experts' float64 reference: the mirrored chain and each output's worst-case error bound.
const std = @import("std");
const check = @import("check.zig");
const quant_ref_tests = @import("quant_ref_tests.zig");
const cuda = @import("cuda");
const exl3fmt = @import("core").exl3_format;
const exl3e = cuda.exl3_experts;
const exl3t = @import("exl3_tests.zig");
const Rig = quant_ref_tests.RigT;
const bf = quant_ref_tests.helpers.bfOf;
const Run = exl3t.Run;

pub const E = 9; // experts of a synthetic layer
pub const D = 512; // model width
const I = 256; // expert width
pub const SLOTS = 8;
const LIMIT: f32 = 10.0; // the SwiGLU clamp
const CAP = 300; // rows of x, pick and wts the builders make (the f64 checks use the first 128)
const rows = [_]usize{ 1, 3, 8, 16, 17, 64, 128 }; // the f64 checks' row counts (the reference builds 128 rows)

const f64_ids = [_][]const u8{ "exl3.experts.f64.mul1.4", "exl3.experts.f64.mul1.5", "exl3.experts.f64.mul1.6" };

/// One synthetic layer: the device tensors, the host trellises the reference unpacks, and the seeded routing.
pub const Built = struct {
    layer: exl3e.Layer,
    bits: []const u32,
    gate: [E][]i16,
    up: [E][]i16,
    down: [E][]i16,
    suh_g: []u16, // fp16 [E, D]
    suh_u: []u16,
    svh_g: []u16, // fp16 [E, I]
    svh_u: []u16,
    suh_d: []u16, // fp16 [E, I]
    svh_d: []u16, // fp16 [E, D]
    dx: u64,
    dpick: u64,
    dwts: u64,
    x: []u16, // bf16 [CAP, D]
    pick: []i32, // [CAP, SLOTS]
    wts: []f32, // [CAP, SLOTS]
};

/// The f64 id checks: every width alone, then the mixed layer folded into the .6 id (the contract's ids).
pub fn checks(r: *Rig, exp: *const exl3e.Experts, state: *Run) void {
    for (f64_ids, 0..) |id, i| {
        var ok = true;
        f64Check(r, exp, &allBits(@intCast(i + 4))) catch |e| {
            state.fail(e, id);
            ok = false;
        };
        if (ok) check.pass("{s}", .{id});
    }
    {
        var ok = true;
        f64Check(r, exp, &.{ 4, 5, 6, 4, 5, 6, 4, 5, 6 }) catch |e| {
            state.fail(e, f64_ids[2]);
            ok = false;
        };
        if (ok) check.pass("{s}", .{f64_ids[2]});
    }
}

fn allBits(bits: u32) [E]u32 {
    return .{ bits, bits, bits, bits, bits, bits, bits, bits, bits };
}

fn widthOf(bits: u32) exl3fmt.Width {
    return .{ .halves = @intCast(2 * bits) };
}

fn f16v(v: u16) f64 {
    return @floatCast(@as(f16, @bitCast(v)));
}

/// bf16 round-to-nearest-even of an f64 (the kernel rounds its fp32 value; the double round is within the tolerance).
fn bf16r(v: f64) f64 {
    const u: u32 = @bitCast(@as(f32, @floatCast(v)));
    return bf(@intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16));
}

pub fn buildLayer(r: *Rig, bits: []const u32) !Built {
    var b: Built = .{
        .layer = undefined,
        .bits = bits,
        .gate = undefined,
        .up = undefined,
        .down = undefined,
        .dx = 0,
        .dpick = 0,
        .dwts = 0,
        .suh_g = undefined,
        .suh_u = undefined,
        .svh_g = undefined,
        .svh_u = undefined,
        .suh_d = undefined,
        .svh_d = undefined,
        .x = undefined,
        .pick = undefined,
        .wts = undefined,
    };
    var gate_p: [E]u64 = undefined;
    var up_p: [E]u64 = undefined;
    var down_p: [E]u64 = undefined;
    var gk2: [E]i32 = undefined;
    var uk2: [E]i32 = undefined;
    var dk2: [E]i32 = undefined;
    const suh_g = try r.a.alloc(u16, E * D);
    const suh_u = try r.a.alloc(u16, E * D);
    const svh_g = try r.a.alloc(u16, E * I);
    const svh_u = try r.a.alloc(u16, E * I);
    const suh_d = try r.a.alloc(u16, E * I);
    const svh_d = try r.a.alloc(u16, E * D);
    for (0..E) |e| {
        b.gate[e] = try randI16(r, (D / 16) * (I / 16) * widthOf(bits[e]).tileWords());
        b.up[e] = try randI16(r, (D / 16) * (I / 16) * widthOf(bits[e]).tileWords());
        b.down[e] = try randI16(r, (I / 16) * (D / 16) * widthOf(bits[e]).tileWords());
        gate_p[e] = try r.dev(std.mem.sliceAsBytes(b.gate[e]));
        up_p[e] = try r.dev(std.mem.sliceAsBytes(b.up[e]));
        down_p[e] = try r.dev(std.mem.sliceAsBytes(b.down[e]));
        gk2[e] = @intCast(2 * bits[e]);
        uk2[e] = gk2[e];
        dk2[e] = gk2[e];
        randScale(r, suh_g[e * D ..][0..D]);
        randScale(r, suh_u[e * D ..][0..D]);
        randScale(r, svh_g[e * I ..][0..I]);
        randScale(r, svh_u[e * I ..][0..I]);
        randScale(r, suh_d[e * I ..][0..I]);
        randScale(r, svh_d[e * D ..][0..D]);
    }
    b.suh_g = suh_g;
    b.suh_u = suh_u;
    b.svh_g = svh_g;
    b.svh_u = svh_u;
    b.suh_d = suh_d;
    b.svh_d = svh_d;
    b.layer = .{
        .gate = try r.dev(std.mem.sliceAsBytes(gate_p[0..])),
        .up = try r.dev(std.mem.sliceAsBytes(up_p[0..])),
        .down = try r.dev(std.mem.sliceAsBytes(down_p[0..])),
        .gate_k2 = try r.dev(std.mem.sliceAsBytes(gk2[0..])),
        .up_k2 = try r.dev(std.mem.sliceAsBytes(uk2[0..])),
        .down_k2 = try r.dev(std.mem.sliceAsBytes(dk2[0..])),
        .suh_g = try r.dev(std.mem.sliceAsBytes(suh_g)),
        .suh_u = try r.dev(std.mem.sliceAsBytes(suh_u)),
        .svh_g = try r.dev(std.mem.sliceAsBytes(svh_g)),
        .svh_u = try r.dev(std.mem.sliceAsBytes(svh_u)),
        .suh_d = try r.dev(std.mem.sliceAsBytes(suh_d)),
        .svh_d = try r.dev(std.mem.sliceAsBytes(svh_d)),
        .experts = E,
        .dims = D,
        .width = I,
        .codebook = .mul1,
        .limit = LIMIT,
        .act_mode = 0,
    };
    b.x = try r.bfs(CAP * D, 0.5); // half-range x, with the small scales, keeps the SwiGLU inputs moderate
    b.pick = try picks(r, CAP);
    b.wts = try wtsOf(r, CAP);
    b.dx = try r.dev(std.mem.sliceAsBytes(b.x));
    b.dpick = try r.dev(std.mem.sliceAsBytes(b.pick));
    b.dwts = try r.dev(std.mem.sliceAsBytes(b.wts));
    return b;
}

fn randI16(r: *Rig, n: usize) ![]i16 {
    const out = try r.a.alloc(i16, n);
    for (out) |*v| v.* = r.rng.int(i16);
    return out;
}

/// Small scales (with the half-range x): the |t||w| sums and the SwiGLU inputs stay moderate for the bound.
fn randScale(r: *Rig, out: []u16) void {
    for (out) |*v| v.* = @bitCast(@as(f16, @floatCast(r.rng.float(f32) * 0.25 - 0.125)));
}

/// Distinct experts a row: a partial Fisher-Yates over the layer's experts, seeded.
fn picks(r: *Rig, cap: usize) ![]i32 {
    const out = try r.a.alloc(i32, cap * SLOTS);
    for (0..cap) |row| {
        var idx: [E]u32 = undefined;
        for (0..E, 0..) |_, i| idx[i] = @intCast(i);
        var n: usize = E;
        for (0..SLOTS) |k| {
            const j = r.rng.uintLessThan(usize, n);
            out[row * SLOTS + k] = @intCast(idx[j]);
            idx[j] = idx[n - 1];
            n -= 1;
        }
    }
    return out;
}

fn wtsOf(r: *Rig, cap: usize) ![]f32 {
    const out = try r.a.alloc(f32, cap * SLOTS);
    for (out) |*v| v.* = 0.1 + 0.9 * r.rng.float(f32);
    return out;
}

/// One routed call over `rows` rows starting at row 0 of the built routing; the scratch and out are fresh canaries.
pub fn runRows(r: *Rig, exp: *const exl3e.Experts, b: *const Built, rows_n: usize) !u64 {
    return (try runSlice(r, exp, b.layer, b.dx, b.dpick, b.dwts, rows_n)).out;
}

pub fn runSlice(r: *Rig, exp: *const exl3e.Experts, layer: exl3e.Layer, x: u64, pick: u64, wts: u64, rows_n: usize) !struct { out: u64, y: u64 } {
    const sb = try r.zeros(try exl3e.scratchBytes(layer, rows_n, SLOTS));
    const sc = try exl3e.scratch(sb, layer, rows_n, SLOTS);
    const out = try r.zeros(rows_n * D * 4);
    try exp.run(r.s, layer, x, D, .bf16, pick, wts, rows_n, out, sc);
    return .{ .out = out, .y = sc.y };
}

fn f64Check(r: *Rig, exp: *const exl3e.Experts, bits: []const u32) !void {
    const b = try buildLayer(r, bits);
    const want = try reference(r, &b, 128);
    for (rows) |m| try closeOut(r, b.bits, m, (try runRows(r, exp, &b, m)), want.want, want.tol);
}

/// fp32 outputs within the f64 reference's tolerance, with the exact FAIL line.
fn closeOut(r: *Rig, bits: []const u32, m: usize, ptr: u64, want: []const f64, tol: []const f64) !void {
    const raw = try r.back(ptr, m * D * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (want[0 .. m * D], tol[0 .. m * D], 0..) |wv, t, i| {
        const g: f64 = @floatCast(std.mem.bytesToValue(f32, raw[4 * i ..][0..4]));
        const q = @abs(g - wv) / t;
        if (q > most) {
            most = q;
            at = i;
        }
    }
    std.debug.print("RESULT exl3.experts.f64.mul1.{d} m={d}: largest difference {d:.3} of its tolerance (element {d})\n", .{ bits[0], m, most, at });
    if (most > 1) {
        std.debug.print("FAIL exl3.experts.f64.mul1.{d}: largest difference {d:.3} of its tolerance\n", .{ bits[0], most });
        return error.TestFailed;
    }
}

/// A 300-row chunk against the f64 reference (the mixed-width layer).
pub fn prompt(r: *Rig, exp: *const exl3e.Experts) !void {
    var bits: [E]u32 = .{ 4, 5, 6, 4, 5, 6, 4, 5, 6 };
    const b = try buildLayer(r, &bits);
    const want = try reference(r, &b, 300);
    const res = try runRows(r, exp, &b, 300);
    const raw = try r.back(res, 300 * D * 4);
    var most: f64 = 0;
    var at: usize = 0;
    for (want.want, want.tol, 0..) |wv, t, i| {
        const g: f64 = @floatCast(std.mem.bytesToValue(f32, raw[4 * i ..][0..4]));
        const q = @abs(g - wv) / t;
        if (q > most) {
            most = q;
            at = i;
        }
    }
    std.debug.print("RESULT exl3.prompt.experts.f64: largest difference {d:.3} of its tolerance (element {d})\n", .{ most, at });
    if (most > 1) return error.TestFailed;
}

/// The f64 reference: the kernel's chain mirrored in f64, each output's error its per-element worst case.
fn reference(r: *Rig, b: *const Built, rows_n: usize) !struct { want: []f64, tol: []f64 } {
    const p = rows_n * SLOTS;
    // xg: fp16((x * suh_g) @ H) per slot; |xg| and its error <= V, the block's sum of |x * suh| over sqrt(128)
    const xs = try r.a.alloc(f64, p * D);
    for (0..p) |s0| {
        const ex: usize = @intCast(b.pick[s0]);
        for (0..D) |i| xs[s0 * D + i] = bf(b.x[(s0 / SLOTS) * D + i]) * f16v(b.suh_g[ex * D + i]);
    }
    exl3fmt.rotate(xs, p, D, 1);
    const xg = try r.a.alloc(f16, p * D);
    for (xg, xs) |*o, v| o.* = @floatCast(v);
    const xgerr = try r.a.alloc(f64, p * D);
    for (xgerr, xs) |*o, v| o.* = @abs(v);
    absRotateCols(xgerr, p, D);
    for (xgerr) |*v| v.* *= 1.0 / 2048.0 + 8.0 / 16777216.0; // the fp16 rounding (2^-11) plus the fp32 rotation's (8 * 2^-24), both of V

    const wq = try r.a.alloc(f16, @max(D, I) * @max(D, I));
    const zg = try r.a.alloc(f64, I * p); // [n][p]
    const zu = try r.a.alloc(f64, I * p);
    const s1g = try r.a.alloc(f64, I * p);
    const s1u = try r.a.alloc(f64, I * p);
    const eg = try r.a.alloc(f64, I * p); // the GEMM's propagated input error, before the output rotation
    const eu = try r.a.alloc(f64, I * p);
    @memset(zg, 0);
    @memset(zu, 0);
    @memset(s1g, 0);
    @memset(s1u, 0);
    @memset(eg, 0);
    @memset(eu, 0);
    for (0..E) |ex| {
        exl3fmt.unpack(widthOf(b.bits[ex]), .mul1, b.gate[ex], D / 16, I / 16, wq[0 .. D * I]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            gemmK(xg[s0 * D ..][0..D], wq, zg[s0..], s1g[s0..], eg[s0..], xgerr[s0 * D ..][0..D], D, I, p);
        }
        exl3fmt.unpack(widthOf(b.bits[ex]), .mul1, b.up[ex], D / 16, I / 16, wq[0 .. D * I]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            gemmK(xg[s0 * D ..][0..D], wq, zu[s0..], s1u[s0..], eu[s0..], xgerr[s0 * D ..][0..D], D, I, p);
        }
    }
    // the fp16 rounding (2^-11) and the fp32 accumulation (2^-17 of 128 roundings), both on the |t||w| sum
    for (eg, s1g) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    for (eu, s1u) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    exl3fmt.rotate(zg, I, p, 0);
    exl3fmt.rotate(zu, I, p, 0);
    // the output Hadamard: worst case is the block's sum of |errors| over sqrt(128), plus the fp32 rotation's own
    absRotate(eg, I, p);
    absRotate(eu, I, p);
    const zgerr = try r.a.alloc(f64, I * p);
    for (zgerr, zg) |*o, v| o.* = @abs(v);
    absRotate(zgerr, I, p);
    for (zgerr) |*v| v.* *= 8.0 / 16777216.0;
    const zuerr = try r.a.alloc(f64, I * p);
    for (zuerr, zu) |*o, v| o.* = @abs(v);
    absRotate(zuerr, I, p);
    for (zuerr) |*v| v.* *= 8.0 / 16777216.0;
    for (eg, zgerr) |*e, v| e.* += v;
    for (eu, zuerr) |*e, v| e.* += v;

    // the SwiGLU (bf16 roundings, f64 exp): g's error is the rotated GEMM error through svh plus a bf16 double rounding
    const v = try r.a.alloc(f64, I * p);
    const verr = try r.a.alloc(f64, I * p);
    for (0..I) |n| {
        for (0..p) |s0| {
            const ex: usize = @intCast(b.pick[s0]);
            const vg = zg[n * p + s0] * f16v(b.svh_g[ex * I + n]);
            const gg = @min(bf16r(vg), @as(f64, LIMIT));
            const vu = zu[n * p + s0] * f16v(b.svh_u[ex * I + n]);
            const uu = @min(@max(bf16r(vu), -@as(f64, LIMIT)), @as(f64, LIMIT));
            const inner = bf16r(gg / (1.0 + std.math.exp(-gg)));
            const act = bf16r(inner * uu);
            v[n * p + s0] = act * f16v(b.suh_d[ex * I + n]);
            const egk = eg[n * p + s0] * @abs(f16v(b.svh_g[ex * I + n])) + (@abs(vg) + @abs(gg)) / 256;
            const euk = eu[n * p + s0] * @abs(f16v(b.svh_u[ex * I + n])) + (@abs(vu) + @abs(uu)) / 256;
            const einner = egk + (@abs(gg) + 1) / 524288 + @abs(inner) / 256;
            verr[n * p + s0] = (@abs(uu) * einner + @abs(inner) * euk + @abs(act) / 256) * @abs(f16v(b.suh_d[ex * I + n])) +
                @abs(v[n * p + s0]) / 16777216; // act's: |silu'| (<= 1) g's, |silu(g)| (<= |g|) u's, expf and roundings
        }
    }
    // the down input: the fp32 rotation's roundings (of the block's sum of |v|) and the fp16 store (2^-11)
    exl3fmt.rotate(v, I, p, 0);
    absRotate(verr, I, p);
    const vrot = try r.a.alloc(f64, I * p);
    for (vrot, v) |*o, val| o.* = @abs(val);
    absRotate(vrot, I, p);
    for (vrot) |*o| o.* *= 8.0 / 16777216.0;
    const xd = try r.a.alloc(f16, p * I);
    const xderr = try r.a.alloc(f64, p * I);
    for (0..p) |s0| {
        for (0..I) |ki| {
            xd[s0 * I + ki] = @floatCast(v[ki * p + s0]);
            xderr[s0 * I + ki] = verr[ki * p + s0] + vrot[ki * p + s0] + @abs(xd[s0 * I + ki]) / 2048;
        }
    }

    // down: the GEMM, its |t| and |w| sums, and the propagated input error
    const zd = try r.a.alloc(f64, D * p); // [d][p]
    const s1d = try r.a.alloc(f64, D * p);
    const de = try r.a.alloc(f64, D * p);
    @memset(zd, 0);
    @memset(s1d, 0);
    @memset(de, 0);
    for (0..E) |ex| {
        exl3fmt.unpack(widthOf(b.bits[ex]), .mul1, b.down[ex], I / 16, D / 16, wq[0 .. I * D]);
        for (0..p) |s0| {
            if (b.pick[s0] != @as(i32, @intCast(ex))) continue;
            for (0..I) |ki| {
                const t: f64 = @floatCast(xd[s0 * I + ki]);
                const ta = @abs(t);
                const row = wq[ki * D ..][0..D];
                const xek = xderr[s0 * I + ki];
                for (0..D) |d| {
                    const wv: f64 = @floatCast(row[d]);
                    zd[d * p + s0] += t * wv;
                    s1d[d * p + s0] += ta * @abs(wv);
                    de[d * p + s0] += @abs(wv) * xek;
                }
            }
        }
    }
    // the fp16 xd rounding (2^-11 of the |t||w| sum) and the fp32 accumulation (2^-17), then the output Hadamard's mix
    for (de, s1d) |*e, s| e.* += s * (1.0 / 2048.0 + 1.0 / 131072.0);
    exl3fmt.rotate(zd, D, p, 0);
    absRotate(de, D, p);
    const zderr = try r.a.alloc(f64, D * p);
    for (zderr, zd) |*o, val| o.* = @abs(val);
    absRotate(zderr, D, p);
    for (zderr) |*o| o.* *= 8.0 / 16777216.0;
    for (de, zderr) |*e, val| e.* += val;

    // the per-slot outputs and the weighted combine: the fp32 fma chain (8 roundings of the running sum) and the store
    const want = try r.a.alloc(f64, rows_n * D);
    const tol = try r.a.alloc(f64, rows_n * D);
    for (0..rows_n) |row| {
        for (0..D) |d| {
            var acc: f32 = 0;
            var err: f64 = 0;
            var swy: f64 = 0;
            for (0..SLOTS) |k| {
                const s0 = row * SLOTS + k;
                const ex: usize = @intCast(b.pick[s0]);
                const y: f32 = @floatCast(zd[d * p + s0] * f16v(b.svh_d[ex * D + d]));
                acc = @mulAdd(f32, b.wts[row * SLOTS + k], y, acc);
                const yerr = de[d * p + s0] * @abs(f16v(b.svh_d[ex * D + d])) + @abs(y) / 16777216;
                err += @abs(b.wts[row * SLOTS + k]) * yerr;
                swy += @abs(b.wts[row * SLOTS + k] * y);
            }
            want[row * D + d] = acc;
            tol[row * D + d] = err + swy / 2097152 + @abs(want[row * D + d]) / 16777216 + 1e-30;
        }
    }
    return .{ .want = want, .tol = tol };
}

/// One output column's GEMM over k: the sums into [n][p], their |t| sums, the propagated input error.
fn gemmK(xrow: []const f16, wq: []const f16, z: []f64, s1: []f64, e: []f64, xerr: []const f64, k: usize, n: usize, p: usize) void {
    for (0..k) |ki| {
        const t: f64 = @floatCast(xrow[ki]);
        const ta = @abs(t);
        const wrow = wq[ki * n ..][0..n];
        const xek = xerr[ki];
        for (0..n) |col| {
            const wv: f64 = @floatCast(wrow[col]);
            z[col * p] += t * wv;
            s1[col * p] += ta * @abs(wv);
            e[col * p] += @abs(wv) * xek;
        }
    }
}

/// Each 128-block along the row axis replaced by the sum of its |entries| over sqrt(128) (the worst case).
fn absRotate(x: []f64, nrows: usize, cols: usize) void {
    const inv = 1.0 / @sqrt(@as(f64, 128.0));
    var r: usize = 0;
    while (r < nrows) : (r += 128) {
        for (0..cols) |c| {
            var s: f64 = 0;
            for (0..128) |n| s += @abs(x[(r + n) * cols + c]);
            s *= inv;
            for (0..128) |n| x[(r + n) * cols + c] = s;
        }
    }
}

test "the ids are the contract's" {
    try std.testing.expectEqualStrings("exl3.experts.f64.mul1.4", f64_ids[0]);
    try std.testing.expectEqualStrings("exl3.experts.f64.mul1.6", f64_ids[2]);
}

/// The column-axis twin: each 128-run along the columns replaced by the sum of its |entries| over sqrt(128).
fn absRotateCols(x: []f64, nrows: usize, cols: usize) void {
    const inv = 1.0 / @sqrt(@as(f64, 128.0));
    for (0..nrows) |row| {
        var c: usize = 0;
        while (c < cols) : (c += 128) {
            var s: f64 = 0;
            for (0..128) |n| s += @abs(x[row * cols + c + n]);
            s *= inv;
            for (0..128) |n| x[row * cols + c + n] = s;
        }
    }
}
