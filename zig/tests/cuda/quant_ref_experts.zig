//! Grouped FP8 and NVFP4 experts against float64 references: a routed plan, SwiGLU gate-up, down to fp32 pair rows.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");

const grouped = cuda.grouped;
const experts = cuda.experts;
const h = qr.helpers;
const Rig = qr.RigT;

const E = 9; // experts, the shared one last
const NI = 256; // width
const D = 512; // dims
const slots = 4; // three routed picks a row, then the shared expert

/// One expert matrix's float64 values, by (row, col).
fn Matrix(comptime T: type) type {
    return struct { ctx: T, value: *const fn (T, usize, usize, usize) f64 };
}

fn picks(r: *Rig, rows: usize) ![]i32 {
    const out = try r.a.alloc(i32, rows * slots);
    for (0..rows) |row| {
        var seen: [E - 1]bool = @splat(false);
        for (0..slots - 1) |j| {
            var e = r.rng.uintLessThan(usize, E - 1);
            while (seen[e]) e = r.rng.uintLessThan(usize, E - 1);
            seen[e] = true;
            out[row * slots + j] = @intCast(e);
        }
        out[row * slots + slots - 1] = E - 1;
    }
    return out;
}

/// The shared plan for `pairs` picks at `tile` pairs an item: its scratch and a bound on its items.
fn plan(r: *Rig, router: grouped.Router, dpicks: u64, pairs: usize, tile: usize) !struct { p: grouped.Plan, items: usize } {
    const wide = pairs > grouped.small;
    const p: grouped.Plan = .{
        .members = try r.zeros(pairs * 4),
        .items = try r.zeros(grouped.maxItems(pairs, E, 16) * 12),
        .counts = try r.zeros(8),
        .rank = try r.zeros((if (wide) pairs else 1) * 4),
        .hist = try r.zeros((if (wide) (pairs + 1023) / 1024 * E else 1) * 4),
    };
    try router.route(r.s, dpicks, pairs, E, tile, p);
    return .{ .p = p, .items = grouped.maxItems(pairs, E, tile) };
}

/// SwiGLU as the kernels round it: gate and up to bf16, silu to bf16, the product to bf16 (no clamp).
fn swiglu(g: f64, u: f64) f64 {
    const gv = h.bfOf(h.bf16Of(@floatCast(g)));
    const uv = h.bfOf(h.bf16Of(@floatCast(u)));
    const s = h.bfOf(h.bf16Of(@floatCast(gv / (1 + @exp(-gv)))));
    return s * uv;
}

/// gate-up and down for one routed batch: references from the kernel's own bf16 activations for the down call.
fn batch(r: *Rig, comptime label: []const u8, g: experts.Grouped, router: grouped.Router, l: experts.Layer, x: []const u16, rows: usize, prompt: bool, gate: anytype, up: anytype, down: anytype) !struct { act: u64, y: u64 } {
    const pairs = rows * slots;
    const pk = try picks(r, rows);
    const tile = if (prompt) experts.promptTile(l.format) else 16;
    const pl = try plan(r, router, try r.dev(std.mem.sliceAsBytes(pk)), pairs, tile);
    const act = try r.zeros(pairs * NI * 2);
    const y = try r.zeros(pairs * D * 4);
    const up_in: experts.Rows = .{ .x = try r.dev(std.mem.sliceAsBytes(x[0 .. rows * D])), .stride = D, .slots = slots };
    const down_in: experts.Rows = .{ .x = act, .stride = NI };
    if (prompt) {
        try g.prompt(r.s, .up, up_in, l, pl.p, act, pl.items, pairs, tile, -1);
        try g.prompt(r.s, .down_bf16, down_in, l, pl.p, y, pl.items, pairs, tile, -1);
    } else {
        try g.decode(r.s, .up, up_in, l, pl.p, act, pl.items, pairs, tile, -1);
        try g.decode(r.s, .down_f32, down_in, l, pl.p, y, pl.items, pairs, tile, -1);
    }
    // gate-up: each pair's row of x through its expert, SwiGLU with the kernels' roundings
    const want = try r.a.alloc(f64, pairs * NI);
    const tol = try r.a.alloc(f64, pairs * NI);
    for (0..pairs) |p| {
        const e: usize = @intCast(pk[p]);
        const xr = x[(p / slots) * D ..][0..D];
        for (0..NI) |j| {
            var gs: f64 = 0;
            var us: f64 = 0;
            var abs: f64 = 0;
            for (0..D) |i| {
                const xv = h.bfOf(xr[i]);
                gs += xv * gate.value(gate.ctx, e, j, i);
                us += xv * up.value(up.ctx, e, j, i);
                abs += @abs(xv * gate.value(gate.ctx, e, j, i)) + @abs(xv * up.value(up.ctx, e, j, i));
            }
            const a = swiglu(gs, us);
            want[p * NI + j] = a;
            // a bf16 step either side of each of the three roundings, and fp32 sums
            tol[p * NI + j] = @abs(a) * std.math.pow(f64, 2, -6) + abs * std.math.pow(f64, 2, -12) + 1e-30;
        }
    }
    var name: [96]u8 = undefined;
    const path = if (prompt) "prompt" else "decode";
    try r.close(try std.fmt.bufPrint(&name, label ++ " experts {s} gate-up, {d} rows x {d} slots", .{ path, rows, slots }), act, false, want, tol);
    // down from the kernel's own activations: fp32 (decode) or bf16 sums (prompt)
    const got_act = try r.back(act, pairs * NI * 2);
    const dwant = try r.a.alloc(f64, pairs * D);
    const dtol = try r.a.alloc(f64, pairs * D);
    for (0..pairs) |p| {
        const e: usize = @intCast(pk[p]);
        for (0..D) |j| {
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (0..NI) |i| {
                const t = h.bfOf(std.mem.bytesToValue(u16, got_act[2 * (p * NI + i) ..][0..2])) * down.value(down.ctx, e, j, i);
                sum += t;
                abs += @abs(t);
            }
            dwant[p * D + j] = sum;
            dtol[p * D + j] = (if (prompt) @abs(sum) * std.math.pow(f64, 2, -7) else 0) + abs * std.math.pow(f64, 2, -12) + 1e-30;
        }
    }
    try r.close(try std.fmt.bufPrint(&name, label ++ " experts {s} down, {d} pairs", .{ path, pairs }), y, !prompt, dwant, dtol);
    return .{ .act = act, .y = y };
}

/// One joint route and 128-pair items must be byte-identical to two independent 64-pair calls.
fn nvfp4PairCase(
    r: *Rig,
    ex: *experts.Nvfp4Experts,
    router: grouped.Router,
    l: experts.Layer,
    x: []const u16,
    rows_a: usize,
    rows_b: usize,
    comptime skew: bool,
    comptime ni: usize,
    comptime dims: usize,
) !void {
    const pa = try picks(r, rows_a);
    const pb = try picks(r, rows_b);
    if (skew) for ([_][]i32{ pa, pb }) |one| for (one, 0..) |*e, p| {
        if (p % slots != slots - 1) e.* = if ((p / slots) % 5 == 0) 1 else 0;
    };
    const pk = try r.a.alloc(i32, pa.len + pb.len);
    @memcpy(pk[0..pa.len], pa);
    @memcpy(pk[pa.len..], pb);
    const na = rows_a * slots;
    const nb = rows_b * slots;
    const p0 = try plan(r, router, try r.dev(std.mem.sliceAsBytes(pa)), na, 64);
    const p1 = try plan(r, router, try r.dev(std.mem.sliceAsBytes(pb)), nb, 64);
    const joint = try plan(r, router, try r.dev(std.mem.sliceAsBytes(pk)), na + nb, 128);
    const a0 = try r.zeros(na * ni * 2);
    const a1 = try r.zeros(nb * ni * 2);
    const aj = try r.zeros((na + nb) * ni * 2);
    const y0 = try r.zeros(na * dims * 4);
    const y1 = try r.zeros(nb * dims * 4);
    const yj = try r.zeros((na + nb) * dims * 4);
    const xa = try r.dev(std.mem.sliceAsBytes(x[0 .. rows_a * dims]));
    const xb = try r.dev(std.mem.sliceAsBytes(x[rows_a * dims ..][0 .. rows_b * dims]));
    const in0: experts.Rows = .{ .x = xa, .stride = dims, .slots = slots };
    const in1: experts.Rows = .{ .x = xb, .stride = dims, .slots = slots };
    const g: experts.Grouped = .{ .nvfp4 = ex };
    try g.prompt(r.s, .up, in0, l, p0.p, a0, p0.items, na, 64, -1);
    try g.prompt(r.s, .down_bf16, .{ .x = a0, .stride = ni }, l, p0.p, y0, p0.items, na, 64, -1);
    try g.prompt(r.s, .up, in1, l, p1.p, a1, p1.items, nb, 64, -1);
    try g.prompt(r.s, .down_bf16, .{ .x = a1, .stride = ni }, l, p1.p, y1, p1.items, nb, 64, -1);
    try cuda.pair_experts.run(ex, r.s, .up, in0, in1, l, joint.p, aj, aj + na * ni * 2, na, joint.items, -1);
    const down0: experts.Rows = .{ .x = aj, .stride = ni };
    const down1: experts.Rows = .{ .x = aj + na * ni * 2, .stride = ni };
    try cuda.pair_experts.run(ex, r.s, .down_bf16, down0, down1, l, joint.p, yj, yj + na * dims * 4, na, joint.items, -1);
    const ba = try r.back(a0, na * ni * 2);
    const bb = try r.back(a1, nb * ni * 2);
    const bj = try r.back(aj, (na + nb) * ni * 2);
    var same = std.mem.eql(u8, bj[0..ba.len], ba) and std.mem.eql(u8, bj[ba.len..], bb);
    const oa = try r.back(y0, na * dims * 4);
    const ob = try r.back(y1, nb * dims * 4);
    const oj = try r.back(yj, (na + nb) * dims * 4);
    same = same and std.mem.eql(u8, oj[0..oa.len], oa) and std.mem.eql(u8, oj[oa.len..], ob);
    try check.expect(same, "nvfp4 paired prompt {d}+{d} rows at {d}/{d}: joint bytes equal two 64-pair calls", .{ rows_a, rows_b, ni, dims });
}

/// Prompt items of expert `skip` (the shared one, as a family that computes it itself passes) are left alone.
fn skipped(
    r: *Rig,
    comptime label: []const u8,
    g: experts.Grouped,
    router: grouped.Router,
    l: experts.Layer,
    x: []const u16,
    rows: usize,
) !void {
    const pairs = rows * slots;
    const pk = try picks(r, rows);
    const pl = try plan(r, router, try r.dev(std.mem.sliceAsBytes(pk)), pairs, experts.promptTile(l.format));
    const fill = try r.a.alloc(u8, pairs * @max(NI, D) * 2);
    @memset(fill, 0xA5);
    const skip: i32 = E - 1;
    const xd = try r.dev(std.mem.sliceAsBytes(x[0 .. rows * D]));
    const up_in: experts.Rows = .{ .x = xd, .stride = D, .slots = slots };
    const a0 = try r.zeros(pairs * NI * 2);
    const a1 = try r.dev(fill[0 .. pairs * NI * 2]);
    try g.prompt(r.s, .up, up_in, l, pl.p, a0, pl.items, pairs, experts.promptTile(l.format), -1);
    try g.prompt(r.s, .up, up_in, l, pl.p, a1, pl.items, pairs, experts.promptTile(l.format), skip);
    const up_plain = try r.back(a0, pairs * NI * 2);
    try leftAlone(label ++ " prompt gate-up", pk, skip, up_plain, try r.back(a1, pairs * NI * 2), NI * 2, fill);
    const down_in: experts.Rows = .{ .x = a0, .stride = NI };
    const y0 = try r.zeros(pairs * D * 2);
    const y1 = try r.dev(fill[0 .. pairs * D * 2]);
    try g.prompt(r.s, .down_bf16, down_in, l, pl.p, y0, pl.items, pairs, experts.promptTile(l.format), -1);
    try g.prompt(r.s, .down_bf16, down_in, l, pl.p, y1, pl.items, pairs, experts.promptTile(l.format), skip);
    const down_plain = try r.back(y0, pairs * D * 2);
    try leftAlone(label ++ " prompt down", pk, skip, down_plain, try r.back(y1, pairs * D * 2), D * 2, fill);
}

/// Each pair's row: the skipped expert's still the fill, every other one equal to the run without skip.
fn leftAlone(
    name: []const u8,
    pk: []const i32,
    skip: i32,
    plain: []const u8,
    got: []const u8,
    row: usize,
    fill: []const u8,
) !void {
    var left: usize = 0;
    var differ: usize = 0;
    for (pk, 0..) |e, p| {
        const b = got[p * row ..][0..row];
        if (e == skip) {
            if (std.mem.eql(u8, b, fill[0..row])) left += 1 else differ += 1;
        } else if (!std.mem.eql(u8, plain[p * row ..][0..row], b)) differ += 1;
    }
    const what = "{s}: expert {d}'s {d} pairs left alone, every other pair as without skip ({d} differ)";
    try check.expect(differ == 0 and left > 0, what, .{ name, skip, left, differ });
}

const Fp8M = struct { codes: []const u8, inv: []const f32, n: usize, k: usize };
fn fp8Val(c: Fp8M, e: usize, row: usize, col: usize) f64 {
    const nb = c.n / 128;
    const kb = c.k / 128;
    return h.e4m3Of(c.codes[(e * c.n + row) * c.k + col]) * c.inv[(e * nb + row / 128) * kb + col / 128];
}

const Fp4M = struct { codes: []const u8, scales: []const u8, global: []const f32, n: usize, k: usize };
fn fp4Val(c: Fp4M, e: usize, row: usize, col: usize) f64 {
    const byte = c.codes[(e * c.n + row) * (c.k / 2) + col / 2];
    const nib: u8 = if (col & 1 == 0) byte & 0xF else byte >> 4;
    return h.e2m1Of(nib) * h.e4m3Of(c.scales[(e * c.n + row) * (c.k / 16) + col / 16]) * c.global[e];
}

fn fp8Mat(r: *Rig, n: usize, k: usize) !Fp8M {
    const codes = try r.a.alloc(u8, E * n * k);
    for (codes) |*c| c.* = h.fp8CodeOf(r);
    const inv = try r.a.alloc(f32, E * (n / 128) * (k / 128));
    for (inv) |*v| v.* = r.rng.float(f32) * 0.01 + 0.001;
    return .{ .codes = codes, .inv = inv, .n = n, .k = k };
}

fn fp4Mat(r: *Rig, n: usize, k: usize) !Fp4M {
    const codes = try r.a.alloc(u8, E * n * k / 2);
    r.rng.bytes(codes);
    const scales = try r.a.alloc(u8, E * n * k / 16);
    for (scales) |*v| v.* = 0x20 + r.rng.uintLessThan(u8, 0x28);
    const global = try r.a.alloc(f32, E);
    for (global) |*v| v.* = 0.005 + r.rng.float(f32) * 0.01;
    return .{ .codes = codes, .scales = scales, .global = global, .n = n, .k = k };
}

pub fn run(r: *Rig) !void {
    const sms: usize = @intCast(try r.gpu.ctx.attribute(.multiprocessor_count));
    var mod = try cuda.Module.load(r.gpu.d, cuda.kernels.experts);
    defer mod.unload();
    const router = try grouped.Router.resolve(mod);
    const x = try r.bfs(300 * D, 1.0);
    // block-FP8
    {
        const gm = try fp8Mat(r, NI, D);
        const um = try fp8Mat(r, NI, D);
        const dm = try fp8Mat(r, D, NI);
        const one = NI * D;
        const pu = try r.a.alloc(u32, E * one / 2);
        const pd = try r.a.alloc(u32, E * one / 4);
        const us = try r.a.alloc(f32, E * 2 * (NI / 128) * (D / 128));
        const per = (NI / 128) * (D / 128);
        for (0..E) |e| {
            try experts.Fp8Experts.packGateUp(r.a, pu[e * one / 2 ..][0 .. one / 2], gm.codes[e * one ..][0..one], um.codes[e * one ..][0..one], NI, D);
            experts.Fp8Experts.packOne(pd[e * one / 4 ..][0 .. one / 4], dm.codes[e * one ..][0..one], D, NI);
            @memcpy(us[2 * e * per ..][0..per], gm.inv[e * per ..][0..per]);
            @memcpy(us[(2 * e + 1) * per ..][0..per], um.inv[e * per ..][0..per]);
        }
        const l: experts.Layer = .{ .format = .fp8g, .up = try r.dev(std.mem.sliceAsBytes(pu)), .down = try r.dev(std.mem.sliceAsBytes(pd)), .up_scale = try r.dev(std.mem.sliceAsBytes(us)), .down_scale = try r.dev(std.mem.sliceAsBytes(dm.inv)), .width = NI, .dims = D, .experts = E };
        var ex = try experts.Fp8Experts.load(r.gpu.d, sms);
        defer ex.unload();
        const g: experts.Grouped = .{ .fp8g = &ex };
        const M = Matrix(Fp8M);
        for ([_]usize{ 1, 3 }) |rows| _ = try batch(r, "fp8", g, router, l, x, rows, false, M{ .ctx = gm, .value = fp8Val }, M{ .ctx = um, .value = fp8Val }, M{ .ctx = dm, .value = fp8Val });
        _ = try batch(r, "fp8", g, router, l, x, 300, true, M{ .ctx = gm, .value = fp8Val }, M{ .ctx = um, .value = fp8Val }, M{ .ctx = dm, .value = fp8Val });
        try skipped(r, "fp8", g, router, l, x, 300);
    }
    // NVFP4
    {
        const gm = try fp4Mat(r, NI, D);
        const um = try fp4Mat(r, NI, D);
        const dm = try fp4Mat(r, D, NI);
        const W = experts.Nvfp4Experts;
        const blk = NI / W.cols * (D / 32) * W.words;
        const pu = try r.a.alloc(u32, E * 2 * blk);
        const pd = try r.a.alloc(u32, E * blk);
        const wb = NI * D / 2;
        const sb = NI * D / 16;
        const us = try r.a.alloc(f32, 2 * E);
        for (0..E) |e| {
            const gate: [2][]const u8 = .{ gm.codes[e * wb ..][0..wb], gm.scales[e * sb ..][0..sb] };
            const up: [2][]const u8 = .{ um.codes[e * wb ..][0..wb], um.scales[e * sb ..][0..sb] };
            W.packGateUp(pu[e * 2 * blk ..][0 .. 2 * blk], gate, up, NI, D);
            W.packOne(pd[e * blk ..][0..blk], dm.codes[e * wb ..][0..wb], dm.scales[e * sb ..][0..sb], D, NI);
            us[2 * e] = gm.global[e];
            us[2 * e + 1] = um.global[e];
        }
        const l: experts.Layer = .{ .format = .nvfp4, .up = try r.dev(std.mem.sliceAsBytes(pu)), .down = try r.dev(std.mem.sliceAsBytes(pd)), .up_scale = try r.dev(std.mem.sliceAsBytes(us)), .down_scale = try r.dev(std.mem.sliceAsBytes(dm.global)), .width = NI, .dims = D, .experts = E };
        var ex = try W.load(r.gpu.d, sms);
        defer ex.unload();
        const g: experts.Grouped = .{ .nvfp4 = &ex };
        const M = Matrix(Fp4M);
        for ([_]usize{ 1, 3, 40 }) |rows| _ = try batch(r, "nvfp4", g, router, l, x, rows, false, M{ .ctx = gm, .value = fp4Val }, M{ .ctx = um, .value = fp4Val }, M{ .ctx = dm, .value = fp4Val });
        _ = try batch(r, "nvfp4", g, router, l, x, 300, true, M{ .ctx = gm, .value = fp4Val }, M{ .ctx = um, .value = fp4Val }, M{ .ctx = dm, .value = fp4Val });
        try skipped(r, "nvfp4", g, router, l, x, 300);
        for ([_]struct { usize, usize }{ .{ 1, 3 }, .{ 33, 65 }, .{ 128, 256 } }) |rows| {
            try nvfp4PairCase(r, &ex, router, l, x, rows[0], rows[1], false, NI, D);
        }
        // GLM's production block-16 shapes: M=3 gate/up, M=32 down, NT=2, KC=2.
        const pn = 192;
        const pdk = 2048;
        const pgm = try fp4Mat(r, pn, pdk);
        const pum = try fp4Mat(r, pn, pdk);
        const pdm = try fp4Mat(r, pdk, pn);
        const pblk = pn / W.cols * (pdk / 32) * W.words;
        const ppu = try r.a.alloc(u32, E * 2 * pblk);
        const ppd = try r.a.alloc(u32, E * pblk);
        const pwb = pn * pdk / 2;
        const psb = pn * pdk / 16;
        const pus = try r.a.alloc(f32, 2 * E);
        for (0..E) |e| {
            const gate: [2][]const u8 = .{ pgm.codes[e * pwb ..][0..pwb], pgm.scales[e * psb ..][0..psb] };
            const up: [2][]const u8 = .{ pum.codes[e * pwb ..][0..pwb], pum.scales[e * psb ..][0..psb] };
            W.packGateUp(ppu[e * 2 * pblk ..][0 .. 2 * pblk], gate, up, pn, pdk);
            W.packOne(ppd[e * pblk ..][0..pblk], pdm.codes[e * pwb ..][0..pwb], pdm.scales[e * psb ..][0..psb], pdk, pn);
            pus[2 * e] = pgm.global[e];
            pus[2 * e + 1] = pum.global[e];
        }
        const pl: experts.Layer = .{ .format = .nvfp4, .up = try r.dev(std.mem.sliceAsBytes(ppu)), .down = try r.dev(std.mem.sliceAsBytes(ppd)), .up_scale = try r.dev(std.mem.sliceAsBytes(pus)), .down_scale = try r.dev(std.mem.sliceAsBytes(pdm.global)), .width = pn, .dims = pdk, .experts = E };
        const px = try r.bfs(4096 * pdk, 1.0);
        try nvfp4PairCase(r, &ex, router, pl, px, 2048, 2048, true, pn, pdk);
    }
    const what = "experts fp8g and nvfp4 (decode, prompt): gate-up and down within float64 tolerance";
    check.pass(what ++ "; a skipped expert's prompt items left alone", .{});
}
