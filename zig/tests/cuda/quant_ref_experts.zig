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
        try g.prompt(r.s, .up, up_in, l, pl.p, act, pl.items, -1);
        try g.prompt(r.s, .down_bf16, down_in, l, pl.p, y, pl.items, -1);
    } else {
        try g.decode(r.s, .up, up_in, l, pl.p, act, pl.items, -1);
        try g.decode(r.s, .down_f32, down_in, l, pl.p, y, pl.items, -1);
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
            try W.packGateUp(r.a, pu[e * 2 * blk ..][0 .. 2 * blk], .{ gm.codes[e * wb ..][0..wb], gm.scales[e * sb ..][0..sb] }, .{ um.codes[e * wb ..][0..wb], um.scales[e * sb ..][0..sb] }, NI, D);
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
    }
    check.pass("experts fp8g (decode, prompt) and nvfp4 (decode): gate-up and down within float64 tolerance", .{});
}
