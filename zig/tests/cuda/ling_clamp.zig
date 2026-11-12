//! Ling's SwiGLU order, min(silu(g), L) * clip(u, -L, L): glue and grouped experts vs float64, old kernels' bytes.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const ling = @import("ling_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const lw = @import("ling_weights_tests.zig");

const Rig = qr.RigT;
const h = qr.helpers;
const W = ling.weights;
const ex = cuda.experts;
const glue = glm.glue;
const Config = ling.config.Config;
const two7 = std.math.pow(f64, 2, -7);

/// Ling's activation in float64 on bf16 values: silu(g) as `s`, then min(s, L) and clip(u, -L, L); L 0 for none.
fn act(s: f64, u: f64, limit: f64) f64 {
    if (limit == 0) return s * u;
    return @min(s, limit) * std.math.clamp(u, -limit, limit);
}

fn silu(g: f64) f64 {
    return g / (1 + @exp(-g));
}

/// The glue's tf_ling_swiglu at L 10 and 2 on gates and ups straddling L vs float64; GLM's order beside it.
fn glueOrder(r: *Rig, lk: *const ling.ops.Kernels, set: glue.Set) !void {
    for ([_]f32{ 10, 2 }) |lim| for ([_]usize{ 1024, 704 }) |w| {
        const rows = 40;
        const gu = try r.bfs(rows * 2 * w, 1.5 * lim);
        const dgu = try r.dev(std.mem.sliceAsBytes(gu));
        const outs = [2]u64{ try r.zeros(rows * w * 2), try r.zeros(rows * w * 2) };
        const xs = try r.zeros(rows * w / 64 * 4);
        try ling.ops.swiglu(lk, r.s, dgu, outs[0], xs, lim, rows, w);
        try glue.swiglu(set, r.s, dgu, outs[1], xs, lim, rows, w);
        const got = try r.back(outs[0], rows * w * 2);
        const before = try r.back(outs[1], rows * w * 2);
        var most: f64 = 0;
        var above: usize = 0;
        var wide: usize = 0;
        var differ: usize = 0;
        for (0..rows) |m| for (0..w) |d| {
            const g = h.bfOf(gu[m * 2 * w + d]);
            const u = h.bfOf(gu[m * 2 * w + w + d]);
            const want = act(silu(g), u, lim);
            const i = m * w + d;
            const v = h.bfOf(std.mem.bytesToValue(u16, got[2 * i ..][0..2]));
            most = @max(most, @abs(v - want) / (@abs(want) * two7 + 1e-30));
            above += @intFromBool(silu(g) > lim);
            wide += @intFromBool(@abs(u) > lim);
            differ += @intFromBool(!std.mem.eql(u8, got[2 * i ..][0..2], before[2 * i ..][0..2]));
        };
        r.worst = @max(r.worst, most);
        std.debug.print("RESULT glue L {d}, width {d}: float64 at {d:.3} of 2^-7; silu(g) > L on {d}, |u| > L on {d} of {d}; {d} differ from GLM's order\n", .{ lim, w, most, above, wide, rows * w, differ });
        try check.expect(most <= 1 and above > 0 and wide > 0, "tf_ling_swiglu at L {d}: Ling's order within 2^-7 of float64", .{lim});
        // bf16(silu(g)) is 10 for every bf16 g >= 10, so the two orders agree on bytes at 10 and part at 2
        try check.expect(if (lim == 10) differ == 0 else differ > 0, "L {d}: {d} bytes differ from GLM's order", .{ lim, differ });
    };
}

/// A plan for `pairs` pairs, pair p on expert lw.pick(p), at the given tile.
fn plan(r: *Rig, kit: *const lw.Kit, pairs: usize, slots: usize, count: usize, tile: usize) !cuda.grouped.Plan {
    const pk = try r.a.alloc(i32, pairs);
    for (pk, 0..) |*e, p| e.* = lw.pick(p, slots, count);
    const p: cuda.grouped.Plan = .{ .members = try r.zeros(pairs * 4), .items = try r.zeros(cuda.grouped.maxItems(pairs, count, 16) * 12), .counts = try r.zeros(8), .rank = try r.zeros(4), .hist = try r.zeros(4) };
    try kit.router.route(r.s, try r.dev(std.mem.sliceAsBytes(pk)), pairs, count, tile, p);
    return p;
}

/// The grouped gate-up with Layer.after at L 10 and 2, decode and prompt rows, vs float64 from the file.
fn expertsOrder(r: *Rig, kit: *const lw.Kit, ck: *core.Checkpoint, c: *const Config, i: u32, l: ex.Layer) !void {
    const slots = @min(4, l.experts);
    const ni = l.width;
    const d = l.dims;
    var b: [2][192]u8 = undefined;
    const e0: u32 = @intCast(lw.pick(0, slots, l.experts));
    const g0 = try lw.Mat.read(ck, c, try ling.names.moduleName(&b[0], i, .expert_gate, e0), ni, d);
    const probe = try lw.reference(r.a, g0, try r.bfs(d, 1), 1);
    var ss: f64 = 0;
    for (probe.want) |v| ss += v * v;
    const rms = @sqrt(ss / @as(f64, @floatFromInt(ni)));
    for ([_]bool{ false, true }) |prompt| for ([_]f32{ 10, 2 }) |lim| {
        const rows: usize = if (prompt) 24 else 3;
        const pairs = rows * slots;
        const tile: usize = if (prompt) ex.promptTile(l.format) else 16;
        const p = try plan(r, kit, pairs, slots, l.experts, tile);
        const x = try r.bfs(rows * d, @floatCast(std.math.pow(f64, 2, @round(std.math.log2(1.2 * lim / rms)))));
        var lc = l;
        lc.limit = lim;
        lc.after = true;
        const out = try r.zeros(pairs * ni * 2);
        const in: ex.Rows = .{ .x = try r.dev(std.mem.sliceAsBytes(x)), .stride = d, .slots = slots };
        const items = cuda.grouped.maxItems(pairs, l.experts, tile);
        if (prompt) try kit.grouped().prompt(r.s, .up, in, lc, p, out, items, pairs, tile, -1) else try kit.grouped().decode(r.s, .up, in, lc, p, out, items, pairs, tile, -1);
        const want = try r.a.alloc(f64, pairs * ni);
        const tol = try r.a.alloc(f64, pairs * ni);
        var above: usize = 0;
        for (0..pairs) |q| {
            const e: u32 = @intCast(lw.pick(q, slots, l.experts));
            const g = try lw.Mat.read(ck, c, try ling.names.moduleName(&b[0], i, .expert_gate, e), ni, d);
            const u = try lw.Mat.read(ck, c, try ling.names.moduleName(&b[1], i, .expert_up, e), ni, d);
            const gr = try lw.reference(r.a, g, x[(q / slots) * d ..][0..d], 1);
            const ur = try lw.reference(r.a, u, x[(q / slots) * d ..][0..d], 1);
            for (0..ni) |j| {
                const gv = h.bfOf(h.bf16Of(@floatCast(gr.want[j])));
                const uv = h.bfOf(h.bf16Of(@floatCast(ur.want[j])));
                const s = h.bfOf(h.bf16Of(@floatCast(silu(gv))));
                want[q * ni + j] = act(s, uv, lim);
                above += @intFromBool(s > lim);
                // the bf16 steps (2^-6), then each product's bound: silu' under 1.1, min and clip 1-Lipschitz
                tol[q * ni + j] = @abs(want[q * ni + j]) * std.math.pow(f64, 2, -6) + 1.1 * @min(@abs(uv), lim) * gr.tol[j] + @min(@abs(s), lim) * ur.tol[j] + 1e-30;
            }
        }
        var name: [128]u8 = undefined;
        try r.close(try std.fmt.bufPrint(&name, "layer {d} {t} gate-up L {d} ({s}), silu(g) > L on {d} of {d}", .{ i, l.format, lim, if (prompt) "prompt" else "decode", above, pairs * ni }), out, false, want, tol);
        try check.expect(above > 0, "some gates past L", .{});
    };
}

/// `ling-clamp DIR`: the glue's Ling order vs float64, then the clamped MoE layer's grouped experts (FP8 and NVFP4).
pub fn run(gpu: check.Gpu, dir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xc1a);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*x| x.free();
        r.bufs.deinit(gpu.gpa);
    }
    var gm = try cuda.Module.load(gpu.d, cuda.kernels.glm_glue);
    defer gm.unload();
    const ks = try glue.Kernels.load(gm);
    var lk = try ling.ops.Kernels.load(gpu.d);
    defer lk.unload();
    try glueOrder(&r, &lk, .{ .glue = &ks });
    var kit: lw.Kit = undefined;
    try kit.init(gpu);
    defer kit.deinit();
    var why: ling.config.Why = .{};
    var o = W.open(gpu.gpa, gpu.io, dir, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ dir, why.text() });
        return e;
    };
    defer o.ck.close();
    const last: u32 = o.c.layers - 1;
    var w = try W.load(gpu.gpa, gpu.d, s, &kit.pack, &o.ck, o.c, &.{last}, .{ .globals = false });
    defer w.deinit();
    const moe = w.layers[0].ffn.moe;
    try check.expect(moe.shared_limit == ling.limits(&w.config, last).shared, "shared limit from the config", .{});
    switch (moe.routed) {
        .grouped => |l| {
            try check.expect(l.after and l.limit == ling.limits(&w.config, last).routed, "loader: Ling's order at the config's limit", .{});
            try expertsOrder(&r, &kit, &o.ck, &w.config, last, l);
        },
        .lanes => std.debug.print("RESULT {t}: routed experts run as lanes, through the glue checked above\n", .{o.c.quant}),
    }
    check.pass("ling-clamp {t}: min(silu(g), L) * clip(u, -L, L) within float64 bounds (worst {d:.3}), GLM's order beside it", .{ o.c.quant, r.worst });
}
