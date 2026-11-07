//! `ling-exl3-clamp DIR [LAYER]`: an EXL3 MoE layer's grouped gate-up in Ling's clamp order vs float64 on its own g, u.
const std = @import("std");
const cuda = @import("cuda");
const ling = @import("ling_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const lw = @import("ling_weights_tests.zig");

const Rig = qr.RigT;
const h = qr.helpers;
const ex = cuda.experts;

fn silu(g: f64) f64 {
    return g / (1 + @exp(-g));
}

/// `l`'s gate (or up) tables as a down projection, dims and width swapped: the down epilogue writes fp32 g (or u).
fn asDown(l: ex.Layer, up: bool) ex.Layer {
    var x = l.exl3.?;
    x.down = if (up) x.up else x.gate;
    x.down_k2 = if (up) x.up_k2 else x.gate_k2;
    x.suh_d = if (up) x.suh_u else x.suh_g;
    x.svh_d = if (up) x.svh_u else x.svh_g;
    x.dims = l.width;
    x.width = l.dims;
    var o = l;
    o.exl3 = x;
    o.dims = l.width;
    o.width = l.dims;
    o.limit = 0;
    o.after = false;
    return o;
}

fn plan(r: *Rig, router: *const cuda.grouped.Router, pairs: usize, slots: usize, count: usize) !cuda.grouped.Plan {
    const pk = try r.a.alloc(i32, pairs);
    for (pk, 0..) |*e, p| e.* = lw.pick(p, slots, count);
    const p: cuda.grouped.Plan = .{ .members = try r.zeros(pairs * 4), .items = try r.zeros(cuda.grouped.maxItems(pairs, count, 16) * 12), .counts = try r.zeros(8), .rank = try r.zeros(4), .hist = try r.zeros(4) };
    try router.route(r.s, try r.dev(std.mem.sliceAsBytes(pk)), pairs, count, 16, p);
    return p;
}

fn f32s(r: *Rig, ptr: u64, n: usize) ![]const f32 {
    const b = try r.back(ptr, n * 4);
    const out = try r.a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), b);
    return out;
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const layer: u32 = if (rest.len > 1) try std.fmt.parseInt(u32, rest[1], 10) else 4;
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0xe3c1);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*x| x.free();
        r.bufs.deinit(gpu.gpa);
    }
    var why: ling.config.Why = .{};
    var o = ling.weights.open(gpu.gpa, gpu.io, rest[0], &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ rest[0], why.text() });
        return e;
    };
    defer o.ck.close();
    var pack = try cuda.modelopt.Packer.load(gpu.d);
    defer pack.unload();
    var w = try ling.weights.load(gpu.gpa, gpu.d, s, &pack, &o.ck, o.c, &.{layer}, .{ .globals = false });
    defer w.deinit();
    const l = switch (w.layers[0].ffn) {
        .moe => |m| switch (m.routed) {
            .grouped => |x| x,
            .lanes => return error.NotGrouped,
        },
        .dense => return error.NotMoe,
    };
    try check.expect(l.format == .exl3 and l.after, "layer {d}: EXL3 experts loaded in Ling's order (after {})", .{ layer, l.after });
    var kx = try cuda.exl3_experts.Experts.load(gpu.d);
    defer kx.unload();
    var g: cuda.exl3_grouped.Grouped = .{ .d = gpu.d, .kernels = &kx };
    defer g.deinit();
    var em = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer em.unload();
    const router = try cuda.grouped.Router.resolve(em);
    const ni = l.width;
    const d = l.dims;
    const slots = @min(4, l.experts);
    for ([_]usize{ 3, 24 }) |rows| {
        const pairs = rows * slots;
        const p = try plan(&r, &router, pairs, slots, l.experts);
        const items = cuda.grouped.maxItems(pairs, l.experts, 16);
        const x = try r.bfs(rows * d, 4);
        const xp = try r.a.alloc(u16, pairs * d); // pair p's row: x[p / slots]
        for (0..pairs) |q| @memcpy(xp[q * d ..][0..d], x[(q / slots) * d ..][0..d]);
        const gp = try r.zeros(pairs * ni * 4);
        const up = try r.zeros(pairs * ni * 4);
        const pin: ex.Rows = .{ .x = try r.dev(std.mem.sliceAsBytes(xp)), .stride = d };
        try g.run(r.s, .down_f32, pin, asDown(l, false), p, gp, items, -1, false);
        try g.run(r.s, .down_f32, pin, asDown(l, true), p, up, items, -1, false);
        const gv = try f32s(&r, gp, pairs * ni);
        const uv = try f32s(&r, up, pairs * ni);
        // L at the 90th percentile of silu(g): the clamp bites on a tenth of the values
        const sv = try r.a.alloc(f64, gv.len);
        for (sv, gv) |*o2, x2| o2.* = silu(x2);
        std.mem.sort(f64, sv, {}, std.sort.asc(f64));
        const lim: f32 = @floatCast(sv[sv.len * 9 / 10]);
        const in: ex.Rows = .{ .x = try r.dev(std.mem.sliceAsBytes(x)), .stride = d, .slots = slots };
        var outs: [3][]const u8 = undefined;
        for ([_]bool{ true, false, true }, [_]f32{ lim, lim, 0 }, 0..) |after, lv, k| {
            var lc = l;
            lc.limit = lv;
            lc.after = after;
            lc.exl3.?.limit = lv;
            const out = try r.zeros(pairs * ni * 2);
            try g.run(r.s, .up, in, lc, p, out, items, -1, rows > 16);
            outs[k] = try r.back(out, pairs * ni * 2);
        }
        var most: f64 = 0;
        var most0: f64 = 0; // the unclamped control, silu(g) * u: checks how g and u were read
        var over: usize = 0;
        var at: usize = 0;
        var above: usize = 0;
        var wide: usize = 0;
        var off: usize = 0; // GLM's order (clamp before silu) outside this tolerance
        const L: f64 = lim;
        const e16 = std.math.pow(f64, 2, -16);
        for (0..pairs * ni) |i| {
            const gg: f64 = gv[i];
            const uu: f64 = uv[i];
            // the K-split order g and u were summed in moves each by its 128-block's scale (the Hadamard mixes them)
            var bg: f64 = 0;
            var bu: f64 = 0;
            for (i / 128 * 128..i / 128 * 128 + 128) |j| {
                bg = @max(bg, @abs(gv[j]));
                bu = @max(bu, @abs(uv[j]));
            }
            const sg = silu(gg);
            const want = @min(sg, L) * std.math.clamp(uu, -L, L);
            const tol = @abs(want) * std.math.pow(f64, 2, -8) + (1.1 * @min(@abs(uu), L) * bg + @min(@abs(sg), L) * bu) * e16 + 1e-30;
            const want0 = sg * uu;
            const tol0 = @abs(want0) * std.math.pow(f64, 2, -8) + (1.1 * @abs(uu) * bg + @abs(sg) * bu) * e16 + 1e-30;
            const got = h.bfOf(std.mem.bytesToValue(u16, outs[0][2 * i ..][0..2]));
            const glm_order = h.bfOf(std.mem.bytesToValue(u16, outs[1][2 * i ..][0..2]));
            const got0 = h.bfOf(std.mem.bytesToValue(u16, outs[2][2 * i ..][0..2]));
            const e = @abs(got - want) / tol;
            if (e > most) at = i;
            most = @max(most, e);
            over += @intFromBool(e > 1);
            most0 = @max(most0, @abs(got0 - want0) / tol0);
            above += @intFromBool(sg > L);
            wide += @intFromBool(@abs(uu) > L);
            off += @intFromBool(@abs(glm_order - want) > tol);
        }
        r.worst = @max(r.worst, most);
        const gw = h.bfOf(std.mem.bytesToValue(u16, outs[0][2 * at ..][0..2]));
        std.debug.print("RESULT layer {d} EXL3 gate-up, {d} rows ({d} pairs x {d}), L {d:.4}: float64 at {d:.3} of its tolerance ({d} over; worst g {e:.4} u {e:.4} got {e:.4}), unclamped control at {d:.3}; silu(g) > L on {d}, |u| > L on {d}; GLM's order off on {d}\n", .{ layer, rows, pairs, ni, lim, most, over, gv[at], uv[at], gw, most0, above, wide, off });
        try check.expect(most <= 1 and most0 <= 1 and above > 0 and wide > 0 and off > 0, "{d} rows: Ling's order within tolerance, the clamp exercised, GLM's order told apart", .{rows});
    }
    check.pass("ling-exl3-clamp: layer {d}'s EXL3 experts in Ling's order min(silu(g), L) * clip(u, -L, L), worst {d:.3} of tolerance", .{ layer, r.worst });
}
