//! The draft head's small kernels vs hand-computed values: argmax ties and masks, alpha, gate cell, logits, SwiGLU.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const gm = @import("metal_vision_gemm.zig");

const t = std.testing;

fn bf(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn unbf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

fn rb(x: f64) f64 {
    return unbf(bf(@floatCast(x)));
}

const Gpu = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    lib: mtl.Library,

    fn init() !Gpu {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        errdefer device.deinit();
        const queue = device.queue() catch return error.SkipZigTest;
        errdefer queue.deinit();
        const src = try frags.source(device, t.allocator, ks.inkling2_draft);
        defer t.allocator.free(src);
        return .{ .device = device, .queue = queue, .lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx()) };
    }

    fn deinit(g: Gpu) void {
        g.lib.deinit();
        g.queue.deinit();
        g.device.deinit();
    }

    fn buffer(g: Gpu, comptime T: type, values: []const T) !mtl.Buffer {
        const b = try g.device.buffer(@max(values.len * @sizeOf(T), 16), mtl.ResourceOptions.shared);
        @memcpy(b.slice(T, values.len), values);
        return b;
    }

    /// Runs `name` once: `bufs` at 0 .., then `value` after them; groups or threads as `grid` with 256 / `group` lanes.
    fn run(g: Gpu, name: []const u8, bufs: []const mtl.Buffer, value: anytype, grid: mtl.Size, group: u32, by_groups: bool) !void {
        const pipe = try mtl.Pipeline.init(g.device, g.lib, name, false);
        defer pipe.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = g.queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(pipe);
        for (bufs, 0..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(value, bufs.len);
        if (by_groups) e.dispatchGroups(grid, mtl.Size.of(group, 1, 1)) else e.dispatchThreads(grid, mtl.Size.of(group, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        try t.expect(cb.failure() == null);
    }
};

test "pick takes the best group value and the lowest id on a tie, including all -inf" {
    const g = try Gpu.init();
    defer g.deinit();
    const inf = std.math.inf(f32);
    const cases = [_]struct { best: [4]f32, ids: [4]u32, want: u32 }{
        .{ .best = .{ 5, 7, 7, 3 }, .ids = .{ 10, 40, 2, 99 }, .want = 2 },
        .{ .best = .{ -inf, -inf, -inf, -inf }, .ids = .{ 31, 8, 8000, 9 }, .want = 8 },
        .{ .best = .{ 1, 2, 3, 4 }, .ids = .{ 1, 2, 3, 4 }, .want = 4 },
    };
    for (cases) |c| {
        const best = try g.buffer(f32, &c.best);
        defer best.deinit();
        const ids = try g.buffer(u32, &c.ids);
        defer ids.deinit();
        const seq = try g.buffer(u32, &.{ 0, 0, 0, 0, 0, 0 });
        defer seq.deinit();
        try g.run("ik2d_pick", &.{ best, ids, seq }, [2]u32{ 4, 2 }, mtl.Size.of(1, 1, 1), 1024, true);
        try t.expectEqual(c.want, seq.slice(u32, 6)[3]);
        try t.expectEqual(@as(u32, 0), seq.slice(u32, 6)[2]);
    }
}

test "alpha is the bf16 softmax of each fuse_weight row" {
    const g = try Gpu.init();
    defer g.deinit();
    const raw = [_]f32{ 0, 0, 1, -1, 3, 0.5, -2, -2.5 };
    var bits: [raw.len]u16 = undefined;
    for (raw, &bits) |r, *b| b.* = bf(r);
    const fw = try g.buffer(u16, &bits);
    defer fw.deinit();
    const out = try g.device.buffer(raw.len * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    try g.run("ik2d_alpha", &.{ fw, out }, [2]u32{ 4, 2 }, mtl.Size.of(4, 1, 1), 64, false);
    for (0..4) |r| {
        const a = unbf(bits[r * 2]);
        const b = unbf(bits[r * 2 + 1]);
        const top = @max(a, b);
        const z = @exp(a - top) + @exp(b - top);
        try t.expectEqual(rb(@exp(a - top) / z), unbf(out.slice(u16, raw.len)[r * 2]));
        try t.expectEqual(rb(@exp(b - top) / z), unbf(out.slice(u16, raw.len)[r * 2 + 1]));
    }
}

test "the gate cell and the SwiGLU product follow the reference's roundings" {
    const g = try Gpu.init();
    defer g.deinit();
    const rank = 8;
    var raw_bits: [3 * rank]u16 = undefined;
    var prng = std.Random.DefaultPrng.init(4);
    for (&raw_bits) |*b| b.* = bf((prng.random().float(f32) * 2 - 1) * 3);
    var old_bits: [rank]u16 = undefined;
    for (&old_bits) |*b| b.* = bf(prng.random().float(f32) * 2 - 1);
    for ([_]u32{ 0, 3 }) |step| {
        const raw = try g.buffer(u16, &raw_bits);
        defer raw.deinit();
        const state = try g.buffer(u16, &old_bits);
        defer state.deinit();
        const latent = try g.device.buffer(rank * 2, mtl.ResourceOptions.shared);
        defer latent.deinit();
        try g.run("ik2d_cell", &.{ raw, state, latent }, [3]u32{ 128, rank, step }, mtl.Size.of(rank, 1, 1), 64, false);
        for (0..rank) |j| {
            const gate = rb(1.0 / (1.0 + @exp(-unbf(raw_bits[j]))));
            const old = if (step == 0) 0.0 else unbf(old_bits[j]);
            const keep = rb(gate * old);
            const add = rb(rb(1.0 - gate) * rb(std.math.tanh(unbf(raw_bits[rank + j]))));
            try t.expectApproxEqAbs(rb(keep + add), unbf(state.slice(u16, rank)[j]), 0.02 * @abs(rb(keep + add)) + 1e-3);
            try t.expectApproxEqAbs(rb(std.math.tanh(unbf(raw_bits[2 * rank + j]))), unbf(latent.slice(u16, rank)[j]), 0.02);
        }
    }
    var gate_bits: [2 * 40]u16 = undefined;
    for (&gate_bits) |*b| b.* = bf((prng.random().float(f32) * 2 - 1) * 4);
    const gate = try g.buffer(u16, gate_bits[0..40]);
    defer gate.deinit();
    const up = try g.buffer(u16, gate_bits[40..]);
    defer up.deinit();
    const act = try g.device.buffer(40 * 2, mtl.ResourceOptions.shared);
    defer act.deinit();
    try g.run("ik2d_swiglu", &.{ gate, up, act }, @as(u32, 20), mtl.Size.of(20, 2, 1), 20, false);
    for (0..40) |i| {
        const x = unbf(gate_bits[i]);
        const want = rb(x / (1.0 + @exp(-x)) * unbf(gate_bits[40 + i]));
        try t.expectApproxEqAbs(want, unbf(act.slice(u16, 40)[i]), 0.02 * @abs(want) + 1e-3);
    }
}

test "step logits are capped, bias-corrected, masked past the real vocabulary, and reduced per group of 8" {
    const g = try Gpu.init();
    defer g.deinit();
    const vocab = 20;
    const unpadded = 18;
    const rank = 2;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    var base: [vocab]u16 = undefined;
    var fold_w: [vocab * rank]u16 = undefined;
    var fold_b: [vocab]u16 = undefined;
    const latent_bits = [rank]u16{ bf(0.7), bf(-0.4) };
    for (&base) |*b| b.* = bf((rng.float(f32) * 2 - 1) * 60);
    for (&fold_w) |*b| b.* = bf(rng.float(f32) * 2 - 1);
    for (&fold_b) |*b| b.* = bf(rng.float(f32) * 2 - 1);
    base[19] = bf(1e4); // masked past `unpadded`: must not win
    const base_b = try g.buffer(u16, &base);
    defer base_b.deinit();
    const latent = try g.buffer(u16, &latent_bits);
    defer latent.deinit();
    const fw = try g.buffer(u16, &fold_w);
    defer fw.deinit();
    const fb = try g.buffer(u16, &fold_b);
    defer fb.deinit();
    const best = try g.device.buffer(3 * 4, mtl.ResourceOptions.shared);
    defer best.deinit();
    const ids = try g.device.buffer(3 * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    const Step = extern struct { vocab: u32, unpadded: u32, rank: u32, cap: f32, lanes: u32, base_stride: u32, groups: u32 };
    try g.run("ik2d_step_logits", &.{ base_b, latent, fw, fb, best, ids }, Step{ .vocab = vocab, .unpadded = unpadded, .rank = rank, .cap = 30, .lanes = 1, .base_stride = 0, .groups = 3 }, mtl.Size.of(3, 1, 1), 256, true);
    var ys: [vocab]f64 = undefined;
    for (0..vocab) |v| {
        ys[v] = if (v < unpadded) blk: {
            var s: f64 = 0;
            for (0..rank) |r| s += unbf(latent_bits[r]) * unbf(fold_w[v * rank + r]);
            break :blk rb(30.0 * std.math.tanh(rb(unbf(base[v]) + rb(unbf(fold_b[v]) + s)) / 30.0));
        } else -std.math.inf(f64);
    }
    for (0..3) |grp| {
        var top: f64 = -std.math.inf(f64);
        for (ys[grp * 8 .. @min(grp * 8 + 8, vocab)]) |y| top = @max(top, y);
        try t.expectApproxEqAbs(top, @as(f64, best.slice(f32, 3)[grp]), 0.25);
        const id = ids.slice(u32, 3)[grp];
        try t.expect(id >= grp * 8 and id < @min(grp * 8 + 8, unpadded));
        try t.expect(ys[id] >= top - 0.25);
    }
}

test "the small-row GEMM has the tile GEMM's bits at every row count" {
    const g = try Gpu.init();
    defer g.deinit();
    var gemm = try gm.Gemm.init(g.device, g.lib, &.{.bf16_nt});
    defer gemm.deinit();
    const small = gemm.rows orelse return error.SkipZigTest; // a tensor-unit GPU keeps the tiles
    const n = 80;
    const k = 608; // two whole 256-deep blocks and a 96-deep rest
    var prng = std.Random.DefaultPrng.init(21);
    const rng = prng.random();
    var x: [gm.Gemm.rows_max * k]u16 = undefined;
    var w: [n * k]u16 = undefined;
    for (&x) |*v| v.* = bf((rng.float(f32) * 2 - 1) * 3);
    for (&w) |*v| v.* = bf((rng.float(f32) * 2 - 1) * 3);
    const xb = try g.buffer(u16, &x);
    defer xb.deinit();
    const wb = try g.buffer(u16, &w);
    defer wb.deinit();
    for ([_]u32{ 1, 3, 7, 8 }) |m| {
        const shape: gm.Shape = .{ .m = m, .n = n, .k = k, .lda = k, .ldb = k, .ldd = n };
        var out: [2]mtl.Buffer = undefined;
        for (&out, 0..) |*o, i| {
            o.* = try g.device.buffer(gm.Gemm.rows_max * n * 2, mtl.ResourceOptions.shared);
            @memset(o.slice(u16, gm.Gemm.rows_max * n), 0xffff);
            gemm.rows = if (i == 0) small else null;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = g.queue.commandBuffer();
            const e = cb.compute(.serial);
            gemm.run(e, .bf16_nt, .{ .buf = xb }, .{ .buf = wb }, .{ .buf = o.* }, shape);
            e.end();
            cb.commit();
            cb.wait();
            try t.expect(cb.failure() == null);
        }
        defer for (out) |o| o.deinit();
        try t.expectEqualSlices(u16, out[1].slice(u16, gm.Gemm.rows_max * n), out[0].slice(u16, gm.Gemm.rows_max * n));
    }
    gemm.rows = small;
}
