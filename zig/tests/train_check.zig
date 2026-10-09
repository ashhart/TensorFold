//! The learning step's gradients against the engine's own forward, moving each layer along its own, on the real model.
const std = @import("std");
const tf = @import("tensorfold");

const nm = tf.nemotron;
const block = nm.adapters.block;
const max_rank = nm.adapters.max_rank;

/// Layers whose change is checked, down from the top, so the first wrong backward shows where agreement ends.
const checked = [_]usize{ 51, 50, 49, 48, 47, 46, 43, 42, 31, 12, 0 };

/// Loss changes each change is moved to make along its own gradient; the first, 10x the forward's jitter, decides.
const targets = [_]f64{ 0.1, 0.05 };

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) {
        std.debug.print("usage: tf-train-check MODEL_DIR\n", .{});
        std.process.exit(2);
    }
    const m = try nm.Model.load(gpa, init.io, args[1], false);
    const b = try nm.backend.Metal.init(gpa, m, .{ .capacity = 4096, .chunk = 512, .drafts = false, .streams = 2 });
    const t = try nm.train.Trainer.init(gpa, b);
    var ids: [48]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + (i * 7919) % 30000);
    const start = 30;
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    for (t.sites.list) |*site| for (site.seek.slice(f32, nm.adapters.candidates * site.in)) |*v| {
        v.* = rng.float(f32) * 2 - 1;
    };
    t.sites.frame();
    const first_candidates = try gpa.alloc(f32, m.config.layers * block * nm.adapters.candidates);
    @memset(first_candidates, 0);
    for (0..m.config.layers * block) |q| first_candidates[q * nm.adapters.candidates + q % block] = 1;
    try t.sites.open(first_candidates);
    for (t.sites.list) |*site| site.gate(0).* = -2;
    t.sites.attach(&m.weights, true);
    const first = t.sites.first();
    for (t.sites.list) |*site| for (site.b.slice(f32, max_rank * site.out)[first * site.out ..][0 .. block * site.out]) |*v| {
        v.* = (rng.float(f32) * 2 - 1) * 1e-3;
    };
    _ = try t.step(&ids, start, .loss);
    const f0 = std.Io.Clock.awake.now(init.io);
    const plain = (try t.step(&ids, start, .loss)).loss;
    const forward_ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - f0.toNanoseconds())) / 1e6;
    std.debug.print("forward and loss alone {d:.1} ms, GPU {d:.1} ms\n", .{ forward_ms, b.last_ms });
    const t0 = std.Io.Clock.awake.now(init.io);
    const with_grad = (try t.step(&ids, start, .grad)).loss;
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(init.io).toNanoseconds() - t0.toNanoseconds())) / 1e6;
    std.debug.print("loss {d:.5} (with gradients {d:.5}), a step with gradients {d:.1} ms, GPU {d:.1} ms\n", .{ plain, with_grad, ms, b.last_ms });
    if (std.c.getenv("TF_TRAIN_TIME_ONLY") != null) return;
    if (std.c.getenv("TF_TRAIN_PROFILE") != null) {
        var prof: nm.encoder.Profiler = .{ .queue = m.queue, .kernels = &m.kernels };
        t.profile = &prof;
        _ = try t.step(&ids, start, .grad);
        t.profile = null;
        prof.report(1);
        return;
    }
    var worst: f64 = 0;
    for (checked) |li| {
        const site = &t.sites.list[li];
        const nb = block * site.out;
        const bf = site.b.slice(f32, max_rank * site.out)[first * site.out ..][0..nb];
        const gb = site.gb.slice(f32, nb);
        const keep_b = try gpa.dupe(f32, bf);
        var norm2: f64 = 0;
        for (gb) |g| norm2 += g * g;
        std.debug.print("layer {d:>2} {s:<9} |g| {e:.2}", .{ li, @tagName(m.config.kinds[li]), @sqrt(norm2) });
        for (targets) |target| {
            const eta: f32 = @floatCast(target / norm2);
            for (bf, keep_b, gb) |*x, k, g| x.* = k - eta * g;
            const down = (try t.step(&ids, start, .loss)).loss;
            for (bf, keep_b, gb) |*x, k, g| x.* = k + eta * g;
            const up = (try t.step(&ids, start, .loss)).loss;
            @memcpy(bf, keep_b);
            const measured = (up - down) / 2;
            const ratio = measured / target;
            if (target == targets[0]) worst = @max(worst, @abs(ratio - 1));
            std.debug.print("  predicted {d:.3} measured {d:.4} ratio {d:.2}", .{ target, measured, ratio });
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("{s}: worst ratio off 1 by {d:.3}\n", .{ if (worst < 0.25) "ok" else "MISMATCH", worst });
    if (worst >= 0.25) std.process.exit(1);
}
