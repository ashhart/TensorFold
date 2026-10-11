//! `tensorfold train-check MODEL`: each layer's change moved along its own gradient against the forward (on CUDA).
const std = @import("std");
const nemotron = @import("nemotron");

const sites = nemotron.sites;

/// Layers whose change is checked, down from the top, so the first wrong backward shows where agreement ends.
const checked = [_]usize{ 51, 50, 49, 48, 47, 46, 43, 42, 31, 12, 0 };

/// Loss changes each change is moved to make along its own gradient; the first, 10x the forward's jitter, decides.
const targets = [_]f64{ 0.1, 0.05 };

fn ms(io: std.Io, since: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - since.toNanoseconds())) / 1e6;
}

pub fn check(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, dir: []const u8) !u8 {
    var none: ?nemotron.sites.Sites = null;
    const t = try nemotron.train.Trainer.init(gpa, io, e, dir, &none);
    defer t.deinit();
    var ids: [48]u32 = undefined;
    for (&ids, 0..) |*v, i| v.* = @intCast(1000 + (i * 7919) % 30000);
    const start = 30;
    var prng = std.Random.DefaultPrng.init(5);
    const rng = prng.random();
    for (t.sites.list) |*site| for (site.seek.slice(f32, sites.candidates * site.in)) |*v| {
        v.* = rng.float(f32) * 2 - 1;
    };
    t.sites.frame();
    const layers = e.c.layers;
    const first_candidates = try gpa.alloc(f32, layers * sites.block * sites.candidates);
    defer gpa.free(first_candidates);
    @memset(first_candidates, 0);
    for (0..layers * sites.block) |q| first_candidates[q * sites.candidates + q % sites.block] = 1;
    try t.sites.open(first_candidates);
    for (t.sites.list) |*site| site.gate(0).* = -2;
    t.sites.attach(&e.w, true);
    const first = t.sites.first();
    for (t.sites.list) |*site| for (site.b.slice(f32, sites.max_rank * site.out)[first * site.out ..][0 .. sites.block * site.out]) |*v| {
        v.* = (rng.float(f32) * 2 - 1) * 1e-3;
    };
    var turn: nemotron.train.Turn = .{ .decide = start - 4 }; // the block acts on every row after its decision row
    turn.bits.set(0);
    _ = try t.step(&ids, start, .loss, turn);
    const f0 = std.Io.Clock.awake.now(io);
    const plain = (try t.step(&ids, start, .loss, turn)).loss;
    const forward_ms = ms(io, f0);
    const t0 = std.Io.Clock.awake.now(io);
    const with_grad = (try t.step(&ids, start, .grad, turn)).loss;
    std.debug.print("forward and loss alone {d:.1} ms; loss {d:.5} (with gradients {d:.5}), a step with gradients {d:.1} ms\n", .{ forward_ms, plain, with_grad, ms(io, t0) });
    var worst: f64 = 0;
    for (checked) |li| {
        const site = &t.sites.list[li];
        const nb = sites.block * site.out;
        const bf = site.b.slice(f32, sites.max_rank * site.out)[first * site.out ..][0..nb];
        const gb = site.gb.slice(f32, nb);
        const keep_b = try gpa.dupe(f32, bf);
        defer gpa.free(keep_b);
        var norm2: f64 = 0;
        for (gb) |g| norm2 += g * g;
        std.debug.print("layer {d:>2} {s:<9} |g| {e:.2}", .{ li, @tagName(e.c.kinds[li]), @sqrt(norm2) });
        for (targets) |target| {
            const eta: f32 = @floatCast(target / norm2);
            for (bf, keep_b, gb) |*x, k, g| x.* = k - eta * g;
            const down = (try t.step(&ids, start, .loss, turn)).loss;
            for (bf, keep_b, gb) |*x, k, g| x.* = k + eta * g;
            const up = (try t.step(&ids, start, .loss, turn)).loss;
            @memcpy(bf, keep_b);
            const measured = (up - down) / 2;
            const ratio = measured / target;
            if (target == targets[0]) worst = @max(worst, @abs(ratio - 1));
            std.debug.print("  predicted {d:.3} measured {d:.4} ratio {d:.2}", .{ target, measured, ratio });
        }
        std.debug.print("\n", .{});
    }
    std.debug.print("{s}: worst ratio off 1 by {d:.3}\n", .{ if (worst < 0.25) "ok" else "MISMATCH", worst });
    return if (worst < 0.25) 0 else 1;
}
