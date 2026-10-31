//! IdolPipeline.decode on Metal: the octree's 8 levels to anchors, the Gaussian decoder, then the Gaussians.
const std = @import("std");
const mtl = @import("metal");
const dc = @import("decoder_config.zig");
const octree = @import("octree.zig");
const gs = @import("gaussian.zig");
const Decoder = @import("decoder.zig").Decoder;

pub const Timing = struct { octree: f64 = 0, octree_gpu: f64 = 0, gaussian: f64 = 0, gaussian_gpu: f64 = 0, activation: f64 = 0 };

/// Each level's parents and their counts, kept when a caller asks (for comparing a run's topology).
pub const Trace = struct {
    parents: [dc.max_level][]const [3]u32 = @splat(&.{}),
    counts: [dc.max_level][]const u32 = @splat(&.{}),
    probs: [dc.max_level][]const f32 = @splat(&.{}),

    pub fn deinit(t: *Trace, gpa: std.mem.Allocator) void {
        for (t.parents, t.counts, t.probs) |p, c, q| {
            gpa.free(p);
            gpa.free(c);
            gpa.free(q);
        }
        t.* = .{};
    }
};

pub const Result = struct {
    anchors: []f32,
    raw: []f32,
    gaussians: gs.Gaussians,
    time: Timing,

    pub fn deinit(r: *Result, gpa: std.mem.Allocator) void {
        gpa.free(r.anchors);
        gpa.free(r.raw);
        r.gaussians.free(gpa);
    }
};

/// The anchors for `points` points: each level's probabilities on Metal, its split on the host.
pub fn anchors(gpa: std.mem.Allocator, t: *Decoder, points: usize, draws: *octree.Draws, trace: ?*Trace, time: *Timing) ![]f32 {
    var level: octree.Level = .{ .coords = try gpa.dupe([3]u32, &.{.{ 0, 0, 0 }}), .counts = try gpa.dupe(u32, &.{@intCast(points)}) };
    defer {
        gpa.free(level.coords);
        gpa.free(level.counts);
    }
    for (1..dc.max_level + 1) |l| {
        const rows = level.coords.len;
        const centres = try gpa.alloc(f32, rows * 3);
        defer gpa.free(centres);
        octree.centres(level.coords, @intCast(l), centres);
        const probs = try gpa.alloc(f32, rows * dc.children);
        defer gpa.free(probs);
        const tm = try t.octreeLogits(centres, l, probs);
        time.octree += tm.wall;
        time.octree_gpu += tm.gpu;
        const started = mtl.clock.seconds();
        octree.softmax(probs);
        const next = try octree.descend(gpa, level, probs, @intCast(l), draws, null);
        time.octree += mtl.clock.seconds() - started;
        if (trace) |tr| {
            tr.parents[l - 1] = try gpa.dupe([3]u32, level.coords);
            tr.counts[l - 1] = try gpa.dupe(u32, level.counts);
            tr.probs[l - 1] = try gpa.dupe(f32, probs);
        }
        gpa.free(level.coords);
        gpa.free(level.counts);
        level = next;
    }
    const started = mtl.clock.seconds();
    const out = try gpa.alloc(f32, points * 3);
    errdefer gpa.free(out);
    try octree.anchors(level, dc.max_level, draws, out);
    time.octree += mtl.clock.seconds() - started;
    return out;
}

/// The Gaussians of `latent` [8192, 16]: `gaussians` (32,768 to 1,048,576, rounded down to whole anchors).
pub fn decode(gpa: std.mem.Allocator, t: *Decoder, latent: []const f32, gaussians: usize, draws: *octree.Draws, trace: ?*Trace) !Result {
    const points = try dc.anchors(gaussians);
    if (points > t.cap) return error.AnchorCount;
    try t.setLatent(latent);
    var time: Timing = .{};
    const pts = try anchors(gpa, t, points, draws, trace, &time);
    errdefer gpa.free(pts);
    const raw = try gpa.alloc(f32, points * dc.raw_width);
    errdefer gpa.free(raw);
    const tg = try t.gaussianRows(pts, raw);
    time.gaussian = tg.wall;
    time.gaussian_gpu = tg.gpu;
    const started = mtl.clock.seconds();
    const g = try gs.Gaussians.alloc(gpa, points * dc.per_anchor);
    activateParallel(t.params(), raw, pts, g);
    time.activation = mtl.clock.seconds() - started;
    return .{ .anchors = pts, .raw = raw, .gaussians = g, .time = time };
}

/// gaussian.activate split by anchors over up to 8 threads; each Gaussian's values depend on its own row alone.
pub fn activateParallel(p: gs.Params, raw: []const f32, pts: []const f32, g: gs.Gaussians) void {
    const n = pts.len / 3;
    const threads = @min(8, @max(1, n / 512));
    var pool: [8]?std.Thread = @splat(null);
    const per = (n + threads - 1) / threads;
    for (0..threads) |i| {
        const lo = @min(n, i * per);
        const hi = @min(n, lo + per);
        if (lo == hi) continue;
        const part: gs.Gaussians = .{
            .xyz = g.xyz[lo * 3 * dc.per_anchor .. hi * 3 * dc.per_anchor],
            .features_dc = g.features_dc[lo * 3 * dc.per_anchor .. hi * 3 * dc.per_anchor],
            .scaling = g.scaling[lo * 3 * dc.per_anchor .. hi * 3 * dc.per_anchor],
            .rotation = g.rotation[lo * 4 * dc.per_anchor .. hi * 4 * dc.per_anchor],
            .opacity = g.opacity[lo * dc.per_anchor .. hi * dc.per_anchor],
        };
        const args = .{ p, raw[lo * dc.raw_width .. hi * dc.raw_width], pts[lo * 3 .. hi * 3], part };
        pool[i] = std.Thread.spawn(.{}, gs.activate, args) catch blk: {
            @call(.auto, gs.activate, args);
            break :blk null;
        };
    }
    for (pool) |th| if (th) |h| h.join();
}

test "the parallel activation equals the serial one byte for byte" {
    const a = std.testing.allocator;
    const n = 1500;
    const raw = try a.alloc(f32, n * dc.raw_width);
    defer a.free(raw);
    const pts = try a.alloc(f32, n * 3);
    defer a.free(pts);
    var rng = std.Random.DefaultPrng.init(5);
    for (raw) |*v| v.* = rng.random().floatNorm(f32);
    for (pts) |*v| v.* = rng.random().float(f32);
    var pert: [dc.per_anchor * 3]f32 = undefined;
    for (&pert) |*v| v.* = rng.random().floatNorm(f32);
    const p: gs.Params = .{ .rep = dc.released, .base_offset_scale = -1, .perturbation = &pert };
    const one = try gs.Gaussians.alloc(a, n * dc.per_anchor);
    defer one.free(a);
    const many = try gs.Gaussians.alloc(a, n * dc.per_anchor);
    defer many.free(a);
    gs.activate(p, raw, pts, one);
    activateParallel(p, raw, pts, many);
    try std.testing.expectEqualSlices(f32, one.xyz.ptr[0 .. n * dc.per_anchor * 14], many.xyz.ptr[0 .. n * dc.per_anchor * 14]);
}
