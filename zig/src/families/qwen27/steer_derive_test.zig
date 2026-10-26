//! The producer's statistics: the per-prompt weighting, the per-layer difference of means, the unit lengths.
const std = @import("std");
const derive = @import("steer_derive");
const glp = @import("glp");

/// A set with one prompt per layer: every column of a layer holds the same value.
fn onePrompt(layers: usize, hidden: usize, values: []const f32) ![]f32 {
    const sums = try std.testing.allocator.alloc(f32, layers * hidden);
    for (values, 0..) |v, i| {
        sums[i * hidden] = v;
        for (1..hidden) |c| sums[i * hidden + c] = v;
    }
    return sums;
}

test "every prompt counts once in a set's mean, not once per token" {
    const a = std.testing.allocator;
    var set = try derive.Mean.init(a, 1, 2);
    defer set.deinit(a);
    // A prompt of two rows of 1.0, then a prompt of a hundred rows of 3.0.
    try set.addPrompt(&.{ 2, 2 }, 2);
    try set.addPrompt(&.{ 300, 300 }, 100);
    var mean = [_]f32{ 0, 0 };
    set.mean(0, &mean);
    // The two prompts weigh the same: (1 + 3) / 2, not (2 * 1 + 100 * 3) / 102.
    try std.testing.expectApproxEqAbs(@as(f32, 2), mean[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), mean[1], 1e-6);
}

test "the direction is the difference of the two sets' means, one unit long" {
    const a = std.testing.allocator;
    var harmful = try derive.Mean.init(a, 2, 2);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 2, 2);
    defer harmless.deinit(a);
    // Layer 0 differs by (2, 0); layer 1 by (0, 4).
    try harmful.addPrompt(&.{ 3, 0, 0, 4 }, 1);
    try harmless.addPrompt(&.{ 1, 0, 0, 0 }, 1);
    var derived = try derive.derive(a, harmful, harmless, 2, 2);
    defer derived.deinit(a);
    try std.testing.expectEqual(@as(usize, 2), derived.layers.len);
    try std.testing.expectEqual(@as(u32, 0), derived.layers[0].id);
    try std.testing.expectApproxEqAbs(@as(f32, 1), derived.layers[0].dir[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), derived.layers[0].dir[1], 1e-6);
    try std.testing.expectEqual(@as(u32, 1), derived.layers[1].id);
    try std.testing.expectApproxEqAbs(@as(f32, 0), derived.layers[1].dir[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 1), derived.layers[1].dir[1], 1e-6);
    for (derived.layers) |l| {
        var norm: f32 = 0;
        for (l.dir) |v| norm += v * v;
        try std.testing.expectApproxEqAbs(@as(f32, 1), @sqrt(norm), 1e-5);
    }
}

test "a layer whose two means agree is dropped, not emitted as a zero vector" {
    const a = std.testing.allocator;
    var harmful = try derive.Mean.init(a, 3, 2);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 3, 2);
    defer harmless.deinit(a);
    // Layer 0 agrees, layer 1 differs, layer 2 agrees.
    try harmful.addPrompt(&.{ 1, 1, 5, 5, 7, 7 }, 1);
    try harmless.addPrompt(&.{ 1, 1, 2, 2, 7, 7 }, 1);
    var derived = try derive.derive(a, harmful, harmless, 3, 2);
    defer derived.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), derived.layers.len);
    try std.testing.expectEqual(@as(u32, 1), derived.layers[0].id);
    // The difference is (3, 3), so the direction is the diagonal, half of the square root of two each.
    const half = @as(f32, 1) / @sqrt(@as(f32, 2));
    try std.testing.expectApproxEqAbs(half, derived.layers[0].dir[0], 1e-6);
    try std.testing.expectApproxEqAbs(half, derived.layers[0].dir[1], 1e-6);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, derived.dropped);
}

test "the derived directions round-trip through the GLP file the producer writes" {
    const a = std.testing.allocator;
    var harmful = try derive.Mean.init(a, 3, 8);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 3, 8);
    defer harmless.deinit(a);
    const harmful_sums = try onePrompt(3, 8, &.{ 1, 2, 3 });
    defer a.free(harmful_sums);
    const harmless_sums = try onePrompt(3, 8, &.{ 0, 1, 1 });
    defer a.free(harmless_sums);
    try harmful.addPrompt(harmful_sums, 1);
    try harmless.addPrompt(harmless_sums, 1);
    var derived = try derive.derive(a, harmful, harmless, 3, 8);
    defer derived.deinit(a);

    // Layer 0 is not steerable, so the file holds the other layers only.
    var layers: std.ArrayList(glp.Layer) = .empty;
    defer layers.deinit(a);
    for (derived.layers) |l| if (l.id != 0) try layers.append(a, .{ .id = l.id, .dir = l.dir });
    try std.testing.expectEqual(@as(usize, 2), layers.items.len);

    const bytes = try glp.write(a, .{
        .mode = .project,
        .alpha = 1.0,
        .hook_point = glp.residual_site,
        .base_model = "example/model",
        .base_revision = "0123456789abcdef",
        .method = derive.method,
        .contrast = derive.contrast,
        .layers = layers.items,
    });
    defer a.free(bytes);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var why: glp.Why = .{};
    const vector = try glp.read(arena.allocator(), bytes, &why);
    try std.testing.expectEqual(glp.Mode.project, vector.mode);
    try std.testing.expectEqualStrings(glp.residual_site, vector.hook_point);
    try std.testing.expectEqualStrings(derive.method, vector.method);
    try std.testing.expectEqualStrings(derive.contrast, vector.contrast);
    try std.testing.expectEqualStrings("example/model", vector.base_model);
    try std.testing.expectEqualStrings("0123456789abcdef", vector.base_revision);
    try std.testing.expectEqual(@as(f32, 1.0), vector.alpha);
    try std.testing.expectEqual(@as(usize, 8), vector.width);
    try std.testing.expectEqual(@as(usize, 2), vector.layers.len);
    try std.testing.expectEqual(@as(u32, 1), vector.layers[0].id);
    try std.testing.expectEqual(@as(u32, 2), vector.layers[1].id);
    for (vector.layers, derived.layers[1..]) |read, made| {
        try std.testing.expectEqual(made.dir.len, read.dir.len);
        for (made.dir, read.dir) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "the producer's file at the 27B's width reads back the same directions" {
    const a = std.testing.allocator;
    const hidden = 5120;
    var harmful = try derive.Mean.init(a, 2, hidden);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 2, hidden);
    defer harmless.deinit(a);
    const harmful_sums = try onePrompt(2, hidden, &.{ 1, 2 });
    defer a.free(harmful_sums);
    const harmless_sums = try onePrompt(2, hidden, &.{ 0, 1 });
    defer a.free(harmless_sums);
    try harmful.addPrompt(harmful_sums, 1);
    try harmless.addPrompt(harmless_sums, 1);
    var derived = try derive.derive(a, harmful, harmless, 2, hidden);
    defer derived.deinit(a);

    var layers: std.ArrayList(glp.Layer) = .empty;
    defer layers.deinit(a);
    for (derived.layers) |l| if (l.id != 0) try layers.append(a, .{ .id = l.id, .dir = l.dir });
    const bytes = try glp.write(a, .{
        .mode = .project,
        .alpha = 1.0,
        .hook_point = glp.residual_site,
        .base_model = "Qwen/Qwen3.8-27B",
        .method = derive.method,
        .contrast = derive.contrast,
        .layers = layers.items,
    });
    defer a.free(bytes);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var why: glp.Why = .{};
    const vector = try glp.read(arena.allocator(), bytes, &why);
    try std.testing.expectEqual(@as(usize, hidden), vector.width);
    try std.testing.expectEqual(@as(usize, 1), vector.layers.len);
    try std.testing.expectEqual(@as(u32, 1), vector.layers[0].id);
    for (derived.layers[1].dir, vector.layers[0].dir) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-6);
}
