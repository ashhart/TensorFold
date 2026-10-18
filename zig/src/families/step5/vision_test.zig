//! Complete tower metadata fixtures use real Checkpoint and Tensor values with no mapped payload or device.
const std = @import("std");
const production = @import("vision.zig");
const ck = @import("../../core/checkpoint_metal.zig");
fn add(c: *ck.Checkpoint, name: []const u8, dims: []const usize) !void {
    var t: ck.Tensor = .{ .buffer = undefined, .offset = 0, .bytes = 2, .dtype = .bf16, .rank = dims.len };
    for (dims, 0..) |n, i| {
        t.shape[i] = n;
        t.bytes *= n;
    }
    const owned = try c.allocator.dupe(u8, name);
    errdefer c.allocator.free(owned);
    try c.tensors.put(c.allocator, owned, t);
}
fn fixture(c: *ck.Checkpoint) !void {
    const w = 1536;
    try add(c, "vision_model.conv1.weight", &.{ w, 3, 14, 14 });
    try add(c, "vision_model.ln_pre.weight", &.{w});
    try add(c, "vision_model.ln_pre.bias", &.{w});
    const names = [_][]const u8{ "ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias", "attn.in_proj_weight", "attn.in_proj_bias", "attn.out_proj.weight", "attn.out_proj.bias", "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias", "ls_1.gamma", "ls_2.gamma" };
    const dims = [_][]const usize{ &.{w}, &.{w}, &.{w}, &.{w}, &.{ 3 * w, w }, &.{3 * w}, &.{ w, w }, &.{w}, &.{ 8960, w }, &.{8960}, &.{ w, 8960 }, &.{w}, &.{w}, &.{w} };
    for (0..47) |i| for (names, dims) |suffix, shape| {
        var name: [160]u8 = undefined;
        try add(c, try std.fmt.bufPrint(&name, "vision_model.transformer.resblocks.{d}.{s}", .{ i, suffix }), shape);
    };
    try add(c, "vision_model.vit_downsampler1.bias", &.{2 * w});
    try add(c, "vision_model.vit_downsampler2.bias", &.{4 * w});
    try add(c, "vit_large_projector.weight", &.{ 4096, 4 * w });
    try add(c, "vision_model.vit_downsampler1.weight", &.{ 2 * w, w, 3, 3 });
    try add(c, "vision_model.vit_downsampler2.weight", &.{ 4 * w, 2 * w, 3, 3 });
    try add(c, "vision_model.positional_embedding", &.{ 52 * 52, w });
}

test "step5 host: tower projector requires the language width it actually writes" {
    try production.requireHidden(4096);
    for ([_]u32{ 0, 128, 4095, 4097, std.math.maxInt(u32) }) |hidden| {
        try std.testing.expectError(error.UnsupportedStep5Tower, production.requireHidden(hidden));
    }
}

test "step5 host: every tower tensor validates its axes before any GPU allocation" {
    var cp = ck.Checkpoint.init(std.testing.allocator);
    defer cp.deinit();
    try fixture(&cp);
    try std.testing.expectEqual(@as(usize, 667), cp.tensors.count());
    try production.validate(&cp, 4096);
    var it = cp.tensors.valueIterator();
    while (it.next()) |t| {
        const saved = t.*;
        t.rank += 1;
        const result = production.validate(&cp, 4096);
        t.* = saved;
        try std.testing.expectError(error.UnexpectedStep5Tensor, result);
    }
}

test "step5 host: tower same-count transposes and invalid byte extents are refused" {
    var cp = ck.Checkpoint.init(std.testing.allocator);
    defer cp.deinit();
    try fixture(&cp);
    const t = cp.tensors.getPtr("vit_large_projector.weight").?;
    std.mem.swap(usize, &t.shape[0], &t.shape[1]);
    try std.testing.expectError(error.UnexpectedStep5Tensor, production.validate(&cp, 4096));
    std.mem.swap(usize, &t.shape[0], &t.shape[1]);
    t.bytes += 1;
    try std.testing.expectError(error.UnexpectedStep5Tensor, production.validate(&cp, 4096));
}

test "step5 host: malformed tower returns before an undefined device or pipeline is used" {
    var cp = ck.Checkpoint.init(std.testing.allocator);
    defer cp.deinit();
    try fixture(&cp);
    try std.testing.expectError(error.UnsupportedStep5Tower, production.Vision.load(std.testing.allocator, undefined, &cp, 128));
    cp.tensors.getPtr("vision_model.conv1.weight").?.shape[0] = 1;
    try std.testing.expectError(error.UnexpectedStep5Tensor, production.Vision.load(std.testing.allocator, undefined, &cp, 4096));
}

test "step5 host: tower equivalent flattened patch and singleton-batch position layouts stay admitted" {
    var cp = ck.Checkpoint.init(std.testing.allocator);
    defer cp.deinit();
    try fixture(&cp);
    const patch = cp.tensors.getPtr("vision_model.conv1.weight").?;
    patch.rank = 2;
    patch.shape[0] = 1536;
    patch.shape[1] = 588;
    const pos = cp.tensors.getPtr("vision_model.positional_embedding").?;
    pos.rank = 3;
    pos.shape[0] = 1;
    pos.shape[1] = 2704;
    pos.shape[2] = 1536;
    try production.validate(&cp, 4096);
}
