//! The tower's names and layouts on fixture indexes: the original's and mlx-vlm's naming, mixed, partial and text-only.
const std = @import("std");
const vn = @import("vision_names.zig");
const vc = @import("vision_config.zig");

const DType = enum { bf16, f16, u32 };
const Src = struct { dtype: DType, shape: []const usize };

/// The two checkpoints' tower names: the one that loaded, then the oQ4 MLX conversion's (mlx-vlm's).
const namings = [_][]const u8{ "model.visual.", "vision_tower." };
const lm = [_][]const u8{ "language_model.model.embed_tokens.weight", "language_model.model.layers.0.self_attn.kv_b_proj.weight", "language_model.lm_head.weight" };
const patch_last = [_]usize{ vc.hidden, vc.temporal, vc.patch, vc.patch, 3 };
const down_last = [_]usize{ vc.out_hidden, vc.merge, vc.merge, vc.hidden };

/// An index's text: the language model's names, each tower tensor under `prefix` but `skip`, then `extra`.
fn indexText(a: std.mem.Allocator, prefix: ?[]const u8, skip: ?[]const u8, extra: []const []const u8) ![]const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    try names.appendSlice(a, &lm);
    if (prefix) |p| for (vn.tensors) |t| if (skip == null or !std.mem.eql(u8, t.suffix, skip.?)) {
        try names.append(a, try std.fmt.allocPrint(a, "{s}{s}", .{ p, t.suffix }));
    };
    try names.appendSlice(a, extra);
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "{\"metadata\": {\"total_size\": 1}, \"weight_map\": {");
    for (names.items, 0..) |n, i| try text.appendSlice(a, try std.fmt.allocPrint(a, "{s}\"{s}\": \"model-0000{d}-of-00003.safetensors\"", .{ if (i == 0) "" else ", ", n, i % 3 + 1 }));
    try text.appendSlice(a, "}}");
    return text.items;
}

/// The tower's prefix as a loader reads it from an index's text.
fn towerOf(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    var tally: vn.Tally = .{};
    for (doc.object.get("weight_map").?.object.keys()) |k| tally.add(k);
    return tally.tower();
}

/// The shards' headers for a tower under `prefix`: bf16, its conv kernels channels-last when `mlx`.
fn headers(a: std.mem.Allocator, prefix: []const u8, mlx: bool) !std.StringHashMapUnmanaged(Src) {
    var out: std.StringHashMapUnmanaged(Src) = .empty;
    for (lm) |n| try out.put(a, n, .{ .dtype = .u32, .shape = &.{ 8, 8 } });
    for (vn.tensors) |t| {
        const shape: []const usize = if (!mlx) t.shape else if (t.place == .top and t.place.top == .patch_w) &patch_last else if (t.place == .top and t.place.top == .down_w) &down_last else t.shape;
        try out.put(a, try std.fmt.allocPrint(a, "{s}{s}", .{ prefix, t.suffix }), .{ .dtype = .bf16, .shape = shape });
    }
    return out;
}

/// A stored kernel's rows as the tower reads them (arranged by `transposeOf`, else as stored).
fn arranged(a: std.mem.Allocator, p: vn.Planned(Src), stored: []const u16) ![]u16 {
    const out = try a.dupe(u16, stored);
    if (vn.transposeOf(p.spec, p.layout)) |ab| vn.transposeRows(out, stored, ab[0], ab[1]);
    return out;
}

test "GLM vision names: the original and mlx-vlm's naming plan the same tower" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plans: [2][]vn.Planned(Src) = undefined;
    for (namings, &plans, 0..) |p, *out, k| {
        const prefix = try towerOf(a, try indexText(a, p, null, &.{}));
        try std.testing.expectEqualStrings(p, prefix);
        var srcs = try headers(a, prefix, k == 1);
        out.* = try vn.plan(Src, a, prefix, &srcs);
    }
    try std.testing.expectEqual(@as(usize, 347), plans[0].len);
    try std.testing.expectEqualStrings("model.visual.blocks.0.attn.k_norm.weight", plans[0][2 + 7].name);
    try std.testing.expectEqualStrings("vision_tower.blocks.0.attn.k_norm.weight", plans[1][2 + 7].name);
    for (plans[0], plans[1]) |x, y| {
        try std.testing.expectEqualStrings(x.spec.suffix, y.spec.suffix);
        try std.testing.expectEqualStrings(x.name[namings[0].len..], y.name[namings[1].len..]);
        try std.testing.expectEqual(x.src.dtype, y.src.dtype);
        try std.testing.expectEqualSlices(usize, x.spec.shape, y.spec.shape);
        try std.testing.expectEqual(std.meta.activeTag(x.spec.place), std.meta.activeTag(y.spec.place));
        try std.testing.expectEqual(vn.Layout.original, x.layout);
        var nx: usize = 1;
        var ny: usize = 1;
        for (x.src.shape) |d| nx *= d;
        for (y.src.shape) |d| ny *= d;
        try std.testing.expectEqual(nx, ny);
    }
    try std.testing.expectEqual(vn.Layout.channels_last, plans[1][0].layout);
    try std.testing.expectEqual(vn.Layout.channels_last, plans[1][2 + vc.depth * 14 + 1].layout);
}

test "GLM vision names: both layouts of the conv kernels reach the tower as the same elements" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var plans: [2][]vn.Planned(Src) = undefined;
    for (namings, &plans, 0..) |p, *out, k| {
        var srcs = try headers(a, p, k == 1);
        out.* = try vn.plan(Src, a, p, &srcs);
    }
    const rows = 3;
    // patch embedding: PyTorch [O, C, T, H, W] against mlx-vlm's [O, T, H, W, C]
    const pw = 3 * vc.temporal * vc.patch * vc.patch;
    const torch_p = try a.alloc(u16, rows * pw);
    const mlx_p = try a.alloc(u16, rows * pw);
    for (torch_p, 0..) |*v, i| v.* = @truncate(i *% 2654435761 >> 7);
    for (0..rows) |o| for (0..3) |c| for (0..vc.temporal) |t| for (0..vc.patch) |h| for (0..vc.patch) |w| {
        mlx_p[o * pw + ((t * vc.patch + h) * vc.patch + w) * 3 + c] = torch_p[o * pw + ((c * vc.temporal + t) * vc.patch + h) * vc.patch + w];
    };
    try std.testing.expectEqualSlices(u16, try arranged(a, plans[0][0], torch_p), try arranged(a, plans[1][0], mlx_p));
    try std.testing.expectEqualSlices(u16, torch_p, try arranged(a, plans[1][0], mlx_p));
    // the merge: PyTorch [O, C, 2, 2] against mlx-vlm's [O, 2, 2, C]; the tower reads [O, 2, 2, C]
    const dw = vc.hidden * vc.merge * vc.merge;
    const torch_d = try a.alloc(u16, rows * dw);
    const mlx_d = try a.alloc(u16, rows * dw);
    for (torch_d, 0..) |*v, i| v.* = @truncate(i *% 40503 >> 3);
    for (0..rows) |o| for (0..vc.hidden) |c| for (0..vc.merge) |kh| for (0..vc.merge) |kw| {
        mlx_d[o * dw + (kh * vc.merge + kw) * vc.hidden + c] = torch_d[o * dw + (c * vc.merge + kh) * vc.merge + kw];
    };
    const down = 2 + vc.depth * 14 + 1;
    try std.testing.expectEqualSlices(u16, try arranged(a, plans[0][down], torch_d), try arranged(a, plans[1][down], mlx_d));
    try std.testing.expectEqualSlices(u16, mlx_d, try arranged(a, plans[0][down], torch_d));
}

test "GLM vision names: CUDA's per-tensor arrangement matches the plan's and refuses a third kernel shape" {
    const original = vn.tensors[0].shape;
    try std.testing.expectEqual(@as(?[2]usize, null), try vn.arrangement("patch_embed.proj.weight", original));
    try std.testing.expectEqual(@as(?[2]usize, .{ 392, 3 }), try vn.arrangement("patch_embed.proj.weight", &patch_last));
    try std.testing.expectEqual(@as(?[2]usize, .{ vc.hidden, 4 }), try vn.arrangement("downsample.weight", &.{ vc.out_hidden, vc.hidden, 2, 2 }));
    try std.testing.expectEqual(@as(?[2]usize, null), try vn.arrangement("downsample.weight", &down_last));
    try std.testing.expectEqual(@as(?[2]usize, null), try vn.arrangement("blocks.0.attn.qkv.weight", &.{ 1, 2 }));
    try std.testing.expectEqual(@as(?[2]usize, null), try vn.arrangement("blocks.0.attn.qkv.weight_scale", &.{ 1, 2 }));
    try std.testing.expectError(error.BadVisionShape, vn.arrangement("patch_embed.proj.weight", &.{ vc.hidden, 3, 14, 14, 2 }));
    try std.testing.expectError(error.BadVisionShape, vn.arrangement("downsample.weight", &.{ vc.out_hidden, 2, vc.hidden, 2 }));
}

test "GLM vision names: a tower under both prefixes refuses" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.MixedVisionTower, towerOf(a, try indexText(a, "model.visual.", null, &.{"vision_tower.blocks.0.attn.proj.bias"})));
    try std.testing.expectError(error.MixedVisionTower, towerOf(a, try indexText(a, "vision_tower.", null, &.{"model.visual.patch_embed.proj.weight"})));
}

test "GLM vision names: a partial tower refuses under either prefix" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (namings) |p| {
        try std.testing.expectError(error.PartialVisionTower, towerOf(a, try indexText(a, p, "blocks.23.mlp.down_proj.bias", &.{})));
        try std.testing.expectError(error.PartialVisionTower, towerOf(a, try indexText(a, p, "merger.down_proj.weight", &.{})));
        const one = try std.fmt.allocPrint(a, "{s}blocks.0.attn.k_norm.weight", .{p});
        try std.testing.expectError(error.PartialVisionTower, towerOf(a, try indexText(a, null, null, &.{one})));
        // names beside the tower's own (a quantized projection's scales) neither fill nor break it
        const scale = try std.fmt.allocPrint(a, "{s}blocks.0.attn.qkv.weight_scale", .{p});
        try std.testing.expectEqualStrings(p, try towerOf(a, try indexText(a, p, null, &.{scale})));
    }
}

test "GLM vision names: a text-only index has no tower" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try std.testing.expectError(error.NoVisionTower, towerOf(a, try indexText(a, null, null, &.{})));
    try std.testing.expectError(error.NoVisionTower, towerOf(a, try indexText(a, null, null, &.{ "visual.blocks.0.norm1.weight", "model.vision_tower.x" })));
}

test "GLM vision names: the plan refuses a missing, mistyped or misshapen tensor" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for (namings) |p| {
        var srcs = try headers(a, p, false);
        const qkv = try std.fmt.allocPrint(a, "{s}blocks.5.attn.qkv.weight", .{p});
        srcs.getPtr(qkv).?.dtype = .f16;
        try std.testing.expectError(error.UnexpectedTensor, vn.plan(Src, a, p, &srcs));
        srcs.getPtr(qkv).?.* = .{ .dtype = .bf16, .shape = &.{ vc.hidden, 3 * vc.hidden } };
        try std.testing.expectError(error.UnexpectedTensor, vn.plan(Src, a, p, &srcs));
        srcs.getPtr(qkv).?.shape = vn.tensors[2 + 5 * 14 + 2].shape;
        _ = try vn.plan(Src, a, p, &srcs);
        const patch = try std.fmt.allocPrint(a, "{s}patch_embed.proj.weight", .{p});
        srcs.getPtr(patch).?.shape = &.{ vc.hidden, 3, vc.patch, vc.patch, vc.temporal };
        try std.testing.expectError(error.UnexpectedTensor, vn.plan(Src, a, p, &srcs));
        srcs.getPtr(patch).?.shape = vn.tensors[0].shape;
        _ = srcs.remove(try std.fmt.allocPrint(a, "{s}downsample.bias", .{p}));
        try std.testing.expectError(error.MissingTensor, vn.plan(Src, a, p, &srcs));
    }
}
