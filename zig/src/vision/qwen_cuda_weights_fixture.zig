//! Small synthetic shards and a recording byte sink for the actual tower weight loader.
const std = @import("std");

pub const Config = struct {
    hidden: usize = 2,
    intermediate: usize = 3,
    depth: usize = 2,
    out_hidden: usize = 5,

    pub fn merged(c: Config) usize {
        return c.hidden * 4;
    }
};
pub const config: Config = .{};
const Block = struct { norm1_w: u64, norm1_b: u64, qkv_w: u64, qkv_b: u64, proj_w: u64, proj_b: u64, norm2_w: u64, norm2_b: u64, fc1_w: u64, fc1_b: u64, fc2_w: u64, fc2_b: u64 };
pub const Sink = struct {
    a: std.mem.Allocator,
    copies: std.ArrayList([]u8) = .empty,
    fail_at: ?usize = null,
    patch_w: u64 = 0,
    patch_b: u64 = 0,
    pos: u64 = 0,
    blocks: [2]Block = std.mem.zeroes([2]Block),
    merger_norm_w: u64 = 0,
    merger_norm_b: u64 = 0,
    fc1_w: u64 = 0,
    fc1_b: u64 = 0,
    fc2_w: u64 = 0,
    fc2_b: u64 = 0,

    pub fn put(s: *Sink, bytes: []const u8) !u64 {
        if (s.fail_at == s.copies.items.len) return error.InjectedCopyFailure;
        try s.copies.ensureUnusedCapacity(s.a, 1);
        s.copies.appendAssumeCapacity(try s.a.dupe(u8, bytes));
        return s.copies.items.len;
    }

    pub fn deinit(s: *Sink) void {
        for (s.copies.items) |bytes| s.a.free(bytes);
        s.copies.deinit(s.a);
    }
};
pub const Tensor = struct { suffix: []const u8, shape: []const usize };
pub const tensors = [_]Tensor{
    .{ .suffix = "patch_embed.proj.weight", .shape = &.{ 2, 2, 16, 16, 3 } },
    .{ .suffix = "patch_embed.proj.bias", .shape = &.{2} },
    .{ .suffix = "pos_embed.weight", .shape = &.{ 2304, 2 } },
    .{ .suffix = "blocks.0.norm1.weight", .shape = &.{2} },
    .{ .suffix = "blocks.0.norm1.bias", .shape = &.{2} },
    .{ .suffix = "blocks.0.attn.qkv.weight", .shape = &.{ 6, 2 } },
    .{ .suffix = "blocks.0.attn.qkv.bias", .shape = &.{6} },
    .{ .suffix = "blocks.0.attn.proj.weight", .shape = &.{ 2, 2 } },
    .{ .suffix = "blocks.0.attn.proj.bias", .shape = &.{2} },
    .{ .suffix = "blocks.0.norm2.weight", .shape = &.{2} },
    .{ .suffix = "blocks.0.norm2.bias", .shape = &.{2} },
    .{ .suffix = "blocks.0.mlp.linear_fc1.weight", .shape = &.{ 3, 2 } },
    .{ .suffix = "blocks.0.mlp.linear_fc1.bias", .shape = &.{3} },
    .{ .suffix = "blocks.0.mlp.linear_fc2.weight", .shape = &.{ 2, 3 } },
    .{ .suffix = "blocks.0.mlp.linear_fc2.bias", .shape = &.{2} },
    .{ .suffix = "blocks.1.norm1.weight", .shape = &.{2} },
    .{ .suffix = "blocks.1.norm1.bias", .shape = &.{2} },
    .{ .suffix = "blocks.1.attn.qkv.weight", .shape = &.{ 6, 2 } },
    .{ .suffix = "blocks.1.attn.qkv.bias", .shape = &.{6} },
    .{ .suffix = "blocks.1.attn.proj.weight", .shape = &.{ 2, 2 } },
    .{ .suffix = "blocks.1.attn.proj.bias", .shape = &.{2} },
    .{ .suffix = "blocks.1.norm2.weight", .shape = &.{2} },
    .{ .suffix = "blocks.1.norm2.bias", .shape = &.{2} },
    .{ .suffix = "blocks.1.mlp.linear_fc1.weight", .shape = &.{ 3, 2 } },
    .{ .suffix = "blocks.1.mlp.linear_fc1.bias", .shape = &.{3} },
    .{ .suffix = "blocks.1.mlp.linear_fc2.weight", .shape = &.{ 2, 3 } },
    .{ .suffix = "blocks.1.mlp.linear_fc2.bias", .shape = &.{2} },
    .{ .suffix = "merger.norm.weight", .shape = &.{2} },
    .{ .suffix = "merger.norm.bias", .shape = &.{2} },
    .{ .suffix = "merger.linear_fc1.weight", .shape = &.{ 8, 8 } },
    .{ .suffix = "merger.linear_fc1.bias", .shape = &.{8} },
    .{ .suffix = "merger.linear_fc2.weight", .shape = &.{ 5, 8 } },
    .{ .suffix = "merger.linear_fc2.bias", .shape = &.{5} },
};
pub const Options = struct {
    hf: bool = false,
    indexed: bool = false,
    missing: ?usize = null,
    wrong_dtype: bool = false,
    wrong_patch_shape: bool = false,
    extra: bool = false,
    mixed: bool = false,
};

/// Distinct bf16 bit patterns make every channel, frame and spatial permutation observable.
pub fn value(tensor: usize, element: usize) u16 {
    return @intCast((tensor * 7919 + element * 17 + 0x3800) % 0x10000);
}

pub fn count(shape: []const usize) usize {
    var n: usize = 1;
    for (shape) |dim| n *= dim;
    return n;
}

fn appendTensor(a: std.mem.Allocator, header: *std.ArrayList(u8), data: *std.ArrayList(u8), name: []const u8, shape: []const usize, dtype: []const u8, id: usize, hf: bool) !void {
    var shape_text: std.ArrayList(u8) = .empty;
    defer shape_text.deinit(a);
    for (shape, 0..) |dim, i| {
        const part = try std.fmt.allocPrint(a, "{s}{d}", .{ if (i == 0) "" else ",", dim });
        defer a.free(part);
        try shape_text.appendSlice(a, part);
    }
    const size = 2 * count(shape);
    const entry = try std.fmt.allocPrint(a, "{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[{s}],\"data_offsets\":[{d},{d}]}}", .{ if (header.items.len == 1) "" else ",", name, dtype, shape_text.items, data.items.len, data.items.len + size });
    defer a.free(entry);
    try header.appendSlice(a, entry);
    const begin = data.items.len;
    try data.resize(a, begin + size);
    for (0..count(shape)) |i| {
        var stored = i;
        if (id == 0 and hf and shape.len == 5 and shape[1] == 3) {
            const output = i / 1536;
            const channel = (i / 512) % 3;
            const time_y_x = i % 512;
            stored = output * 1536 + time_y_x * 3 + channel;
        }
        std.mem.writeInt(u16, data.items[begin + i * 2 ..][0..2], value(id, stored), .little);
    }
}

fn shard(tmp: std.testing.TmpDir, name: []const u8, options: Options, parity: ?usize) !void {
    const a = std.testing.allocator;
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(a);
    try header.append(a, '{');
    for (tensors, 0..) |tensor, i| {
        if (options.missing == i or (parity != null and i % 2 != parity.?)) continue;
        const full = try std.fmt.allocPrint(a, "{s}{s}", .{ if (options.hf) "model.visual." else "vision_tower.", tensor.suffix });
        defer a.free(full);
        const patch_shape: []const usize = if (options.hf)
            (if (options.wrong_patch_shape) &.{ 2, 3, 2, 8, 32 } else &.{ 2, 3, 2, 16, 16 })
        else
            (if (options.wrong_patch_shape) &.{ 2, 2, 16, 8, 6 } else tensor.shape);
        const shape = if (i == 0) patch_shape else tensor.shape;
        try appendTensor(a, &header, &data, full, shape, if (options.wrong_dtype and i == 0) "F16" else "BF16", i, options.hf);
    }
    if (parity == null or parity.? == 0) {
        if (options.extra) try appendTensor(a, &header, &data, if (options.hf) "model.visual.extra" else "vision_tower.extra", &.{1}, "BF16", 22, false);
        if (options.mixed) try appendTensor(a, &header, &data, "vision_tower.patch_embed.proj.weight", &.{ 2, 2, 16, 16, 3 }, "BF16", 0, false);
        try appendTensor(a, &header, &data, "language_model.unrelated", &.{1}, "BF16", 23, false);
    }
    try header.append(a, '}');
    var image: std.ArrayList(u8) = .empty;
    defer image.deinit(a);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, header.items.len, .little);
    try image.appendSlice(a, &size);
    try image.appendSlice(a, header.items);
    try image.appendSlice(a, data.items);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = name, .data = image.items });
}

pub fn write(tmp: std.testing.TmpDir, options: Options) !void {
    if (!options.indexed) return shard(tmp, "model.safetensors", options, null);
    try shard(tmp, "model-00001-of-00002.safetensors", options, 0);
    try shard(tmp, "model-00002-of-00002.safetensors", options, 1);
    const a = std.testing.allocator;
    var map: std.ArrayList(u8) = .empty;
    defer map.deinit(a);
    try map.appendSlice(a, "{\"weight_map\":{");
    for (tensors, 0..) |tensor, i| {
        const entry = try std.fmt.allocPrint(a, "{s}\"{s}{s}\":\"model-0000{d}-of-00002.safetensors\"", .{ if (i == 0) "" else ",", if (options.hf) "model.visual." else "vision_tower.", tensor.suffix, i % 2 + 1 });
        defer a.free(entry);
        try map.appendSlice(a, entry);
    }
    try map.appendSlice(a, "}}");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors.index.json", .data = map.items });
}

pub fn directory(tmp: std.testing.TmpDir, buf: *[96]u8) ![]const u8 {
    return std.fmt.bufPrint(buf, ".zig-cache/tmp/{s}", .{tmp.sub_path});
}
