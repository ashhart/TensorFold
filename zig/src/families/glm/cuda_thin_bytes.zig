//! GLM's weight bytes under TF_GLM_LAYERS: counted by tensor from the shard headers, only those the loader reads.
const std = @import("std");
const core = @import("core");

/// The names the CUDA loader reads from: the language model's (layers, embedding, norm) and the head.
const prefixes = [_][]const u8{ "model.language_model.", "lm_head." };
const layer = "model.language_model.layers.";

/// The bytes `weightShare` divides: the files' `total`, or under TF_GLM_LAYERS (`setting`) the tensors `bytes` counts.
pub fn count(a: std.mem.Allocator, io: std.Io, dir: []const u8, total: u64, setting: ?[]const u8, layers: u32, mtp: bool) u64 {
    const v = setting orelse return total;
    const keep = std.fmt.parseInt(u32, v, 10) catch return total; // the open refuses it by name
    if (keep == 0 or keep > layers) return total;
    return bytes(a, io, dir, keep, layers, mtp) catch total;
}

/// The stored bytes of every tensor a load of the first `keep` layers reads.
pub fn bytes(a: std.mem.Allocator, io: std.Io, dir: []const u8, keep: u32, layers: u32, mtp: bool) !u64 {
    var ck = try core.Checkpoint.openModelPrefixes(a, io, dir, &prefixes);
    defer ck.close();
    var total: u64 = 0;
    for (ck.files.items) |*f| for (f.names.keys()) |name| {
        if (reads(name, keep, layers, mtp)) total += f.get(name).?.bytes.len;
    };
    return total;
}

/// Layers below `keep`, the MTP layers from `layers` on when `mtp`, and every tensor outside a layer.
fn reads(name: []const u8, keep: u32, layers: u32, mtp: bool) bool {
    if (!std.mem.startsWith(u8, name, layer)) return true;
    const rest = name[layer.len..];
    const i = std.fmt.parseInt(u32, rest[0 .. std.mem.indexOfScalar(u8, rest, '.') orelse rest.len], 10) catch return true;
    return i < keep or (mtp and i >= layers);
}

/// One safetensors file of BF16 tensors, `sizes[i]` elements each, under `names[i]`.
fn writeShard(dir: std.Io.Dir, io: std.Io, path: []const u8, names: []const []const u8, sizes: []const usize) !void {
    const a = std.testing.allocator;
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    try header.append(a, '{');
    var at: usize = 0;
    for (names, sizes, 0..) |n, s, i| {
        if (i > 0) try header.append(a, ',');
        try header.print(a, "\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[{d}],\"data_offsets\":[{d},{d}]}}", .{ n, s, at, at + 2 * s });
        at += 2 * s;
    }
    try header.append(a, '}');
    const data = try a.alloc(u8, 8 + header.items.len + at);
    defer a.free(data);
    std.mem.writeInt(u64, data[0..8], header.items.len, .little);
    @memcpy(data[8..][0..header.items.len], header.items);
    @memset(data[8 + header.items.len ..], 0);
    try dir.writeFile(io, .{ .sub_path = path, .data = data });
}

test "GLM thin weights: TF_GLM_LAYERS counts the layers below N, MTP with drafts, the rest whole; unset keeps the files" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    // three layers and one MTP layer (index 3), over two shards an index names; a tower the loader does not read
    const one = [_][]const u8{ "model.language_model.embed_tokens.weight", "model.language_model.layers.0.mlp.w", "model.language_model.layers.1.mlp.w", "model.visual.blocks.0.w" };
    const two = [_][]const u8{ "model.language_model.layers.2.mlp.w", "model.language_model.layers.3.mlp.w", "model.language_model.layers.10.mlp.w", "model.language_model.norm.weight", "lm_head.weight" };
    try writeShard(tmp.dir, io, "model-1.safetensors", &one, &.{ 64, 100, 101, 1000 });
    try writeShard(tmp.dir, io, "model-2.safetensors", &two, &.{ 102, 50, 7, 8, 32 });
    var index: std.ArrayList(u8) = .empty;
    defer index.deinit(a);
    try index.appendSlice(a, "{\"metadata\":{},\"weight_map\":{");
    for (one) |n| try index.print(a, "\"{s}\":\"model-1.safetensors\",", .{n});
    for (two, 0..) |n, i| try index.print(a, "\"{s}\":\"model-2.safetensors\"{s}", .{ n, if (i + 1 < two.len) "," else "" });
    try index.appendSlice(a, "}}");
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index.items });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    const outside = 2 * (64 + 8 + 32); // embedding, norm, head
    try std.testing.expectEqual(@as(u64, outside + 2 * (100 + 101)), try bytes(a, io, dir, 2, 3, false));
    try std.testing.expectEqual(@as(u64, outside + 2 * (100 + 101 + 50 + 7)), try bytes(a, io, dir, 2, 3, true));
    try std.testing.expectEqual(@as(u64, outside + 2 * (100 + 101 + 102)), try bytes(a, io, dir, 3, 3, false));
    try std.testing.expectEqual(@as(u64, outside + 2 * 100), count(a, io, dir, 999, "1", 3, false));
    // unset, malformed, 0 or past the config's layers: the files' count, as before
    for ([_]?[]const u8{ null, "x", "0", "4" }) |s| try std.testing.expectEqual(@as(u64, 999), count(a, io, dir, 999, s, 3, true));
}
