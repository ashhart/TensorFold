//! The width each Nemotron-H matrix is read at on HIP: config.json's per-module entries, and the widths admitted.
const std = @import("std");
const config = @import("../config.zig");
const tensors = @import("tensors.zig");

const Config = config.Config;
const Width = config.Width;

/// The MLX affine widths the HIP launches read for Nemotron-H, as the registry names them.
pub const formats: []const []const u8 = &.{ "mlx-q4g64", "mlx-q5g64", "mlx-q6g64", "mlx-q8g64" };
/// What the refusals say Nemotron on HIP reads.
pub const served_text = "MLX affine 4-, 5-, 6- or 8-bit matrices in groups of 64";

/// A width the HIP launches read for Nemotron-H: 4, 5, 6 or 8 bits in groups of 64.
pub fn served(w: Width) bool {
    return w.group == 64 and (w.bits == 4 or w.bits == 5 or w.bits == 6 or w.bits == 8);
}

/// The width of the matrix `key` (a projection's name before `.weight`), or null for a name no entry covers.
pub fn of(c: *const Config, key: []const u8) ?Width {
    if (std.mem.eql(u8, key, tensors.embed)) return c.embed_width;
    if (std.mem.eql(u8, key, tensors.lm_head)) return c.head_width;
    const stem = "backbone.layers.";
    if (!std.mem.startsWith(u8, key, stem)) return null;
    const rest = key[stem.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const i = std.fmt.parseInt(usize, rest[0..dot], 10) catch return null;
    const mixer = ".mixer.";
    if (i >= c.layers or !std.mem.startsWith(u8, rest[dot..], mixer)) return null;
    const leaf = rest[dot + mixer.len ..];
    for (config.members(c.kinds[i]), 0..) |m, j| if (std.mem.eql(u8, m, leaf)) return c.widths[i][j];
    return null;
}

/// The first matrix at a width the launches do not read (its name written into `buf`), else null.
pub fn unserved(c: *const Config, buf: []u8) ?config.Named {
    if (!served(c.embed_width)) return .{ .name = tensors.embed, .width = c.embed_width };
    for (0..c.layers) |i| for (config.members(c.kinds[i]), 0..) |m, j| if (!served(c.widths[i][j])) {
        const name = std.fmt.bufPrint(buf, "backbone.layers.{d}.mixer.{s}", .{ i, m }) catch m;
        return .{ .name = name, .width = c.widths[i][j] };
    };
    if (!served(c.head_width)) return .{ .name = tensors.lm_head, .width = c.head_width };
    return null;
}

/// A short label of the widths a checkpoint holds, as `mlx-q4g64` or `mlx-q4g64+q5g64+q8g64` for a mix.
pub fn label(c: *const Config, buf: []u8) []const u8 {
    var seen: [9]bool = @splat(false);
    seen[@min(c.embed_width.bits, 8)] = true;
    seen[@min(c.head_width.bits, 8)] = true;
    for (0..c.layers) |i| for (0..config.members(c.kinds[i]).len) |j| {
        seen[@min(c.widths[i][j].bits, 8)] = true;
    };
    var w: std.Io.Writer = .fixed(buf);
    var first = true;
    for (seen, 0..) |on, bits| if (on) {
        w.print("{s}q{d}g64", .{ if (first) "mlx-" else "+", bits }) catch break;
        first = false;
    };
    return w.buffered();
}

/// The MTP head file's width from its `__metadata__` quantization text (null: the stock head, 4-bit in groups of 64).
pub fn headOf(a: std.mem.Allocator, quantization: ?[]const u8) !Width {
    const text = quantization orelse return .{};
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, text, .{}) catch return error.BadHeadWidth;
    if (doc != .object) return error.BadHeadWidth;
    const bits = doc.object.get("bits") orelse return error.BadHeadWidth;
    const group = doc.object.get("group_size") orelse return error.BadHeadWidth;
    if (bits != .integer or group != .integer) return error.BadHeadWidth;
    if (doc.object.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return error.BadHeadWidth;
    return .{
        .bits = std.math.cast(u8, bits.integer) orelse return error.BadHeadWidth,
        .group = std.math.cast(u16, group.integer) orelse return error.BadHeadWidth,
    };
}

/// The `__metadata__` quantization text of the safetensors file at `path`, or null when it states none.
pub fn headQuantization(a: std.mem.Allocator, io: std.Io, path: []const u8) !?[]const u8 {
    var f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var lead: [8]u8 = undefined;
    if (try f.readPositionalAll(io, &lead, 0) != 8) return error.BadHeader;
    const n = std.mem.readInt(u64, &lead, .little);
    if (n > 1 << 26) return error.BadHeader;
    const text = try a.alloc(u8, @intCast(n));
    if (try f.readPositionalAll(io, text, 8) != n) return error.BadHeader;
    return metadataQuantization(a, text);
}

/// `__metadata__.quantization` of a safetensors header's JSON, or null.
pub fn metadataQuantization(a: std.mem.Allocator, header: []const u8) !?[]const u8 {
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, a, header, .{});
    if (doc != .object) return error.BadHeader;
    const meta = doc.object.get("__metadata__") orelse return null;
    if (meta != .object) return null;
    const q = meta.object.get("quantization") orelse return null;
    return if (q == .string) q.string else null;
}

const testing = std.testing;

fn parsed(a: std.mem.Allocator, text: []const u8) !Config {
    var why: config.Why = .{};
    return config.parse(a, text, &why);
}

test "a mixed-width table resolves each matrix to its own entry, the rest to the checkpoint's width" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const stock = @embedFile("../fixtures/config.json");
    const mixed = try std.mem.replaceOwned(u8, a, stock, "\"group_size\": 64,", "\"group_size\": 64, " ++
        "\"backbone.embeddings\": {\"bits\": 8, \"group_size\": 64}, " ++
        "\"lm_head\": {\"bits\": 8, \"group_size\": 64}, " ++
        "\"backbone.layers.0.mixer.in_proj\": {\"bits\": 5, \"group_size\": 64}, " ++
        "\"backbone.layers.0.mixer.out_proj\": {\"bits\": 6, \"group_size\": 64}, " ++
        "\"backbone.layers.1.mixer.shared_experts.up_proj\": {\"bits\": 8, \"group_size\": 64},");
    try testing.expect(!std.mem.eql(u8, mixed, stock));
    const c = try parsed(a, mixed);
    try testing.expectEqual(Width{ .bits = 8 }, of(&c, "backbone.embeddings").?);
    try testing.expectEqual(Width{ .bits = 8 }, of(&c, "lm_head").?);
    try testing.expectEqual(Width{ .bits = 5 }, of(&c, "backbone.layers.0.mixer.in_proj").?);
    try testing.expectEqual(Width{ .bits = 6 }, of(&c, "backbone.layers.0.mixer.out_proj").?);
    try testing.expectEqual(Width{ .bits = 8 }, of(&c, "backbone.layers.1.mixer.shared_experts.up_proj").?);
    try testing.expectEqual(Width{ .bits = 4 }, of(&c, "backbone.layers.1.mixer.shared_experts.down_proj").?);
    try testing.expectEqual(Width{ .bits = 4 }, of(&c, "backbone.layers.1.mixer.switch_mlp.fc1").?);
    try testing.expectEqual(Width{ .bits = 4 }, of(&c, "backbone.layers.2.mixer.in_proj").?);
    try testing.expectEqual(@as(?Width, null), of(&c, "backbone.layers.1.mixer.gate"));
    try testing.expectEqual(@as(?Width, null), of(&c, "backbone.layers.0.mixer.q_proj"));
    try testing.expectEqual(@as(?Width, null), of(&c, "backbone.layers.99.mixer.in_proj"));
    var buf: [96]u8 = undefined;
    try testing.expectEqual(@as(?config.Named, null), unserved(&c, &buf));
    try testing.expectEqualStrings("mlx-q4g64+q5g64+q6g64+q8g64", label(&c, &buf));
}

test "the stock 4-bit checkpoint reads every matrix at 4-bit in groups of 64, as before" {
    const c = try parsed(testing.allocator, @embedFile("../fixtures/config.json"));
    var buf: [96]u8 = undefined;
    try testing.expectEqualStrings("mlx-q4g64", label(&c, &buf));
    try testing.expectEqual(Width{}, of(&c, "lm_head").?);
    try testing.expectEqual(Width{}, of(&c, "backbone.embeddings").?);
    for (0..c.layers) |i| for (config.members(c.kinds[i])) |m| {
        const key = try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer.{s}", .{ i, m });
        try testing.expectEqual(Width{ .bits = 4, .group = 64 }, of(&c, key).?);
    };
}

test "a width the launches do not read is named, the first one found" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const stock = @embedFile("../fixtures/config.json");
    const three = try std.mem.replaceOwned(u8, a, stock, "\"group_size\": 64,", "\"group_size\": 64, " ++
        "\"backbone.layers.5.mixer.q_proj\": {\"bits\": 3, \"group_size\": 64},");
    var buf: [96]u8 = undefined;
    const c = try parsed(a, three);
    const n = unserved(&c, &buf).?;
    try testing.expectEqualStrings("backbone.layers.5.mixer.q_proj", n.name);
    try testing.expectEqual(@as(u8, 3), n.width.bits);
}

test "the MTP head's width comes from its own metadata, the stock 4/64 when it states none" {
    var arena: std.heap.ArenaAllocator = .init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    try testing.expectEqual(Width{ .bits = 4, .group = 64 }, try headOf(a, null));
    try testing.expectEqual(Width{ .bits = 4, .group = 64 }, try headOf(a, "{\"group_size\": 64, \"bits\": 4}"));
    try testing.expectEqual(Width{ .bits = 8, .group = 32 }, try headOf(a, "{\"group_size\": 32, \"bits\": 8}"));
    try testing.expectError(error.BadHeadWidth, headOf(a, "{\"bits\": 4}"));
    try testing.expectError(error.BadHeadWidth, headOf(a, "{\"bits\": 4, \"group_size\": 64, \"mode\": \"mxfp4\"}"));
    const header = "{\"__metadata__\": {\"format\": \"tensorfold-nemotron-mtp\", " ++
        "\"quantization\": \"{\\\"group_size\\\": 64, \\\"bits\\\": 4}\"}, \"a\": {}}";
    const q = (try metadataQuantization(a, header)).?;
    try testing.expectEqual(Width{ .bits = 4, .group = 64 }, try headOf(a, q));
    try testing.expectEqual(@as(?[]const u8, null), try metadataQuantization(a, "{\"a\": {}}"));
}
