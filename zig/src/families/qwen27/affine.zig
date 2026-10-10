//! Resolve checkpoint affine formats and per-module aliases without changing native words or floating metadata.
const std = @import("std");
const object = @import("config.zig").object;

pub const Spec = struct {
    bits: u8,
    group_size: usize = 64,

    pub fn check(s: Spec) !void {
        if (std.mem.indexOfScalar(u8, &.{ 2, 3, 4, 5, 6, 8 }, s.bits) == null or
            std.mem.indexOfScalar(usize, &.{ 32, 64, 128 }, s.group_size) == null) return error.BadAffine;
    }
};

pub fn canonical(path: []const u8) []const u8 {
    var p = path;
    if (std.mem.endsWith(u8, p, ".weight")) p = p[0 .. p.len - 7];
    for ([_][]const u8{ "model.language_model.", "language_model.", "text_model.", "model." }) |prefix| {
        if (std.mem.startsWith(u8, p, prefix)) p = p[prefix.len..];
    }
    return p;
}

fn spec(o: std.json.ObjectMap, fallback: ?u8) !Spec {
    const bits = if (o.get("bits")) |v| blk: {
        if (v != .integer or v.integer < 0 or v.integer > 8) return error.BadAffine;
        break :blk @as(u8, @intCast(v.integer));
    } else fallback orelse return error.BadAffine;
    if (std.mem.indexOfScalar(u8, &.{ 2, 3, 4, 5, 6, 8 }, bits) == null) return error.BadAffine;
    const group = if (o.get("group_size")) |v| blk: {
        if (v != .integer or v.integer < 0 or v.integer > 128) return error.BadAffine;
        break :blk @as(usize, @intCast(v.integer));
    } else 64;
    if (std.mem.indexOfScalar(usize, &.{ 32, 64, 128 }, group) == null) return error.BadAffine;
    if (o.get("mode")) |v| {
        if (v != .null and !(v == .string and (v.string.len == 0 or std.mem.eql(u8, v.string, "affine")))) return error.BadAffine;
    }
    return .{ .bits = bits, .group_size = group };
}

fn block(root: std.json.ObjectMap) !?std.json.ObjectMap {
    const text = if (root.get("text_config")) |v| try object(v) else root;
    for ([_]std.json.ObjectMap{ root, text }) |o| {
        for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
            if (o.get(key)) |v| if (v == .object and v.object.count() > 0) return v.object;
        }
    }
    return null;
}

pub const Formats = struct {
    parsed: std.json.Parsed(std.json.Value),
    quant: ?std.json.ObjectMap,
    global: ?Spec,

    pub fn init(gpa: std.mem.Allocator, json: []const u8) !Formats {
        const p = try std.json.parseFromSlice(std.json.Value, gpa, json, .{});
        errdefer p.deinit();
        const q = try block(try object(p.value));
        if (q) |o| {
            if (o.get("quant_method")) |v| {
                if (v != .null and !(v == .string and (std.mem.eql(u8, v.string, "mlx") or std.mem.eql(u8, v.string, "affine")))) return error.NotAffine;
            }
            return .{ .parsed = p, .quant = q, .global = try spec(o, null) };
        }
        return .{ .parsed = p, .quant = null, .global = null };
    }

    pub fn deinit(f: *Formats) void {
        f.parsed.deinit();
        f.* = undefined;
    }

    pub fn resolve(f: *const Formats, path: ?[]const u8) !?Spec {
        const wanted = canonical(path orelse return f.global);
        const q = f.quant orelse return null;
        var matched = false;
        var found: ?Spec = null;
        var it = q.iterator();
        while (it.next()) |item| {
            if (!std.mem.eql(u8, canonical(item.key_ptr.*), wanted)) continue;
            const v = item.value_ptr.*;
            const own: ?Spec = switch (v) {
                .bool => |yes| if (yes) f.global else null,
                .object => |o| if (o.count() == 0) null else try spec(o, 4),
                else => return error.BadAffine,
            };
            if (matched and !std.meta.eql(found, own)) return error.ConflictingAliases;
            found = own;
            matched = true;
        }
        return if (matched) found else f.global;
    }
};

test "affine formats preserve all declared bit widths and groups" {
    const a = std.testing.allocator;
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        const json = try std.fmt.allocPrint(a, "{{\"quantization\":{{\"bits\":{d},\"group_size\":{d},\"mode\":\"affine\"}}}}", .{ bits, group });
        defer a.free(json);
        var f = try Formats.init(a, json);
        defer f.deinit();
        try std.testing.expectEqual(Spec{ .bits = bits, .group_size = group }, (try f.resolve("model.layers.0.self_attn.q_proj")).?);
    };
}

test "module overrides preserve dense and mixed native formats" {
    var f = try Formats.init(std.testing.allocator,
        \\{"quantization":{"bits":8,"layers.0.linear_attn.in_proj_a":false,
        \\"language_model.model.layers.0.linear_attn.in_proj_b":{},"model.layers.0.mlp.up_proj":{"group_size":32}}}
    );
    defer f.deinit();
    try std.testing.expectEqual(@as(?Spec, null), try f.resolve("language_model.model.layers.0.linear_attn.in_proj_a.weight"));
    try std.testing.expectEqual(@as(?Spec, null), try f.resolve("model.layers.0.linear_attn.in_proj_b"));
    try std.testing.expectEqual(Spec{ .bits = 4, .group_size = 32 }, (try f.resolve("model.layers.0.mlp.up_proj")).?);
    try std.testing.expectEqual(Spec{ .bits = 8 }, (try f.resolve(null)).?);
}

test "root quantization precedes text aliases and nested metadata is accepted" {
    var root = try Formats.init(std.testing.allocator, "{\"quantization\":{\"bits\":4},\"text_config\":{\"quantization_config\":{\"bits\":8}}}");
    defer root.deinit();
    try std.testing.expectEqual(@as(u8, 4), (try root.resolve(null)).?.bits);
    var nested = try Formats.init(std.testing.allocator, "{\"text_config\":{\"quantization_config\":{\"bits\":6,\"group_size\":128}}}");
    defer nested.deinit();
    try std.testing.expectEqual(Spec{ .bits = 6, .group_size = 128 }, (try nested.resolve(null)).?);
}

test "conflicting path aliases refuse the module" {
    var f = try Formats.init(std.testing.allocator, "{\"quantization\":{\"bits\":4,\"model.layers.0.mlp.up_proj\":false,\"language_model.model.layers.0.mlp.up_proj\":true}}");
    defer f.deinit();
    try std.testing.expectError(error.ConflictingAliases, f.resolve("layers.0.mlp.up_proj"));
}

test "unsupported methods and malformed native affine metadata refuse" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.NotAffine, Formats.init(a, "{\"quantization\":{\"bits\":4,\"quant_method\":\"modelopt\"}}"));
    for ([_][]const u8{
        "{\"quantization\":{\"bits\":7}}",
        "{\"quantization\":{\"bits\":true}}",
        "{\"quantization\":{\"bits\":4,\"group_size\":16}}",
        "{\"quantization\":{\"bits\":4,\"mode\":\"mxfp4\"}}",
    }) |json| try std.testing.expectError(error.BadAffine, Formats.init(a, json));
}

test "no affine metadata remains explicit rather than guessing four bits" {
    var f = try Formats.init(std.testing.allocator, "{}");
    defer f.deinit();
    try std.testing.expectEqual(@as(?Spec, null), try f.resolve("lm_head"));
    try std.testing.expectEqualStrings("layers.3.self_attn.q_proj", canonical("model.language_model.model.layers.3.self_attn.q_proj.weight"));
}
