const std = @import("std");

pub const Spec = struct {
    bits: i32 = 4,
    group_size: i32 = 64,

    pub fn validate(s: Spec) !void {
        switch (s.bits) {
            2, 3, 4, 5, 6, 8 => {},
            else => return error.UnsupportedQuantization,
        }
        switch (s.group_size) {
            32, 64, 128 => {},
            else => return error.UnsupportedQuantization,
        }
    }

    pub fn shape(s: Spec, weight: []const i32, scales: []const i32, biases: []const i32) !struct { n: i32, k: i32 } {
        try s.validate();
        if (weight.len != 2 or scales.len != 2 or !std.mem.eql(i32, scales, biases)) return error.InvalidTensorShape;
        if (weight[0] <= 0 or weight[1] <= 0 or scales[0] != weight[0] or scales[1] <= 0) return error.InvalidTensorShape;
        const k = std.math.mul(i32, scales[1], s.group_size) catch return error.InvalidTensorShape;
        if (@as(i64, weight[1]) * 32 != @as(i64, k) * s.bits) return error.InvalidTensorShape;
        return .{ .n = weight[0], .k = k };
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

fn block(config: std.json.Value) ?std.json.ObjectMap {
    if (config != .object) return null;
    for ([_]std.json.Value{ config, config.object.get("text_config") orelse .null }) |source| {
        if (source != .object) continue;
        for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
            const v = source.object.get(key) orelse continue;
            if (v == .object and v.object.count() > 0) return v.object;
        }
    }
    return null;
}

fn integer(o: std.json.ObjectMap, key: []const u8, fallback: ?i32) !i32 {
    const v = o.get(key) orelse return fallback orelse error.UnsupportedQuantization;
    if (v != .integer) return error.UnsupportedQuantization;
    return std.math.cast(i32, v.integer) orelse error.UnsupportedQuantization;
}

fn parse(o: std.json.ObjectMap, fallback: ?i32) !Spec {
    if (o.get("mode")) |v| {
        if (v != .null and (v != .string or (v.string.len > 0 and !std.mem.eql(u8, v.string, "affine")))) return error.UnsupportedQuantization;
    }
    const s = Spec{ .bits = try integer(o, "bits", fallback), .group_size = try integer(o, "group_size", 64) };
    try s.validate();
    return s;
}

pub fn resolve(config: std.json.Value, path: ?[]const u8) !?Spec {
    const b = block(config) orelse return null;
    if (b.get("quant_method")) |v| {
        if (v != .null and (v != .string or (!std.mem.eql(u8, v.string, "mlx") and !std.mem.eql(u8, v.string, "affine")))) return error.UnsupportedQuantization;
    }
    const global = try parse(b, null);
    const wanted = canonical(path orelse return global);
    var found = false;
    var result: ?Spec = global;
    var entries = b.iterator();
    while (entries.next()) |e| {
        if (!std.mem.eql(u8, canonical(e.key_ptr.*), wanted)) continue;
        const v = e.value_ptr.*;
        const spec: ?Spec = switch (v) {
            .bool => |yes| if (yes) global else null,
            .object => |o| if (o.count() == 0) null else try parse(o, 4),
            else => return error.UnsupportedQuantization,
        };
        if (found and !std.meta.eql(result, spec)) return error.ConflictingQuantizationAliases;
        result = spec;
        found = true;
    }
    return result;
}

pub fn nativeLanes(config: std.json.Value) !bool {
    const global = (try resolve(config, null)) orelse return false;
    if (!tensorFormat(global)) return false;
    const values = block(config) orelse return false;
    var it = values.iterator();
    while (it.next()) |e| {
        if (e.value_ptr.* != .object and e.value_ptr.* != .bool) continue;
        if (std.mem.endsWith(u8, e.key_ptr.*, "embed_tokens")) continue;
        var components = std.mem.splitScalar(u8, e.key_ptr.*, '.');
        var vision = false;
        while (components.next()) |part| if (std.mem.eql(u8, part, "vision_tower") or std.mem.eql(u8, part, "visual")) {
            vision = true;
        };
        if (vision) continue;
        if (try resolve(config, e.key_ptr.*)) |spec| if (!tensorFormat(spec)) return false;
    }
    return true;
}
fn tensorFormat(s: Spec) bool {
    return s.group_size == 64 or (s.bits == 4 and s.group_size == 32);
}

test "backend follows all language overrides but ignores vision and embeddings" {
    for ([_][]const u8{
        \\{"quantization":{"bits":4,"group_size":64,"vision_tower.a":{"bits":2,"group_size":128},"model.embed_tokens":{"bits":8,"group_size":128}}}
        ,
        \\{"quantization":{"bits":4,"group_size":64,"model.layers.0.q":{"bits":2,"group_size":128}}}
        ,
        \\{"quantization":{"bits":4,"group_size":32,"model.layers.0.q":false}}
    }, [_]bool{ true, false, true }) |json, expected| {
        const p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
        defer p.deinit();
        try std.testing.expectEqual(expected, try nativeLanes(p.value));
    }
}

test "affine formats validate packed shapes including word crossings" {
    for ([_]i32{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]i32{ 32, 64, 128 }) |group| {
        const s = Spec{ .bits = bits, .group_size = group };
        const geometry = try s.shape(&.{ 17, @divExact(128 * bits, 32) }, &.{ 17, @divExact(128, group) }, &.{ 17, @divExact(128, group) });
        try std.testing.expectEqual(@as(i32, 128), geometry.k);
        try std.testing.expectError(error.InvalidTensorShape, s.shape(&.{ 17, @divExact(128 * bits, 32) + 1 }, &.{ 17, @divExact(128, group) }, &.{ 17, @divExact(128, group) }));
    };
    try std.testing.expectError(error.UnsupportedQuantization, (Spec{ .bits = 7 }).validate());
}

test "MLX overrides default to four bits and 64 groups independently of global" {
    const p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator,
        \\{"quantization":{"bits":8,"group_size":128,"language_model.model.layers.0.q":{"group_size":32},"model.layers.1.q":false,"lm_head":{},"model.layers.2.q":true}}
    , .{});
    defer p.deinit();
    try std.testing.expectEqual(Spec{ .bits = 8, .group_size = 128 }, (try resolve(p.value, null)).?);
    try std.testing.expectEqual(Spec{ .bits = 4, .group_size = 32 }, (try resolve(p.value, "model.layers.0.q.weight")).?);
    try std.testing.expectEqual(@as(?Spec, null), try resolve(p.value, "layers.1.q"));
    try std.testing.expectEqual(@as(?Spec, null), try resolve(p.value, "lm_head"));
    try std.testing.expectEqual(Spec{ .bits = 8, .group_size = 128 }, (try resolve(p.value, "layers.2.q")).?);
}

test "conflicting aliases and invalid override types fail" {
    for ([_][]const u8{
        \\{"quantization":{"bits":4,"a":{"bits":2},"model.a":{"bits":8}}}
        ,
        \\{"quantization":{"bits":4,"a":null}}
    }, [_]anyerror{ error.ConflictingQuantizationAliases, error.UnsupportedQuantization }) |json, err| {
        const p = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
        defer p.deinit();
        try std.testing.expectError(err, resolve(p.value, "a"));
    }
}
