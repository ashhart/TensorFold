//! Admission widths from the family descriptor and the checkpoint's effective per-module affine settings.
const std = @import("std");
const families = @import("families.zig");
const Family = @import("family.zig").Family;
const Quant = @import("family.zig").Quant;

pub const Mismatch = struct { name: []const u8, quant: Quant };

/// The first served matrix whose effective width differs from the family's writable width; invalid configs load later.
pub fn other(a: std.mem.Allocator, f: Family, root: std.json.ObjectMap) !?Mismatch {
    const names = f.widths orelse return error.NoWidthDescription;
    const o = families.scoped(f, root) orelse return error.BadConfig;
    const q = quantBlock(root) orelse quantBlock(o) orelse return error.BadConfig;
    if (q.get("mode")) |m| if (m != .string or !std.mem.eql(u8, m.string, "affine")) return error.BadConfig;
    const global = try width(q, .{ .bits = 0, .group = 64 });
    if (global.bits == 0 or global.group == 0) return error.BadConfig;
    if (try matrix(a, f, q, global, names.embedding, false)) |m| return m;
    const kinds = try families.kinds(a, f, o);
    for (kinds, 0..) |kind, i| {
        const members = switch (kind) {
            'M' => names.mamba,
            '*' => names.attention,
            'E' => names.expert,
            else => return error.BadConfig,
        };
        for (members) |member| {
            const path = try std.fmt.allocPrint(a, "{s}{d}.{s}{s}", .{ f.names.layers, i, names.mixer, member });
            var residual = false;
            for (names.residuals) |r| residual = residual or std.mem.eql(u8, member, r);
            if (try matrix(a, f, q, global, path, residual)) |m| return m;
        }
    }
    return matrix(a, f, q, global, names.head, false);
}

fn quantBlock(o: std.json.ObjectMap) ?std.json.ObjectMap {
    for ([_][]const u8{ "quantization", "quantization_config" }) |key| {
        if (o.get(key)) |v| if (v == .object and v.object.count() > 0) return v.object;
    }
    return null;
}

fn width(o: std.json.ObjectMap, fallback: Quant) !Quant {
    const bits = families.intField(o, "bits") orelse fallback.bits;
    const group = families.intField(o, "group_size") orelse fallback.group;
    return .{ .bits = std.math.cast(u8, bits) orelse return error.BadConfig, .group = std.math.cast(u16, group) orelse return error.BadConfig };
}

fn matrix(a: std.mem.Allocator, f: Family, q: std.json.ObjectMap, global: Quant, name: []const u8, residual: bool) !?Mismatch {
    var got = global;
    var it = q.iterator();
    while (it.next()) |entry| {
        var path = entry.key_ptr.*;
        for (f.widths.?.aliases) |alias| if (std.mem.startsWith(u8, path, alias)) {
            path = path[alias.len..];
            break;
        };
        if (!std.mem.eql(u8, path, name)) continue;
        const v = entry.value_ptr.*;
        got = switch (v) {
            .bool => if (v.bool) global else if (residual) f.quant else return error.BadConfig,
            .object => if (v.object.count() == 0) return error.BadConfig else try width(v.object, .{ .bits = 4, .group = 64 }),
            .null => return error.BadConfig,
            else => got,
        };
    }
    if (got.bits == f.quant.bits and got.group == f.quant.group) return null;
    return .{ .name = try a.dupe(u8, name), .quant = got };
}

test "Metal learner: effective widths use the descriptor, module defaults and served matrix order" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const json = "{\"hybrid_override_pattern\":\"ME*\",\"quantization\":{\"bits\":4,\"group_size\":64,\"unread.module\":{\"bits\":8},\"backbone.layers.1.mixer.shared_experts.up_proj\":{\"bits\":8}}}";
    const root = (try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{})).object;
    const f = families.nemotron_h;
    const bad = (try other(a, f, root)).?;
    try std.testing.expectEqualStrings("backbone.layers.1.mixer.shared_experts.up_proj", bad.name);
    try std.testing.expectEqual(@as(u8, 8), bad.quant.bits);
    try root.getPtr("quantization").?.object.put(a, bad.name, .{ .bool = true });
    try std.testing.expect((try other(a, f, root)) == null);
    var changed = f;
    changed.quant = .{ .bits = 8, .group = 32 };
    const mismatch = (try other(a, changed, root)).?;
    try std.testing.expectEqualStrings(f.widths.?.embedding, mismatch.name);
    const q = &root.getPtr("quantization").?.object;
    try q.put(a, "bits", .{ .integer = 8 });
    try q.put(a, "group_size", .{ .integer = 32 });
    try std.testing.expect((try other(a, changed, root)) == null);
    // A module dict inherits affine defaults, rather than the checkpoint's global width.
    try q.put(a, changed.widths.?.head, .{ .object = .{} });
    try std.testing.expectError(error.BadConfig, other(a, changed, root));
}
