//! Packed expert metadata: checkpoint tensor headers, not config quantization labels, select the format.
const std = @import("std");
const core = @import("core");
const cfg = @import("../config.zig");
const Tensor = core.safetensors.Tensor;

pub const Held = struct {
    counts: [cfg.layers]u16 = @splat(cfg.experts),
    ids: [cfg.layers][cfg.experts]u8 = @splat(identity),
    pub const identity = blk: {
        var ids: [cfg.experts]u8 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast(i);
        break :blk ids;
    };
    pub fn slots(h: *const Held, layer: usize) [cfg.experts]i32 {
        var out: [cfg.experts]i32 = @splat(-1);
        for (h.ids[layer][0..h.counts[layer]], 0..) |id, slot| out[id] = @intCast(slot);
        return out;
    }
};

pub fn parseHeld(value: std.json.Value) !Held {
    if (value != .object or value.object.count() != cfg.layers) return error.BadExpertMap;
    var h: Held = .{};
    for (0..cfg.layers) |li| {
        var key: [8]u8 = undefined;
        const ids = value.object.get(try std.fmt.bufPrint(&key, "{d}", .{li})) orelse return error.BadExpertMap;
        if (ids != .array or ids.array.items.len < cfg.top_k or ids.array.items.len > cfg.experts) return error.BadExpertMap;
        for (ids.array.items, 0..) |id, s| {
            if (id != .integer or id.integer < 0 or id.integer >= cfg.experts) return error.BadExpertMap;
            if (s > 0 and id.integer <= h.ids[li][s - 1]) return error.BadExpertMap;
            h.ids[li][s] = @intCast(id.integer);
        }
        h.counts[li] = @intCast(ids.array.items.len);
    }
    return h;
}

pub fn heldMetadata(metadata: ?std.json.Value) !Held {
    const meta = metadata orelse return .{};
    if (meta != .object) return error.BadExpertMap;
    if (meta.object.get("drama_head_rows") != null) return error.UnsupportedCompactHead;
    return if (meta.object.get("drama_experts")) |v| try parseHeld(v) else .{};
}

pub fn checkNames(h: *const Held, names: anytype) !void {
    var it = names;
    const pre = "model.language_model.layers.";
    while (it.next()) |key| {
        const name = key.*;
        if (!std.mem.startsWith(u8, name, pre)) continue;
        var parts = std.mem.splitScalar(u8, name[pre.len..], '.');
        const li = parts.next() orelse continue;
        if (!std.mem.eql(u8, parts.next() orelse continue, "mlp")) continue;
        if (!std.mem.eql(u8, parts.next() orelse continue, "experts")) continue;
        const l = std.fmt.parseInt(usize, li, 10) catch return error.BadExpertMap;
        const e = std.fmt.parseInt(u8, parts.next() orelse return error.BadExpertMap, 10) catch return error.BadExpertMap;
        if (l >= cfg.layers or std.mem.indexOfScalar(u8, h.ids[l][0..h.counts[l]], e) == null) return error.UnlistedExpert;
    }
}

pub fn nvfp4(w: Tensor, s: Tensor, g: Tensor, n: usize, k: usize) !void {
    if (k % 16 != 0 or !w.is(.u8, &.{ n, k / 2 }) or !s.is(.f8_e4m3, &.{ n, k / 16 }) or !g.is(.f32, &.{})) return error.BadNvfp4Shape;
    if (w.bytes.len != n * k / 2 or s.bytes.len != n * k / 16 or g.bytes.len != 4) return error.BadNvfp4Shape;
    for (s.bytes) |b| if (b == 0x7f or b > 0x80) return error.BadNvfp4Scale;
    const global: f32 = @bitCast(std.mem.readInt(u32, g.bytes[0..4], .little));
    if (!(global > 0 and std.math.isFinite(global))) return error.BadScale;
}

pub fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @truncate((bits +% (0x7fff + ((bits >> 16) & 1))) >> 16);
}
