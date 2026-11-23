//! Strict scalar-rank PEFT LoRA configuration, including target-module admission.
const std = @import("std");

pub const Config = struct {
    rank: usize,
    alpha: f64,
    scale: f64,
    target_modules: ?[]const []const u8 = null,

    pub fn validate(c: Config) !void {
        const end = std.math.pow(f64, 2, @bitSizeOf(usize));
        const rf: f64 = @floatFromInt(c.rank);
        if (c.rank == 0 or rf < 1 or rf >= end) return error.InvalidRank;
        if (!std.math.isFinite(c.alpha) or c.alpha < 0) return error.InvalidAlpha;
        if (c.scale != c.alpha / rf or !std.math.isFinite(c.scale)) return error.InvalidAlpha;
        if (c.target_modules) |declared| {
            if (declared.len == 0) return error.InvalidTargetModules;
            for (declared, 0..) |target, i| {
                if (target.len == 0 or std.mem.eql(u8, target, "all-linear")) return error.UnsupportedTargetModules;
                for (declared[0..i]) |other| if (std.mem.eql(u8, target, other)) return error.InvalidTargetModules;
            }
        }
    }

    pub fn targets(c: Config, module: []const u8) bool {
        const patterns = c.target_modules orelse return true;
        for (patterns) |pattern| {
            if (std.mem.eql(u8, module, pattern)) return true;
            if (module.len > pattern.len + 1 and std.mem.endsWith(u8, module, pattern) and module[module.len - pattern.len - 1] == '.') return true;
        }
        return false;
    }
};

fn number(v: ?std.json.Value) ?f64 {
    return switch (v orelse return null) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn enabled(o: std.json.ObjectMap, key: []const u8) bool {
    const value = o.get(key) orelse return false;
    return value == .bool and value.bool;
}

fn nonEmpty(o: std.json.ObjectMap, key: []const u8) bool {
    const value = o.get(key) orelse return false;
    return switch (value) {
        .object => |m| m.count() != 0,
        .array => |a| a.items.len != 0,
        .null => false,
        else => true,
    };
}

pub fn parse(arena: std.mem.Allocator, json: []const u8) !Config {
    const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, json, .{});
    if (root != .object) return error.InvalidAdapterConfig;
    const o = root.object;
    if (o.get("peft_type")) |kind| if (kind != .string or !std.mem.eql(u8, kind.string, "LORA")) return error.UnsupportedAdapterType;
    for ([_][]const u8{ "use_dora", "use_rslora", "fan_in_fan_out" }) |key| if (enabled(o, key)) return error.UnsupportedLoRAVariant;
    for ([_][]const u8{ "rank_pattern", "alpha_pattern", "modules_to_save" }) |key| if (nonEmpty(o, key)) return error.UnsupportedLoRAVariant;
    if (o.get("bias")) |bias| if (bias != .string or !std.mem.eql(u8, bias.string, "none")) return error.UnsupportedLoRAVariant;
    const rank_value = number(o.get("r")) orelse return error.InvalidRank;
    const alpha = number(o.get("lora_alpha")) orelse return error.InvalidAlpha;
    const end = std.math.pow(f64, 2, @bitSizeOf(usize));
    if (!std.math.isFinite(rank_value) or rank_value < 1 or rank_value != @floor(rank_value) or rank_value >= end) return error.InvalidRank;
    const rank: usize = @intFromFloat(rank_value);
    var targets: ?[]const []const u8 = null;
    if (o.get("target_modules")) |value| {
        if (value != .array) return error.InvalidTargetModules;
        const owned = try arena.alloc([]const u8, value.array.items.len);
        for (value.array.items, owned) |entry, *target| {
            if (entry != .string) return error.InvalidTargetModules;
            target.* = entry.string;
        }
        targets = owned;
    }
    const result: Config = .{ .rank = rank, .alpha = alpha, .scale = alpha / rank_value, .target_modules = targets };
    try result.validate();
    return result;
}
