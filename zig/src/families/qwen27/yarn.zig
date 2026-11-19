//! Static YaRN configuration parsing and frequency/scale computation for Qwen27 / SenseNova-Loop.
const std = @import("std");
const core = @import("core");
const rope_config = @import("rope_config");
const glue = @import("glue.zig");

pub const Yarn = struct {
    factor: f64,
    trained: usize,
    scale: f32,
    freq: [32]f32,
};

pub const Rope = union(enum) {
    plain,
    yarn: Yarn,

    /// Populate a Metal `glue.Rope` descriptor (leaves `yarn_scale == 0` for plain RoPE).
    pub fn apply(r: Rope, base: glue.Rope) glue.Rope {
        var out = base;
        switch (r) {
            .plain => {},
            .yarn => |y| {
                out.yarn_scale = y.scale;
                out.yarn_freq = y.freq;
            },
        }
        return out;
    }
};

pub const Resolved = struct { rope: Rope, ceiling: usize };

pub const Error = error{ UnsupportedRopeScaling, RopeYarnFactor, RopeYarnConflict, BadConfig };

/// Resolve YaRN from `rope_parameters` / `rope_scaling` or `--rope-yarn`; non-default YaRN params are refused.
pub fn resolve(tower: std.json.ObjectMap, root: std.json.ObjectMap, rope_opt: ?std.json.ObjectMap, theta: f64, max_pos: usize, flag: ?f64) Error!Resolved {
    if (flag) |f| if (!std.math.isFinite(f) or f < 1) return error.RopeYarnFactor;
    const host = if (tower.contains("rope_scaling")) tower else root;
    const legacy_v = host.get("rope_scaling");
    if (legacy_v) |lv| switch (lv) {
        .null => {},
        .object => |lo| return resolveBlock(lo, true, theta, max_pos, flag),
        else => return error.BadConfig,
    };
    if (rope_opt) |ro| return resolveBlock(ro, false, theta, max_pos, flag);
    return applyFlag(theta, max_pos, flag);
}

fn resolveBlock(block: std.json.ObjectMap, older: bool, theta: f64, max_pos: usize, flag: ?f64) Error!Resolved {
    const kind = block.get("rope_type") orelse block.get("type") orelse return if (older) error.UnsupportedRopeScaling else .{ .rope = .plain, .ceiling = max_pos };
    if (kind != .string) return error.UnsupportedRopeScaling;
    if (std.mem.eql(u8, kind.string, "default") and !older) return applyFlag(theta, max_pos, flag);
    if (!std.mem.eql(u8, kind.string, "yarn")) return error.UnsupportedRopeScaling;
    for ([_][]const u8{ "mscale", "mscale_all_dim", "attention_factor" }) |key| if (block.get(key)) |v| if (v != .null) return error.UnsupportedRopeScaling;
    for ([_]struct { []const u8, f64 }{ .{ "beta_fast", 32 }, .{ "beta_slow", 1 } }) |b| if (block.get(b[0])) |v| {
        if (v != .null and (try number(v)) != b[1]) return error.UnsupportedRopeScaling;
    };
    if (block.get("truncate")) |v| if (v != .null and !(v == .bool and v.bool)) return error.UnsupportedRopeScaling;
    const factor = try number(block.get("factor") orelse return error.UnsupportedRopeScaling);
    const tv = block.get("original_max_position_embeddings") orelse return error.UnsupportedRopeScaling;
    if (tv != .integer or tv.integer < 1 or tv.integer > std.math.maxInt(u32)) return error.UnsupportedRopeScaling;
    const trained: usize = @intCast(tv.integer);
    if (!(factor > 1) or @as(f64, @floatFromInt(trained)) * factor != @as(f64, @floatFromInt(max_pos))) return error.UnsupportedRopeScaling;
    if (flag) |f| if (f != factor) return error.RopeYarnConflict;
    if (comptime @hasDecl(core, "rope_yarn")) {
        const y = core.rope_yarn.yarn(theta, factor, trained);
        return .{ .rope = .{ .yarn = .{ .factor = factor, .trained = trained, .scale = y.scale, .freq = y.freq } }, .ceiling = max_pos };
    } else return error.UnsupportedRopeScaling;
}

fn applyFlag(theta: f64, max_pos: usize, flag: ?f64) Error!Resolved {
    const f = flag orelse return .{ .rope = .plain, .ceiling = max_pos };
    if (f == 1) return .{ .rope = .plain, .ceiling = max_pos };
    const ceil = @as(f64, @floatFromInt(max_pos)) * f;
    if (ceil != @floor(ceil) or ceil > std.math.maxInt(u32)) return error.RopeYarnFactor;
    if (comptime @hasDecl(core, "rope_yarn")) {
        const y = core.rope_yarn.yarn(theta, f, max_pos);
        return .{ .rope = .{ .yarn = .{ .factor = f, .trained = max_pos, .scale = y.scale, .freq = y.freq } }, .ceiling = @intFromFloat(ceil) };
    } else return error.RopeYarnFactor;
}

fn number(v: std.json.Value) Error!f64 {
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |x| if (std.math.isFinite(x)) x else error.UnsupportedRopeScaling,
        else => error.UnsupportedRopeScaling,
    };
}

test "SenseNova-Loop-9B 512K static YaRN (theta 1e7, factor 2 over 262144) matches core.rope_yarn" {
    const a = std.testing.allocator;
    const json_text =
        \\{"max_position_embeddings":524288,"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],
        \\ "partial_rotary_factor":0.25,"rope_theta":10000000,"rope_type":"yarn","factor":2.0,
        \\ "original_max_position_embeddings":262144,"truncate":true}}
    ;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, json_text, .{});
    defer parsed.deinit();
    const t = parsed.value.object;
    const r = try resolve(t, t, rope_config.params(t) catch unreachable, 10000000, 524288, null);
    try std.testing.expectEqual(@as(usize, 524288), r.ceiling);
    const y = core.rope_yarn.yarn(10000000, 2.0, 262144);
    try std.testing.expectEqual(y.scale, r.rope.yarn.scale);
    try std.testing.expectEqualSlices(f32, &y.freq, &r.rope.yarn.freq);
    try std.testing.expectError(error.RopeYarnConflict, resolve(t, t, rope_config.params(t) catch unreachable, 10000000, 524288, 4.0));
}
