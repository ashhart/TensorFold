//! The complete finite scalar domain of the six Flash Next contract roles.
const window = @import("window_config.zig");
const rotary = @import("rotary_config.zig");
// positioned: fz_idx_pool_pos, image streams' indexer blocks at their first rows' three positions (M-RoPE axes)
pub const Role = enum { mrope, target, mtp, pooled, absolute, positioned };
pub const Mode = enum { normal, yarn };
pub const maximum = window.maximum;
pub const chunk_cases = 65536;
pub const Point = struct { position: u32, index: u32, axis: u32 };

pub fn stride(role: Role) u32 {
    return if (role == .pooled or role == .absolute) 4 else 1;
}

pub fn count(role: Role) u64 {
    return (maximum / stride(role)) * 32;
}

pub fn point(role: Role, linear: u64) Point {
    const i: u32 = @intCast(linear % 32);
    return .{ .position = @intCast((linear / 32) * stride(role)), .index = i, .axis = if (role != .mrope and role != .positioned) 0 else if (i % 3 == 1 and i < 33) 1 else if (i % 3 == 2 and i < 30) 2 else 0 };
}

pub fn table(mode: Mode) [rotary.words]f32 {
    return rotary.table(23.253496170043945, if (mode == .yarn) 4 else 0, window.trained);
}

pub fn modeOf(values: []const f32) !Mode {
    const std = @import("std");
    for ([_]Mode{ .normal, .yarn }) |mode| {
        const wanted = table(mode);
        if (std.mem.eql(u8, std.mem.sliceAsBytes(values), std.mem.asBytes(&wanted))) return mode;
    }
    return error.UnrecordedRotaryTable;
}
