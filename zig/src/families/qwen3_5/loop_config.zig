//! Validated SenseNova loop and mHC settings, independent of device state.
const std = @import("std");

pub const Mhc = struct {
    start: usize,
    end: usize,
    streams: usize,
    iterations: usize,
    fixed_identity: bool,
    pre_scale: f64,
};

pub const Config = struct {
    start: usize,
    end: usize,
    count: usize,
    mhc: ?Mhc,

    pub fn withLoops(c: Config, loops: usize) error{BadLoops}!Config {
        if (loops < 1 or loops > 4) return error.BadLoops;
        var result = c;
        result.count = loops;
        return result;
    }
};

fn object(v: ?std.json.Value) error{BadLoopConfig}!std.json.ObjectMap {
    const x = v orelse return error.BadLoopConfig;
    if (x != .object) return error.BadLoopConfig;
    return x.object;
}

fn uint(o: std.json.ObjectMap, key: []const u8) error{BadLoopConfig}!usize {
    const x = o.get(key) orelse return error.BadLoopConfig;
    if (x != .integer or x.integer < 0) return error.BadLoopConfig;
    return std.math.cast(usize, x.integer) orelse error.BadLoopConfig;
}

fn known(o: std.json.ObjectMap, keys: []const []const u8) error{BadLoopConfig}!void {
    var it = o.iterator();
    while (it.next()) |entry| {
        var found = false;
        for (keys) |key| if (std.mem.eql(u8, entry.key_ptr.*, key)) {
            found = true;
            break;
        };
        if (!found) return error.BadLoopConfig;
    }
}

pub fn parse(root: std.json.ObjectMap, layers: usize) error{BadLoopConfig}!Config {
    const loop = try object(root.get("loop_settings"));
    if (loop.get("use_gate")) |gate| if (gate != .bool or gate.bool) return error.BadLoopConfig; // no loop gate
    const start = try uint(loop, "start");
    const end = try uint(loop, "end");
    const count = try uint(loop, "eval_count");
    if (start >= end or end > layers or count == 0) return error.BadLoopConfig;
    const counts = loop.get("counts") orelse return error.BadLoopConfig;
    if (counts != .array or counts.array.items.len != 1) return error.BadLoopConfig;
    const item = counts.array.items[0];
    if (item != .integer or item.integer < 0 or std.math.cast(usize, item.integer) != count) return error.BadLoopConfig;
    var result: Config = .{ .start = start, .end = end, .count = count, .mhc = null };
    if (root.get("mhc")) |value| if (value != .null) {
        const m = try object(value);
        try known(m, &.{ "start", "end", "streams", "iterations", "fixed_identity", "pre_scale" });
        const ms = try uint(m, "start");
        const me = try uint(m, "end");
        const streams = try uint(m, "streams");
        const iterations = try uint(m, "iterations");
        const fixed = m.get("fixed_identity") orelse return error.BadLoopConfig;
        const scale: f64 = if (m.get("pre_scale")) |v| switch (v) {
            .integer => |i| @floatFromInt(i),
            .float => |f| f,
            else => return error.BadLoopConfig,
        } else 1.0;
        if (ms >= me or me > layers or (streams != 2 and streams != 4) or iterations == 0 or fixed != .bool or
            !std.math.isFinite(scale) or scale <= 0) return error.BadLoopConfig;
        if (!(me <= start or (ms >= start and me <= end) or ms >= end)) return error.BadLoopConfig;
        result.mhc = .{ .start = ms, .end = me, .streams = streams, .iterations = iterations, .fixed_identity = fixed.bool, .pre_scale = scale };
    };
    return result;
}
