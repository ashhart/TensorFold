//! Startup reserves every target slot and the shared prompt slabs before selecting a frame width.
const std = @import("std");
const Config = @import("config.zig").Config;
const frame = @import("gpu_frame.zig");
const planning = @import("attention_plan.zig");
const Budget = @import("gdn_pool.zig").Budget;

pub fn buffers(sizes: anytype) usize {
    var bytes: usize = 0;
    for (sizes) |size| bytes += std.mem.alignForward(usize, @max(size, 16), 16384);
    return bytes;
}

pub fn frameBytes(c: Config, rows: u32) usize {
    return buffers(frame.allocationSizes(c, rows)) + (if (rows > frame.small_rows) buffers(frame.decodeSizes(c, rows)) else 0);
}

/// A prompt row's tap bytes: `planes` raw planes, plus a fusing drafter's row-major copy, fc rows and fused row.
pub fn tapRowBytes(hidden: usize, planes: usize, fused: bool) usize {
    return (planes + if (fused) planes + 2 else 0) * hidden * 2;
}
test "qwen27 memory: an eight-tap fusing drafter's prompt taps cost what its planes and fusion buffers hold" {
    try std.testing.expectEqual(@as(usize, 5 * 5120 * 2), tapRowBytes(5120, 5, false));
    try std.testing.expectEqual(@as(usize, 18 * 4096 * 2), tapRowBytes(4096, 8, true));
}

/// Frame bytes already belong to the model; these are the runner's future bytes at its peak.
pub fn runnerBytes(c: Config, context: u32, slots: u32, rows: u32, shared: bool, ring: usize) !usize {
    return runnerBytesTaps(c, context, slots, rows, shared, ring, tapRowBytes(c.hidden, 5, false));
}
/// runnerBytes with the drafter's own tap row bytes (tapRowBytes).
pub fn runnerBytesTaps(c: Config, context: u32, slots: u32, rows: u32, shared: bool, ring: usize, tap_row: usize) !usize {
    var attention: usize = 0;
    for (0..c.layers) |i| attention += @intFromBool(c.kind(i) == .attention);
    const linear = c.layers - attention;
    const g = try planning.Geometry.fromConfig(c, shared);
    const limits = .{ .rows = rows, .streams = slots, .keys = context, .shared_prefix = shared, .geometry = g };
    const scratch = @import("attention_memory.zig").Layout.init(limits);
    const budget = try Budget.init(try @import("gdn_contract.zig").Shape.init(c), linear, slots, rows);
    const kv = attention * @as(usize, slots) * buffers([_]usize{ @as(usize, context) * c.kvDim() * 2, @as(usize, context) * c.kvDim() * 2 });
    const bindings = attention * buffers([_]usize{ (planning.global_words + planning.max_streams * planning.stream_words) * 4, planning.max_streams * 16, rows * 16 });
    const before = if (slots > 1) buffers([_]usize{budget.state}) else 0;
    const taps = buffers([_]usize{@as(usize, rows) * tap_row});
    return kv + bindings + before + taps + buffers(scratch.sizes) + buffers(budget.allocationSizes()) + slots * (ring + (256 << 20));
}

pub const Fit = struct {
    config: Config,
    context: u32,
    slots: u32,
    shared: bool,
    ring: usize = 0,
    tap_row: usize = 0, // tapRowBytes for the drafter's taps; 0: five raw planes
    current_rows: u32,
    left: usize,

    pub fn cost(f: Fit, rows: u32, slots: u32) !usize {
        return (try runnerBytesTaps(f.config, f.context, slots, rows, f.shared, f.ring, if (f.tap_row > 0) f.tap_row else tapRowBytes(f.config.hidden, 5, false))) + (frameBytes(f.config, rows) -| frameBytes(f.config, f.current_rows));
    }

    pub fn choose(f: Fit, requested: u32, explicit: bool) !u32 {
        if (explicit or requested <= 512) return requested;
        return if (try f.cost(requested, f.slots) <= f.left) requested else 512;
    }

    pub fn streams(f: Fit, rows: u32, fixed: bool) !u32 {
        var n = f.slots;
        while (n > 0) : (n -= 1) {
            if (try f.cost(rows, n) <= f.left) return n;
            if (fixed) break;
        }
        return error.TooManyStreams;
    }
};

/// A development cap can only reduce the device's reported working set.
pub fn workingSet(real: usize, text: ?[]const u8) !usize {
    const value = text orelse return real;
    const cap = std.fmt.parseInt(usize, value, 10) catch return error.BadWorkingSetCap;
    if (cap == 0) return error.BadWorkingSetCap;
    return @min(real, cap);
}

/// The same margin guards startup, cache growth and the first use of a later slot.
pub fn room(real: usize, used: usize, cap: ?[]const u8) !usize {
    return (try workingSet(real, cap)) -| used -| (8 << 30);
}
