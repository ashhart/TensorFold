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

/// Frame bytes already belong to the model; these are the runner's future bytes at its peak.
pub fn runnerBytes(c: Config, context: u32, slots: u32, rows: u32, shared: bool, ring: usize) !usize {
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
    const taps = buffers([_]usize{@as(usize, rows) * 5 * c.hidden * 2});
    return kv + bindings + before + taps + buffers(scratch.sizes) + buffers(budget.allocationSizes()) + slots * (ring + (256 << 20));
}

pub const Fit = struct {
    config: Config,
    context: u32,
    slots: u32,
    shared: bool,
    ring: usize = 0,
    current_rows: u32,
    left: usize,

    pub fn cost(f: Fit, rows: u32, slots: u32) !usize {
        return (try runnerBytes(f.config, f.context, slots, rows, f.shared, f.ring)) + (frameBytes(f.config, rows) -| frameBytes(f.config, f.current_rows));
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
