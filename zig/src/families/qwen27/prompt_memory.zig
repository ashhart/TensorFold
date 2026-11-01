//! Startup reserves every target slot and the shared prompt slabs before selecting a frame width.
const std = @import("std");
const Config = @import("config.zig").Config;
const frame = @import("gpu_frame.zig");
const planning = @import("attention_plan.zig");
const Budget = @import("gdn_pool.zig").Budget;

pub const margin: usize = 8 << 30;
/// RAM a suggested wired limit leaves outside the GPU for macOS and other processes.
const floor: usize = 4 << 30;

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

/// Frame bytes already belong to the model; these are the runner's future bytes at its peak (`planes`: its tap planes).
pub fn runnerBytes(c: Config, context: u32, slots: u32, rows: u32, shared: bool, ring: usize, planes: usize) !usize {
    return runnerBytesTaps(c, context, slots, rows, shared, ring, tapRowBytes(c.hidden, planes, false));
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
    /// The runner's tap planes: the 27B's five, or the drafter's own count.
    planes: usize = 5,
    tap_row: usize = 0, // tapRowBytes for the drafter's taps; 0: five raw planes
    current_rows: u32,
    left: usize,

    pub fn cost(f: Fit, rows: u32, slots: u32) !usize {
        return (try runnerBytesTaps(f.config, f.context, slots, rows, f.shared, f.ring, if (f.tap_row > 0) f.tap_row else tapRowBytes(f.config.hidden, f.planes, false))) + (frameBytes(f.config, rows) -| frameBytes(f.config, f.current_rows));
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
    return (try workingSet(real, cap)) -| used -| margin;
}

/// The iogpu.wired_limit_mb that fits `need` beside `used` and the margin while leaving the floor; null if RAM cannot.
pub fn wiredRaise(used: usize, need: usize, ram: usize) ?usize {
    const total = used +| need +| margin;
    const mb = std.math.divCeil(usize, total, 1 << 20) catch unreachable;
    return if (mb *| (1 << 20) <= ram -| floor) mb else null;
}

/// The tail of a startup refusal: the one change that fits a stream beside the weights and the margin.
pub const Advice = union(enum) {
    lower,
    raise: struct { set: usize, mb: usize },
    no_raise: ?usize,

    pub fn format(a: Advice, w: *std.Io.Writer) !void {
        switch (a) {
            .lower => try w.writeAll(": lower --parallel or --context"),
            .raise => |r| try w.print(": Metal lets this Mac use {d:.1} GiB; sudo sysctl iogpu.wired_limit_mb={d} raises that limit enough for one stream at this context until reboot (or lower --context)", .{ gibs(r.set), r.mb }),
            .no_raise => |ram| if (ram) |bytes| try w.print(": no wired-limit raise fits one stream and leaves {d} GiB of this Mac's {d:.1} GiB of RAM to the system: lower --context", .{ floor >> 30, gibs(bytes) }) else try w.writeAll(": no wired-limit raise is known to fit one stream: lower --context"),
        }
    }
};

/// What a refusal should advise for one stream, from the set and margin the startup room counted.
pub fn advice(real: usize, used: usize, cap: ?[]const u8, one: usize, ram: ?usize) Advice {
    const set = workingSet(real, cap) catch return .lower;
    if (set != real or one <= set -| used -| margin) return .lower;
    if (ram) |r| if (wiredRaise(used, one, r)) |mb| return .{ .raise = .{ .set = set, .mb = mb } };
    return .{ .no_raise = ram };
}

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / @as(f64, @floatFromInt(1 << 30));
}
