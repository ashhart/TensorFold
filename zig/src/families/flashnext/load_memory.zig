//! The preflight floor is checked before checkpoint buffers or shaders are made.
const std = @import("std");
pub const prompt = @import("prompt_memory.zig");
pub const weights_plan = @import("load_weights.zig");
const window = struct {
    const trained = prompt.trained;
    const maximum = 4 * prompt.trained + 64;
};
pub const trained = window.trained;
pub const GiB: u64 = 1 << 30;
pub const margin = @import("sys").memory_policy.margin;
pub const options = "raise the OS wired limit where supported, use a smaller quantization, or reduce streams/context; native --ple-on-ssd is not implemented yet";

pub const Plan = struct {
    weights: u64,
    prompts: u64,
    attention: u64,
    states: u64,
    selectors: u64,
    keepalive: u64 = 64,
    reserve: u64 = margin,

    pub fn total(p: Plan) !u64 {
        var sum: u64 = 0;
        inline for (@typeInfo(Plan).@"struct".field_names) |name| sum = try std.math.add(u64, sum, @field(p, name));
        return sum;
    }
};

/// Explicit long-lived buffers, plus the shared margin for warm-up staging, driver/command buffers and small metadata.
pub fn plan(weights: u64, capacity: usize) !Plan {
    if (capacity < window.trained or capacity > window.maximum or capacity % 4 != 0) return error.ContextTooLong;
    const cap: u64 = capacity;
    const so = 48 * 128 * 128 * 4;
    return .{
        .weights = weights,
        .prompts = 2 * prompt.bytes(),
        .attention = 13 * cap * (2304 + 64),
        .states = 36 * (1 + 16 + 4) * (prompt.CS_ROW + so),
        .selectors = 2 * selector(prompt.PMAX, cap) + selector(16, cap) + 2 * (16 * cap + 16 * 2051 * 4 + 4096),
    };
}

fn selector(rows: u64, cap: u64) u64 {
    const scores = @min(rows * (cap / 4), @max(@as(u64, prompt.PMAX) * (window.trained / 4), 16 * (cap / 4))) * 4;
    return scores + rows * 2051 * 4 + 4 * @max(rows * 4, 64) + 13 * 256 + 5 * 64;
}

pub fn minimum(weights: u64, capacity: usize) !u64 {
    return (try plan(weights, capacity)).total();
}

pub fn admit(device: anytype, weights: u64, capacity: usize) !void {
    if (try minimum(weights, capacity) > device.maxWorkingSet()) return error.FlashNextMemoryShortfall;
}

/// A validated family ceiling always stays inside physical RAM and the GPU recommendation.
pub fn ceiling(ram: ?u64, working: u64, text: ?[]const u8) !u64 {
    const total = ram orelse return error.NoMemoryReading;
    if (total == 0 or working == 0) return error.NoMemoryReading;
    const top = @min(total, working);
    const value = std.mem.trim(u8, text orelse "", " \t");
    if (value.len == 0) return @min(top, total / 100 * 70);
    const gb = std.fmt.parseFloat(f64, value) catch return error.BadLoadLimit;
    if (!std.math.isFinite(gb) or gb <= 0) return error.BadLoadLimit;
    const bytes = gb * 1e9;
    if (bytes >= @as(f64, @floatFromInt(top))) return top;
    return @intFromFloat(bytes);
}

pub fn admitUnder(device: anytype, weights: u64, capacity: usize, ram: ?u64, text: ?[]const u8, extra: u64) !u64 {
    const cap = try ceiling(ram, device.maxWorkingSet(), text);
    if (try std.math.add(u64, try minimum(weights, capacity), extra) > cap) return error.FlashNextMemoryShortfall;
    return cap;
}

pub const Budget = struct {
    steady: u64,
    building: u64,
    pub fn peak(b: Budget) u64 {
        return b.forPhase(true);
    }
    pub fn forPhase(b: Budget, needs_build: bool) u64 {
        return if (needs_build) @max(b.steady, b.building) else b.steady;
    }
};

/// The pack phase precedes weight loading; charge its overlap, not a sum of disjoint phase peaks.
pub fn budget(weights: u64, capacity: usize, inventory: weights_plan.Inventory, extra: u64) !Budget {
    return budgetLive(weights, capacity, inventory, extra, 0);
}

pub fn budgetLive(source_files: u64, capacity: usize, inventory: weights_plan.Inventory, extra: u64, existing: u64) !Budget {
    _ = source_files; // source file padding and transformed source payloads are not resident decode weights
    const fixed = try std.math.add(u64, try minimum(inventory.raw, capacity), extra);
    return .{
        .steady = try std.math.add(u64, fixed, try inventory.resident()),
        .building = try std.math.add(u64, try std.math.add(u64, margin, existing), try inventory.building()),
    };
}

pub fn admitInventory(device: anytype, weights: u64, capacity: usize, ram: ?u64, text: ?[]const u8, extra: u64, inventory: weights_plan.Inventory) !u64 {
    const cap = try ceiling(ram, device.maxWorkingSet(), text);
    if ((try budget(weights, capacity, inventory, extra)).peak() > cap) return error.FlashNextMemoryShortfall;
    return cap;
}

pub fn admitPhase(device: anytype, weights: u64, capacity: usize, ram: ?u64, text: ?[]const u8, extra: u64, inventory: weights_plan.Inventory, existing: u64, needs_build: bool) !u64 {
    const cap = try ceiling(ram, device.maxWorkingSet(), text);
    return admitBudget(cap, try budgetLive(weights, capacity, inventory, extra, existing), needs_build);
}

pub fn admitBudget(cap: u64, value: Budget, needs_build: bool) !u64 {
    if (value.forPhase(needs_build) > cap) return error.FlashNextMemoryShortfall;
    return cap;
}

/// An explicit cache is real storage even when the later cache policy allows it past its ordinary budget.
pub fn withCache(settings: anytype, gib: ?f64) !@TypeOf(settings) {
    var out = settings;
    if (gib) |n| {
        if (!std.math.isFinite(n) or n < 0 or n >= 0x1p34) return error.BadCacheBudget;
        out.startup_extra = try std.math.add(u64, out.startup_extra, @as(u64, @intFromFloat(n * GiB)));
    }
    return out;
}

pub fn growingFits(held: u64, added: u64, cap: u64) bool {
    return held <= cap and added <= cap - held and margin <= cap - held - added;
}

/// Indexed shards counted once, including subfolders; metadata/stat only, no weight payload or packed cache read.
pub fn weightBytes(comptime checkpoint: type, a: std.mem.Allocator, io: std.Io, dir: []const u8) !u64 {
    const paths = try checkpoint.shardFiles(a, io, dir);
    defer checkpoint.freeShardFiles(a, paths);
    var total: u64 = 0;
    for (paths) |path| total = try std.math.add(u64, total, (try std.Io.Dir.cwd().statFile(io, path, .{})).size);
    return total;
}

test "Flash Next memory: a working set just above the old floor refuses before buffers" {
    const Device = struct {
        pub fn maxWorkingSet(_: @This()) u64 {
            return 10752 * GiB / 100;
        }
    };
    const weights = 10546 * GiB / 100;
    try std.testing.expect(weights + margin <= Device.maxWorkingSet(.{}));
    try std.testing.expectError(error.FlashNextMemoryShortfall, admit(Device{}, weights, 262144));
}

test "Flash Next memory: exact bound, overflow and long-window growth" {
    const Device = struct {
        cap: u64,
        pub fn maxWorkingSet(d: @This()) u64 {
            return d.cap;
        }
    };
    const bytes = try minimum(100 * GiB, window.trained);
    try admit(Device{ .cap = bytes }, 100 * GiB, window.trained);
    try std.testing.expectError(error.FlashNextMemoryShortfall, admit(Device{ .cap = bytes - 1 }, 100 * GiB, window.trained));
    try std.testing.expectError(error.Overflow, minimum(std.math.maxInt(u64), window.trained));
    try std.testing.expectError(error.ContextTooLong, minimum(0, window.maximum + 4));
    try std.testing.expect(try minimum(100 * GiB, window.maximum) > bytes);
    const p = try plan(0, window.trained);
    try std.testing.expectEqual(@as(u64, 2) * prompt.bytes(), p.prompts);
    try std.testing.expectEqual(margin, p.reserve);
    try std.testing.expectEqual(@as(u64, 64), p.keepalive);
    std.debug.print("Flash Next memory: prompt bytes={d}, runtime floor={d}, reported checkpoint minimum={d}\n", .{ prompt.bytes(), try p.total(), try minimum(10546 * GiB / 100, window.trained) });
}

test "Flash Next memory: ceiling caps, overrides and unknown readings refuse safely" {
    try std.testing.expectEqual(@as(u64, 70) * GiB, try ceiling(100 * GiB, 90 * GiB, null));
    try std.testing.expectEqual(@as(u64, 60) * GiB, try ceiling(100 * GiB, 60 * GiB, null));
    try std.testing.expectEqual(@as(u64, 85_500_000_000), try ceiling(100 * GiB, 90 * GiB, "85.5"));
    try std.testing.expectEqual(@as(u64, 90) * GiB, try ceiling(100 * GiB, 90 * GiB, "1000"));
    try std.testing.expectEqual(@as(u64, 100) * GiB, try ceiling(100 * GiB, 120 * GiB, "1000"));
    for ([_][]const u8{ "0", "-2", "nan", "inf", "no" }) |v| try std.testing.expectError(error.BadLoadLimit, ceiling(100 * GiB, 90 * GiB, v));
    try std.testing.expectError(error.NoMemoryReading, ceiling(null, 1, null));
    try std.testing.expectError(error.NoMemoryReading, ceiling(1, 0, null));
    try std.testing.expectEqual(std.math.maxInt(u64), try ceiling(std.math.maxInt(u64), std.math.maxInt(u64), "1e300"));
    const Settings = struct { startup_extra: u64 = 0 };
    const large = try withCache(Settings{}, 100);
    try std.testing.expectEqual(@as(u64, 100) * GiB, large.startup_extra);
    const Device = struct {
        pub fn maxWorkingSet(_: @This()) u64 {
            return 128 * GiB;
        }
    };
    try std.testing.expectError(error.FlashNextMemoryShortfall, admitUnder(Device{}, 100 * GiB, trained, 128 * GiB, null, large.startup_extra));
    try std.testing.expectError(error.BadCacheBudget, withCache(Settings{}, std.math.nan(f64)));
    try std.testing.expect(growingFits(0, 8 * GiB, 10 * GiB));
    try std.testing.expect(!growingFits(0, 8 * GiB + 1, 10 * GiB));
    try std.testing.expect(!growingFits(std.math.maxInt(u64), 1, std.math.maxInt(u64)));
}

test "Flash Next memory: transformed and pack peaks refuse beyond the known-buffer floor" {
    const Device = struct {
        cap: u64,
        pub fn maxWorkingSet(d: @This()) u64 {
            return d.cap;
        }
    };
    const fixed = try minimum(0, trained);
    const inventory: weights_plan.Inventory = .{ .raw = 100 * GiB, .source = GiB, .decode = GiB, .prompt = GiB };
    try std.testing.expectError(error.FlashNextMemoryShortfall, admitInventory(Device{ .cap = fixed + 1 }, 0, trained, std.math.maxInt(u64), "1e300", 0, inventory));
    const b = try budget(100 * GiB, trained, inventory, 0);
    try std.testing.expect(b.steady > 100 * GiB + fixed);
    const convert = try budget(0, trained, .{ .source = 64 * GiB, .decode = GiB, .prompt = GiB }, 0);
    try std.testing.expect(convert.building > convert.steady);
    try std.testing.expectError(error.Overflow, budget(0, trained, .{ .decode = std.math.maxInt(u64) }, 0));
}

test {
    _ = weights_plan;
}

test "Flash Next memory: validated no-build routes admit a steady fit without charging a skipped conversion" {
    const Device = struct {
        cap: u64,
        pub fn maxWorkingSet(d: @This()) u64 {
            return d.cap;
        }
    };
    const inventory: weights_plan.Inventory = .{ .raw = GiB, .source = 64 * GiB, .decode = GiB, .prompt = GiB };
    const p = try budgetLive(1000 * GiB, trained, inventory, 0, 0);
    const same = try budgetLive(GiB, trained, inventory, 0, 0);
    // Padding and unused source-file bytes change neither the resident payload nor the build's touched source.
    try std.testing.expectEqual(p, same);
    const cap = p.steady + GiB;
    try std.testing.expect(p.building > cap);
    try std.testing.expectEqual(cap, try admitPhase(Device{ .cap = cap }, 1000 * GiB, trained, 256 * GiB, "1e300", 0, inventory, 0, false));
    try std.testing.expectError(error.FlashNextMemoryShortfall, admitPhase(Device{ .cap = cap }, GiB, trained, 256 * GiB, "1e300", 0, inventory, 0, true));
}

test "Flash Next memory: cold construction has no future prompt, session or vision buffers live beside it" {
    const inventory: weights_plan.Inventory = .{ .source = GiB, .decode = 2 * GiB, .prompt = 3 * GiB };
    const ordinary = try budgetLive(10 * GiB, trained, inventory, 0, 512);
    const requested = try budgetLive(10 * GiB, trained, inventory, 8 * GiB, 512);
    try std.testing.expectEqual(ordinary.building, requested.building);
    try std.testing.expectEqual(ordinary.steady + 8 * GiB, requested.steady);
    try std.testing.expectEqual(@as(u64, 10 * GiB + 512), requested.building);
}
