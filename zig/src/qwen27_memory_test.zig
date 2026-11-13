//! Production memory choices at exact fit boundaries without a device or model load.
const std = @import("std");
const memory = @import("families/qwen27/prompt_memory.zig");
const rows = @import("families/qwen27/prompt_rows.zig");
const Config = @import("families/qwen27/config.zig").Config;

fn config() Config {
    var c = Config{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .heads = 24, .kv_heads = 4, .head_dim = 256, .k_heads = 16, .v_heads = 48, .dk = 128, .dv = 128, .conv_kernel = 4, .eps = 1e-6, .rope_dims = 64, .rope_theta = 1e7 };
    for (0..c.layers) |i| c.kinds[i] = if ((i + 1) % 4 == 0) .attention else .linear;
    return c;
}

fn fixture() memory.Fit {
    return .{ .config = config(), .context = 131072 + 128, .slots = 2, .shared = true, .current_rows = 512, .left = 0 };
}

test "qwen27 memory: a wide frame preserves all configured slots and the exact working-set boundary" {
    var f = fixture();
    const wide = try f.cost(2048, f.slots);
    const narrow = try f.cost(512, f.slots);
    try std.testing.expect(wide > narrow + (12 << 30));
    f.left = wide;
    try std.testing.expectEqual(@as(u32, 2048), try f.choose(rows.requested(17, false, null), false));
    f.left -= 1;
    try std.testing.expectEqual(@as(u32, 512), try f.choose(rows.requested(17, false, null), false));
    f.left = narrow;
    try std.testing.expectEqual(@as(u32, 512), try f.choose(rows.requested(17, false, null), false));
    try std.testing.expectEqual(@as(u32, 2), try f.streams(512, true));
    try std.testing.expectError(error.TooManyStreams, f.streams(2048, true));
    f.left = try f.cost(512, 1);
    try std.testing.expectEqual(@as(u32, 1), try f.streams(512, false));
    f.left -= 1;
    try std.testing.expectError(error.TooManyStreams, f.streams(512, false));
}

test "qwen27 memory: available bytes include free and inactive pages with checked arithmetic" {
    const policy = @import("sys").memory_policy;
    try std.testing.expectEqual(@as(?usize, 12 * 4096), policy.availableBytes(5, 7, 4096));
    try std.testing.expectEqual(@as(?usize, 7 * 16384), policy.availableBytes(0, 7, 16384));
    try std.testing.expectEqual(@as(?usize, 5 * 16384), policy.availableBytes(5, 0, 16384));
    try std.testing.expectEqual(@as(?usize, 0), policy.availableBytes(0, 0, 16384));
    try std.testing.expectEqual(@as(?usize, null), policy.availableBytes(1, 1, 0));
    try std.testing.expectEqual(@as(?usize, null), policy.availableBytes(std.math.maxInt(usize), 1, 4096));
    try std.testing.expectEqual(@as(?usize, null), policy.availableBytes(std.math.maxInt(usize) / 2, 1, 2));
}

test "qwen27 memory: the live available-memory reader is available only on macOS" {
    const available = @import("sys").memory_policy.available();
    if (@import("builtin").os.tag != .macos) {
        try std.testing.expectEqual(@as(?usize, null), available);
        return;
    }
    const bytes = available orelse return error.AvailableMemoryReadingFailed;
    try std.testing.expectEqual(@as(usize, 0), bytes % std.heap.pageSize());
}

test "qwen27 memory: live startup readings warn only past available memory or under pressure" {
    var f = fixture();
    f.left = try f.cost(512, f.slots);
    const page_size = 16384;
    const pages = try std.math.divCeil(usize, f.left, page_size);
    const reclaimed = @import("sys").memory_policy.availableBytes(1, pages - 1, page_size).?;
    try std.testing.expect(f.left > page_size);
    try std.testing.expectEqual(@as(?memory.Warning, null), try f.warning(512, f.slots, 128, reclaimed, 1));
    try std.testing.expectEqual(@as(?memory.Warning, null), try f.warning(512, f.slots, 128, f.left, 1));
    try std.testing.expectEqual(@as(?memory.Warning, null), try f.warning(512, f.slots, 128, null, null));
    const short = (try f.warning(512, f.slots, 128, f.left - 1, 1)).?;
    try std.testing.expectEqual(f.left, short.reserve);
    try std.testing.expectEqual(@as(u32, 1), short.streams.?);
    try std.testing.expect(short.context.? < f.context - 128);
    for ([_]u32{ 2, 4 }) |pressure| {
        const pressured = (try f.warning(512, f.slots, 128, f.left, pressure)).?;
        try std.testing.expectEqual(@as(?u32, null), pressured.streams);
        try std.testing.expectEqual(@as(?u32, null), pressured.context);
    }
    const unknown = (try f.warning(512, f.slots, 128, null, 4)).?;
    try std.testing.expectEqual(@as(?u32, null), unknown.context);
    try std.testing.expectEqual(@as(?u32, null), unknown.streams);
}

test "qwen27 memory: pressure-only warnings explain the pressure without repeating current settings" {
    var f = fixture();
    f.left = try f.cost(512, f.slots);
    for ([_]u32{ 2, 4 }) |pressure| {
        for ([_]usize{ f.left, f.left + (1 << 30) }) |available| {
            const warning = (try f.warning(512, f.slots, 128, available, pressure)).?;
            try std.testing.expectEqual(@as(?u32, null), warning.streams);
            try std.testing.expectEqual(@as(?u32, null), warning.context);
            var buffer: [1024]u8 = undefined;
            const message = try std.fmt.bufPrint(&buffer, "{f}", .{warning});
            try std.testing.expect(std.mem.indexOf(u8, message, "--parallel") == null);
            try std.testing.expect(std.mem.indexOf(u8, message, "--context") == null);
            try std.testing.expect(std.mem.indexOf(u8, message, "settings fit available memory") != null);
            try std.testing.expect(std.mem.indexOf(u8, message, "about memory pressure") != null);
        }
    }
}

test "qwen27 memory: startup recommendations fit the same drafted stream and scratch estimator" {
    for ([_]bool{ false, true }) |shared| {
        var f = fixture();
        f.shared = shared;
        f.slots = 4;
        f.ring = 2048 * 5 * 5120 * 2;
        f.planes = 7;
        f.left = try f.cost(512, f.slots);
        const free = try f.cost(512, 2);
        const warning = (try f.warning(512, f.slots, 128, free, 1)).?;
        try std.testing.expectEqual(@as(u32, 2), warning.streams.?);
        try std.testing.expect(warning.context.? < f.context - 128);
        var smaller = f;
        smaller.context = warning.context.? + 128;
        try std.testing.expect(try smaller.cost(512, f.slots) <= free);
        smaller.context += 1;
        try std.testing.expect(try smaller.cost(512, f.slots) > free);
        const empty = (try f.warning(512, f.slots, 128, 0, null)).?;
        try std.testing.expectEqual(@as(?u32, 0), empty.streams);
        try std.testing.expectEqual(@as(?u32, 0), empty.context);
        try std.testing.expectEqual(@as(u32, 4), try f.streams(512, true));
    }
}

test "qwen27 memory: startup warning explains known and unavailable readings" {
    const cases = [_]struct { warning: memory.Warning, text: []const u8 }{
        .{ .warning = .{ .reserve = 8 << 30, .available = 4 << 30, .pressure = 1, .streams = 1, .context = 8192 }, .text = "4.0 GiB available" },
        .{ .warning = .{ .reserve = 8 << 30, .available = null, .pressure = 4 }, .text = "available memory is unknown" },
        .{ .warning = .{ .reserve = 8 << 30, .available = 0, .pressure = null, .streams = 0, .context = 0 }, .text = "make memory available before starting" },
    };
    for (cases) |case| {
        var buffer: [1024]u8 = undefined;
        const message = try std.fmt.bufPrint(&buffer, "{f}", .{case.warning});
        try std.testing.expect(std.mem.indexOf(u8, message, "8.0 GiB") != null);
        try std.testing.expect(std.mem.indexOf(u8, message, case.text) != null);
    }
}

test "qwen27 memory: overrides keep their width and still refuse an unaffordable startup" {
    var f = fixture();
    f.left = try f.cost(512, 2);
    for ([_][]const u8{ "16", "512", "1024", "2048" }, [_]u32{ 16, 512, 1024, 2048 }) |text, width| {
        try std.testing.expectEqual(width, try f.choose(rows.requested(17, false, text), rows.override(text) != null));
    }
    try std.testing.expectError(error.TooManyStreams, f.streams(2048, true));
    try std.testing.expectEqual(@as(u32, 512), try f.choose(rows.requested(17, false, "invalid"), rows.override("invalid") != null));
}

test "qwen27 memory: startup counts shared scratch once and each slot's state, KV, ring and rollback bank" {
    var f = fixture();
    const narrow = try memory.runnerBytes(f.config, f.context, 1, 512, true, 0, 5);
    const old = @as(usize, f.context) * 16 * 2 * 1024 * 2 + (256 << 20);
    try std.testing.expect(narrow > old + (4 << 30));
    const wide = try memory.runnerBytes(f.config, f.context, 1, 2048, true, 0, 5);
    try std.testing.expect(wide > narrow + (12 << 30));
    try std.testing.expectEqual(wide - narrow + memory.frameBytes(f.config, 2048) - memory.frameBytes(f.config, 512), (try f.cost(2048, 1)) - (try f.cost(512, 1)));
    const ring = 2048 * 5 * 5120 * 2;
    try std.testing.expectEqual(@as(usize, 2 * ring), (try memory.runnerBytes(f.config, f.context, 2, 512, true, ring, 5)) - (try memory.runnerBytes(f.config, f.context, 2, 512, true, 0, 5)));
    f.context = 32768;
    try std.testing.expect(try f.cost(2048, 2) < wide);
    f.shared = false;
    try std.testing.expect(try f.cost(512, 2) > old);
}

test "qwen27 memory: the development working-set cap cannot raise real capacity" {
    const real: usize = 192 << 30;
    try std.testing.expectEqual(real, try memory.workingSet(real, null));
    try std.testing.expectEqual(real, try memory.workingSet(real, "274877906944"));
    try std.testing.expectEqual(@as(usize, 40 << 30), try memory.workingSet(real, "42949672960"));
    for ([_][]const u8{ "", "0", "-1", "1.5", "18446744073709551616" }) |text| try std.testing.expectError(error.BadWorkingSetCap, memory.workingSet(real, text));
}

test "qwen27 memory: working sets account for weights and the unchanged 8 GiB margin" {
    var f = fixture();
    const weights: usize = 16 << 30;
    const wide = weights + (8 << 30) + try f.cost(2048, 2);
    const narrow = weights + (8 << 30) + try f.cost(512, 2);
    for ([_]usize{ wide, wide - 1, narrow }, [_]u32{ 2048, 512, 512 }) |working, want| {
        f.left = try memory.room(working, weights, null);
        try std.testing.expectEqual(want, try f.choose(2048, false));
    }
    try std.testing.expectEqual(@as(usize, 0), try memory.room(20 << 30, weights, null));
    try std.testing.expectEqual(@as(usize, 16 << 30), try memory.room(192 << 30, weights, "42949672960"));
    try std.testing.expectEqual(@as(usize, 0), try memory.room(192 << 30, 200 << 30, null));
}

test "qwen27 memory: a wired-limit raise fits one stream only above the floor and rounds up to a whole MiB" {
    const mib: usize = 1 << 20;
    const used: usize = 10 << 30;
    const need: usize = 5 << 30;
    const total = used + need + (8 << 30);
    const least = total + (4 << 30); // the least RAM whose suggested limit leaves exactly the floor
    try std.testing.expectEqual(@as(?usize, total / mib), memory.wiredRaise(used, need, least));
    try std.testing.expectEqual(@as(?usize, null), memory.wiredRaise(used, need, least - 1));
    // One stream fits this RAM, but the limit it needs would leave less than the floor.
    try std.testing.expectEqual(@as(?usize, null), memory.wiredRaise(used, need, total + (1 << 30)));
    try std.testing.expectEqual(@as(?usize, total / mib + 1), memory.wiredRaise(used, need + 1, least + mib));
}

test "qwen27 memory: the issue's 32 GB Mac needs a raise under 29696 MB, and 16 GB of RAM cannot" {
    const real: usize = 22_870_000_000; // Metal's working set on a 32 GB Mac
    const used: usize = 11_381_000_000; // weights and buffers
    const one: usize = 4_617_000_000; // one stream's scratch at this context
    try std.testing.expect(one > try memory.room(real, used, null)); // 2.7 GiB remains: the refusal
    const mb = memory.wiredRaise(used, one, 32 << 30) orelse return error.NoRaise;
    try std.testing.expect(mb <= 29696); // the issue's fix
    try std.testing.expect(mb * (1 << 20) >= used + one + (8 << 30));
    try std.testing.expectEqual(@as(?usize, null), memory.wiredRaise(used, one, 16 << 30));
}

test "qwen27 memory: a refusal advises lowering parallel, the raise, or lowering context when RAM cannot" {
    const real: usize = 22_870_000_000;
    const used: usize = 11_381_000_000;
    const one: usize = 4_617_000_000;
    const cases = [_]struct { advice: memory.Advice, text: []const u8 }{
        .{ .advice = memory.advice(real, used, null, 1, 32 << 30), .text = ": lower --parallel or --context" },
        .{ .advice = memory.advice(real, used, null, one, 32 << 30), .text = ": Metal lets this Mac use 21.3 GiB; sudo sysctl iogpu.wired_limit_mb=23449 raises that limit enough for one stream at this context until reboot (or lower --context)" },
        .{ .advice = memory.advice(real, used, null, one, 16 << 30), .text = ": no wired-limit raise fits one stream and leaves 4 GiB of this Mac's 16.0 GiB of RAM to the system: lower --context" },
        .{ .advice = memory.advice(real, used, null, one, 26 << 30), .text = ": no wired-limit raise fits one stream and leaves 4 GiB of this Mac's 26.0 GiB of RAM to the system: lower --context" },
        .{ .advice = memory.advice(real, used, "10737418240", one, 32 << 30), .text = ": lower --parallel or --context" },
        .{ .advice = memory.advice(real, used, "1.5", one, 32 << 30), .text = ": lower --parallel or --context" },
        .{ .advice = memory.advice(real, used, "274877906944", one, 32 << 30), .text = ": Metal lets this Mac use 21.3 GiB; sudo sysctl iogpu.wired_limit_mb=23449 raises that limit enough for one stream at this context until reboot (or lower --context)" },
        .{ .advice = memory.advice(real, used, null, one, null), .text = ": no wired-limit raise is known to fit one stream: lower --context" },
    };
    for (cases) |c| {
        var buf: [512]u8 = undefined;
        try std.testing.expectEqualStrings(c.text, try std.fmt.bufPrint(&buf, "{f}", .{c.advice}));
    }
}
