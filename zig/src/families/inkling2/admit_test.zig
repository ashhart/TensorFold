//! Inkling 2 Mini's admission on the host: Mini's looped shape, four RAM classes, and the pressure level as a guard.
const std = @import("std");
const config = @import("config.zig");
const config_test = @import("config_test.zig");
const layout = @import("state_layout.zig");
const admit = @import("admit.zig");
const mp = @import("memory_pressure");

const GiB: u64 = 1 << 30;
const window_rows = 16;
const max_rows = 512;
const fixed_bytes = 189 * GiB / 10; // weights, dense layers and scratch, as Mini's startup line reports them

fn mini() !config.Config {
    var why: config.Why = .{};
    return config.parse(std.testing.allocator, config_test.mini, &why);
}

/// A Mac's load limit: 70% of RAM, at most the GPU working set (two thirds of RAM to 24 GiB, else three quarters).
fn limitOf(ram_gib: u64) u64 {
    const ram = ram_gib * GiB;
    const working_set = if (ram_gib <= 24) ram / 3 * 2 else ram / 4 * 3;
    return @min(ram, @min(working_set, ram / 100 * 70));
}

fn demandAt(c: *const config.Config, context: u32) admit.Demand {
    return .{
        .fixed = fixed_bytes,
        .sampling = @as(u64, c.text.unpadded_vocab) * 64 + 128,
        .stream = layout.laneBytes(c, .{ .capacity = context, .rows = max_rows }, false, window_rows),
    };
}

/// What a fresh process asks for `asked` streams at `context` on a `ram_gib` Mac, nothing else held.
fn serves(c: *const config.Config, ram_gib: u64, context: u32, asked: u32, named: bool) admit.Verdict {
    const d = demandAt(c, context);
    const limit = limitOf(ram_gib);
    return admit.verdict(admit.left(limit, limit -| d.fixed, d), d, asked, named);
}

test "Mini's looped shape: 24 physical layers run as 48 visits, 8 global, and a stream's bytes grow with the window" {
    const c = try mini();
    try std.testing.expectEqual(@as(u32, 24), c.text.layers);
    try std.testing.expectEqual(@as(u32, 48), c.text.executionLayers());
    var global: u32 = 0;
    for (0..c.text.executionLayers()) |l| global += @intFromBool(config.isGlobal(c.text.physicalLayer(@intCast(l))));
    try std.testing.expectEqual(@as(u32, 8), global);
    const short = demandAt(&c, 8192).stream;
    const long = demandAt(&c, 131072).stream;
    // a global visit's keys and values are one slot a position; a sliding visit's ring does not grow
    const per_slot = 2 * layout.kvWidth(&c.text, 5) * 2;
    try std.testing.expectEqual(@as(u64, global) * (131072 - 8192) * per_slot, long - short);
}

test "admitted streams by RAM class at an 8192-token window, --parallel 4 named: a Mac that holds the weights serves all four" {
    const c = try mini();
    const classes = [_]struct { ram: u64, want: admit.Verdict }{
        .{ .ram = 16, .want = .none },
        .{ .ram = 24, .want = .none },
        .{ .ram = 36, .want = .{ .serve = 4 } },
        .{ .ram = 48, .want = .{ .serve = 4 } },
    };
    for (classes) |k| try std.testing.expectEqualDeep(k.want, serves(&c, k.ram, 8192, 4, true));
}

test "admitted streams by RAM class when --parallel is not named: as many as fit, up to the count asked" {
    const c = try mini();
    const classes = [_]struct { ram: u64, context: u32, want: admit.Verdict }{
        .{ .ram = 16, .context = 8192, .want = .none },
        .{ .ram = 24, .context = 8192, .want = .none },
        .{ .ram = 36, .context = 8192, .want = .{ .serve = 8 } },
        .{ .ram = 48, .context = 8192, .want = .{ .serve = 8 } },
        .{ .ram = 36, .context = 131072, .want = .{ .serve = 1 } },
        .{ .ram = 48, .context = 131072, .want = .{ .serve = 3 } },
    };
    for (classes) |k| try std.testing.expectEqualDeep(k.want, serves(&c, k.ram, k.context, 8, false));
    // a named count that does not fit is refused with the count that does
    try std.testing.expectEqualDeep(admit.Verdict{ .short = 1 }, serves(&c, 36, 131072, 4, true));
    try std.testing.expectEqualDeep(admit.Verdict{ .short = 3 }, serves(&c, 48, 131072, 4, true));
}

test "the count is the most streams whose demand fits the load limit" {
    const c = try mini();
    for ([_]u64{ 36, 48 }) |ram| for ([_]u32{ 4096, 8192, 32768, 131072 }) |context| {
        const d = demandAt(&c, context);
        const n = switch (serves(&c, ram, context, 64, false)) {
            .serve => |n| n,
            else => return error.NoneFit,
        };
        try std.testing.expect(d.of(n) <= limitOf(ram));
        try std.testing.expect(n == 64 or d.of(n + 1) > limitOf(ram));
    };
}

test "what the process already holds past its load comes out of the room, whichever count is smaller" {
    const c = try mini();
    const d = demandAt(&c, 8192);
    const limit = limitOf(36);
    const clean = admit.left(limit, limit - d.fixed, d);
    try std.testing.expectEqual(limit - d.fixed - d.sampling, clean);
    try std.testing.expectEqual(clean - 2 * GiB, admit.left(limit, limit - d.fixed - 2 * GiB, d)); // the device holds more than the load
    try std.testing.expectEqual(clean - 2 * GiB, admit.left(limit - 2 * GiB, limit - d.fixed, d)); // the process held more before it
    try std.testing.expectEqual(@as(u64, 0), admit.left(d.fixed - 1, limit, d)); // the fixed part alone is past the limit
}

test "a refusal says what the weights leave, what a stream takes, and how many fit" {
    const c = try mini();
    const a = std.testing.allocator;
    const d = demandAt(&c, 131072);
    const limit = limitOf(36);
    const left = admit.left(limit, limit - d.fixed, d);
    const short = try admit.refusal(a, admit.verdict(left, d, 4, true), d, left, 4, 131072);
    defer a.free(short);
    try std.testing.expect(std.mem.indexOf(u8, short, "--parallel 4 needs") != null and std.mem.indexOf(u8, short, ", 1 fit:") != null);
    const none = try admit.refusal(a, admit.verdict(left / 100, d, 4, true), d, left / 100, 4, 131072);
    defer a.free(none);
    try std.testing.expect(std.mem.indexOf(u8, none, "none fits: lower --context") != null);
}

fn noReading(_: ?*anyopaque) ?mp.Level {
    return null;
}

/// The governor as the engine sets it once the streams are admitted (`fits`), or as every other family runs it.
fn governed(fits: bool, raw: mp.Level) mp.Governor {
    var g: mp.Governor = .{ .source = .{ .read = noReading }, .calm_ns = 0 };
    if (fits) g.streamsFit();
    _ = g.observe(raw, 0);
    return g;
}

test "pressure guards admitted streams only at critical: warning leaves streams that fit running together" {
    for ([_]mp.Level{ .normal, .warning }) |level| try std.testing.expect(!governed(true, level).holds());
    try std.testing.expect(governed(true, .critical).holds());
    // streams not known to fit keep the shared rule: any level above normal holds admission to one stream
    try std.testing.expect(!governed(false, .normal).holds());
    try std.testing.expect(governed(false, .warning).holds());
    try std.testing.expect(governed(false, .critical).holds());
}
