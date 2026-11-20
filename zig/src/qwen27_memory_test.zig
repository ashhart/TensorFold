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

test "qwen27 memory: the margin is an eighth of the working set, 1 GiB on the smallest Macs and 8 GiB from 64 GiB" {
    try std.testing.expectEqual(@as(usize, 1 << 30), memory.margin(5_726_623_744));
    try std.testing.expectEqual(@as(usize, 11_453_251_584 / 8), memory.margin(11_453_251_584));
    try std.testing.expectEqual(@as(usize, 8 << 30), memory.margin(64 << 30));
    try std.testing.expectEqual(@as(usize, 8 << 30), memory.margin(192 << 30));
}

test "qwen27 memory: working sets account for weights and the margin" {
    var f = fixture();
    const weights: usize = 48 << 30;
    const wide = weights + (8 << 30) + try f.cost(2048, 2);
    const narrow = weights + (8 << 30) + try f.cost(512, 2);
    for ([_]usize{ wide, wide - 1, narrow }, [_]u32{ 2048, 512, 512 }) |working, want| {
        f.left = try memory.room(working, weights, null);
        try std.testing.expectEqual(want, try f.choose(2048, false));
    }
    try std.testing.expectEqual(@as(usize, (20 << 30) - (16 << 30) - (20 << 30) / 8), try memory.room(20 << 30, 16 << 30, null));
    try std.testing.expectEqual(@as(usize, (40 << 30) - (16 << 30) - (5 << 30)), try memory.room(192 << 30, 16 << 30, "42949672960"));
    try std.testing.expectEqual(@as(usize, 0), try memory.room(192 << 30, 200 << 30, null));
}

test "qwen27 memory: a frame that cannot fit 512 rows steps down to the narrowest that fits, never below 128" {
    var f = fixture();
    f.context = 4096;
    f.slots = 1;
    f.left = try f.cost(256, 1);
    try std.testing.expectEqual(@as(u32, 256), try f.choose(2048, false));
    try std.testing.expectEqual(@as(u32, 256), try f.choose(512, false));
    f.left -= 1;
    try std.testing.expectEqual(@as(u32, 128), try f.choose(512, false));
    f.left = 0;
    try std.testing.expectEqual(@as(u32, 128), try f.choose(512, false));
    try std.testing.expectEqual(@as(u32, 64), try f.choose(64, false));
    try std.testing.expectEqual(@as(u32, 1024), try f.choose(1024, true));
}

test "qwen27 memory: a wired-limit raise fits one stream only above the floor and rounds up to a whole MiB" {
    const mib: usize = 1 << 20;
    const used: usize = 60 << 30;
    const need: usize = 5 << 30;
    const total = used + need + (8 << 30);
    const least = total + (4 << 30); // the least RAM whose suggested limit leaves exactly the floor
    try std.testing.expectEqual(@as(?usize, total / mib), memory.wiredRaise(used, need, least));
    try std.testing.expectEqual(@as(?usize, null), memory.wiredRaise(used, need, least - 1));
    try std.testing.expectEqual(@as(?usize, null), memory.wiredRaise(used, need, total + (1 << 30)));
    try std.testing.expectEqual(@as(?usize, total / mib + 1), memory.wiredRaise(used, need + 1, least + mib));
}

test "qwen27 memory: the issue's 32 GB Mac now fits one stream, and a stream twice that size needs a raise" {
    const real: usize = 22_870_000_000; // Metal's working set on a 32 GB Mac
    const used: usize = 11_381_000_000; // weights and buffers
    const one: usize = 4_617_000_000; // one stream's scratch at this context
    try std.testing.expect(one <= try memory.room(real, used, null));
    const big: usize = 10_000_000_000;
    try std.testing.expect(big > try memory.room(real, used, null));
    const mb = memory.wiredRaise(used, big, 32 << 30) orelse return error.NoRaise;
    try std.testing.expectEqual(@as(usize, 23975), mb);
    try std.testing.expectEqual(@as(?usize, null), memory.wiredRaise(used, big, 16 << 30));
}

test "qwen27 memory: a refusal advises lowering parallel, the raise, or lowering context when RAM cannot" {
    const real: usize = 22_870_000_000;
    const used: usize = 11_381_000_000;
    const big: usize = 10_000_000_000;
    const cases = [_]struct { advice: memory.Advice, text: []const u8 }{
        .{ .advice = memory.advice(real, used, null, 4_617_000_000, 32 << 30), .text = ": lower --parallel or --context" },
        .{ .advice = memory.advice(real, used, null, big, 32 << 30), .text = ": Metal lets this Mac use 21.3 GiB; sudo sysctl iogpu.wired_limit_mb=23975 raises that limit enough for one stream at this context until reboot (or lower --context)" },
        .{ .advice = memory.advice(real, used, null, big, 16 << 30), .text = ": no wired-limit raise fits one stream and leaves 4 GiB of this Mac's 16.0 GiB of RAM to the system: lower --context" },
        .{ .advice = memory.advice(real, used, null, big, 26 << 30), .text = ": no wired-limit raise fits one stream and leaves 4 GiB of this Mac's 26.0 GiB of RAM to the system: lower --context" },
        .{ .advice = memory.advice(real, used, "10737418240", big, 32 << 30), .text = ": lower --parallel or --context" },
        .{ .advice = memory.advice(real, used, "1.5", big, 32 << 30), .text = ": lower --parallel or --context" },
        .{ .advice = memory.advice(real, used, "274877906944", big, 32 << 30), .text = ": Metal lets this Mac use 21.3 GiB; sudo sysctl iogpu.wired_limit_mb=23975 raises that limit enough for one stream at this context until reboot (or lower --context)" },
        .{ .advice = memory.advice(real, used, null, big, null), .text = ": no wired-limit raise is known to fit one stream: lower --context" },
    };
    for (cases) |c| {
        var buf: [512]u8 = undefined;
        try std.testing.expectEqualStrings(c.text, try std.fmt.bufPrint(&buf, "{f}", .{c.advice}));
    }
}
