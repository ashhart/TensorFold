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
    try std.testing.expectEqual(@as(u32, 2048), try f.choose(rows.requested(17, null), false));
    f.left -= 1;
    try std.testing.expectEqual(@as(u32, 512), try f.choose(rows.requested(17, null), false));
    f.left = narrow;
    try std.testing.expectEqual(@as(u32, 512), try f.choose(rows.requested(17, null), false));
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
        try std.testing.expectEqual(width, try f.choose(rows.requested(17, text), rows.override(text) != null));
    }
    try std.testing.expectError(error.TooManyStreams, f.streams(2048, true));
    try std.testing.expectEqual(@as(u32, 512), try f.choose(rows.requested(17, "invalid"), rows.override("invalid") != null));
}

test "qwen27 memory: startup counts shared scratch once and each slot's state, KV, ring and rollback bank" {
    var f = fixture();
    const narrow = try memory.runnerBytes(f.config, f.context, 1, 512, true, 0);
    const old = @as(usize, f.context) * 16 * 2 * 1024 * 2 + (256 << 20);
    try std.testing.expect(narrow > old + (4 << 30));
    const wide = try memory.runnerBytes(f.config, f.context, 1, 2048, true, 0);
    try std.testing.expect(wide > narrow + (12 << 30));
    try std.testing.expectEqual(wide - narrow + memory.frameBytes(f.config, 2048) - memory.frameBytes(f.config, 512), (try f.cost(2048, 1)) - (try f.cost(512, 1)));
    const ring = 2048 * 5 * 5120 * 2;
    try std.testing.expectEqual(@as(usize, 2 * ring), (try memory.runnerBytes(f.config, f.context, 2, 512, true, ring)) - (try memory.runnerBytes(f.config, f.context, 2, 512, true, 0)));
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

const kv_grow = @import("families/qwen27/kv_grow.zig");

test "qwen27 memory: grow-as-used K/V is off unless TF_Q27_KV_GROW names a 16-row step" {
    try std.testing.expectEqual(@as(?u32, null), try kv_grow.step(null));
    try std.testing.expectEqual(@as(?u32, null), try kv_grow.step("0"));
    try std.testing.expectEqual(@as(?u32, 4096), try kv_grow.step("4096"));
    try std.testing.expectEqual(@as(?u32, 16), try kv_grow.step("16"));
    for ([_][]const u8{ "", "-1", "15", "4097", "x", "99999999999" }) |text| try std.testing.expectError(error.BadKvGrowStep, kv_grow.step(text));
}

test "qwen27 memory: a slot starts at one step and grows to the need, a quarter more, or the context" {
    // first rows: one step, never past the context
    try std.testing.expectEqual(@as(u32, 4096), kv_grow.first(4096, 32784));
    try std.testing.expectEqual(@as(u32, 2000), kv_grow.first(4096, 2000));
    // enough already: unchanged
    try std.testing.expectEqual(@as(u32, 4096), kv_grow.rows(4096, 4096, 4096, 32784));
    try std.testing.expectEqual(@as(u32, 4096), kv_grow.rows(4096, 10, 4096, 32784));
    // the need rounded up to the step
    try std.testing.expectEqual(@as(u32, 8192), kv_grow.rows(4096, 4097, 4096, 32784));
    try std.testing.expectEqual(@as(u32, 32768), kv_grow.rows(4096, 30001, 4096, 32784));
    // a quarter more than held when that is past the rounded need
    try std.testing.expectEqual(@as(u32, 20480), kv_grow.rows(16384, 16385, 4096, 32784));
    try std.testing.expectEqual(@as(u32, 20480), kv_grow.rows(16384, 16385, 16, 32784));
    // never past the context
    try std.testing.expectEqual(@as(u32, 32784), kv_grow.rows(32768, 32780, 4096, 32784));
    try std.testing.expectEqual(@as(u32, 32784), kv_grow.rows(28672, 28673, 4096, 32784));
}

test "qwen27 memory: growing moves every K/V head to its place in the wider buffer" {
    // 4 heads of dim 256 in bf16: head h starts at h * rows * 256 * 2 bytes
    var moves: [4]kv_grow.Move = undefined;
    const got = kv_grow.moves(&moves, 4, 256, 4096, 8192);
    try std.testing.expectEqual(@as(usize, 4), got.len);
    for (got, 0..) |m, h| {
        try std.testing.expectEqual(h * 4096 * 256 * 2, m.from);
        try std.testing.expectEqual(h * 8192 * 256 * 2, m.to);
        try std.testing.expectEqual(@as(usize, 4096 * 256 * 2), m.len);
    }
    // the old buffer's bytes are exactly covered, the new head regions never overlap
    try std.testing.expectEqual(@as(usize, 4 * 4096 * 256 * 2), got[3].from + got[3].len);
    try std.testing.expect(got[2].to + got[2].len <= got[3].to);
    // a slot's bytes at a row count, for the residency set and the memory receipts
    try std.testing.expectEqual(@as(usize, 8 * 2 * 4096 * 1024 * 2), kv_grow.slotBytes(8, 1024, 4096));
}
