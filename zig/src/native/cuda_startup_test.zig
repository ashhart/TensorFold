//! The real chunk reader and loaded metadata drive the startup text without opening an engine or a device.
const std = @import("std");
const flash = @import("flashnext");
const nemo = @import("nemotron").native;
const startup = @import("cuda_startup.zig");
const env = @import("sys").env;
const variable = "TF_FLASHNEXT_CHUNK";

const Saved = struct {
    value: ?[:0]u8,
    fn take() !Saved {
        return .{ .value = if (std.c.getenv(variable)) |v| try std.testing.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null };
    }
    fn restore(s: Saved) void {
        if (s.value) |v| {
            _ = env.setenv(variable, v, 1);
            std.testing.allocator.free(v);
        } else _ = env.unsetenv(variable);
    }
};

fn set(value: ?[:0]const u8) !void {
    const rc = if (value) |v| env.setenv(variable, v, 1) else env.unsetenv(variable);
    try std.testing.expectEqual(@as(c_int, 0), rc);
}

fn line(comptime F: type, loaded: F.Loaded) ![]u8 {
    return std.fmt.allocPrint(std.testing.allocator, startup.format, .{
        @as(u32, 121),      @as(u32, 0),  "GPU",       "",                      @as(f64, 12.5), "", @as(u32, 1),        "",
        "1.0 GiB a stream", @as(f64, 20), @as(f64, 2), startup.rows(F, loaded), "",             "", "own glue kernels", "",
    });
}

fn expected(comptime rows: u32) []const u8 {
    return std.fmt.comptimePrint("CUDA sm_121 device 0 (GPU): model 12.50 GiB; 1 stream at once, 1.0 GiB a stream, of 20.0 GiB left after a 2.0 GiB reserve; prompts in passes of up to {d} rows; own glue kernels", .{rows});
}

fn check(value: ?[:0]const u8, comptime rows: u32) !void {
    const saved = try Saved.take();
    defer saved.restore();
    try set(value);
    var engine: flash.prompt.Engine = undefined;
    engine.chunk = try flash.prompt.chunkRows();
    var loaded: flash.native.Loaded = undefined;
    loaded.prompt_rows = @intCast(engine.chunk);
    loaded.rows = 2;
    const text = try line(flash.native, loaded);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings(expected(rows), text);
}

test "CUDA startup: unset Flash Next chunk reports the default stored pass" {
    try check(null, 3072);
}

test "CUDA startup: Flash Next chunk 4096 reports the stored pass" {
    try check("4096", 4096);
}

test "CUDA startup: Flash Next chunk 16384 reports the stored pass" {
    try check("16384", 16384);
}

test "CUDA startup: the loaded value survives a later environment change" {
    const saved = try Saved.take();
    defer saved.restore();
    try set("4096");
    var engine: flash.prompt.Engine = undefined;
    engine.chunk = try flash.prompt.chunkRows();
    var loaded: flash.native.Loaded = undefined;
    loaded.prompt_rows = @intCast(engine.chunk);
    loaded.rows = 2;
    try set("16384");
    const text = try line(flash.native, loaded);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings(expected(4096), text);
}

test "CUDA startup: Nemotron's complete line keeps its existing bytes" {
    const saved = try Saved.take();
    defer saved.restore();
    try set("16384");
    var loaded: nemo.Loaded = undefined;
    loaded.rows = 2;
    const text = try line(nemo, loaded);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings("CUDA sm_121 device 0 (GPU): model 12.50 GiB; 1 stream at once, 1.0 GiB a stream, of 20.0 GiB left after a 2.0 GiB reserve; prompts in passes of up to 4096 rows; own glue kernels", text);
}

test "CUDA startup: every other registered family retains its declared pass size" {
    inline for (.{ nemo, @import("nemotron").omni, @import("glm_cuda").native, @import("kolibri1").native }) |F| {
        const Loaded = @typeInfo(@typeInfo(@TypeOf(F.open)).@"fn".return_type.?).error_union.payload;
        try std.testing.expect(!@hasField(Loaded, "prompt_rows"));
        const loaded: Loaded = undefined;
        try std.testing.expectEqual(F.prompt_rows, startup.rows(F, loaded));
    }
}

test "CUDA startup: Flash Next cache cuts use the loaded cold pass size, and its replies' warm passes wait for quiet" {
    var loaded: flash.native.Loaded = undefined;
    loaded.prompt_rows = 4096;
    const rules = @import("cuda_snaps.zig").passRules(flash.native, startup.rows(flash.native, loaded));
    try std.testing.expectEqual(@as(u32, 4096), rules.chunk);
    try std.testing.expect(!rules.chunk_starts and rules.lookahead == 1);
    try std.testing.expect(rules.warm == .waits and !rules.planned); // a mark one token short of the pass's end
    try std.testing.expectEqual(nemo.warm_quiet, flash.native.warm_quiet);
}

fn qwenRows(raw: ?[]const u8, context: usize) !@import("qwen35").native.Loaded {
    const qwen = @import("qwen35");
    var e: qwen.Engine = undefined;
    e.scratch_rows = qwen.solo.scratchRows(try qwen.solo.rowsOf(raw, false), context);
    var loaded: qwen.native.Loaded = undefined;
    loaded.rows = 2;
    loaded.prompt_rows = @intCast(e.scratch_rows);
    return loaded;
}

test "CUDA startup: Qwen's real solo reader reports default 4096 and enabled 8192 in line and cache rules" {
    const qwen = @import("qwen35");
    inline for (.{ .{ @as(?[]const u8, null), 4096 }, .{ @as(?[]const u8, "8192"), 8192 } }) |case| {
        const loaded = try qwenRows(case[0], 32768);
        const text = try line(qwen.native, loaded);
        defer std.testing.allocator.free(text);
        try std.testing.expectEqualStrings(expected(case[1]), text);
        const rules = @import("cuda_snaps.zig").passRules(qwen.native, startup.rows(qwen.native, loaded));
        try std.testing.expectEqual(@as(u32, case[1]), rules.chunk);
        try std.testing.expect(rules.planned and rules.grid == 4096);
        try std.testing.expectEqual(@as(u32, 8192), qwen.native.prompt_rows); // the real upper bound stays
        try std.testing.expectEqual(@as(usize, 8192), qwen.engine.max_rows);
    }
}

test "CUDA startup: Qwen's context-capped loaded rows survive a later environment change" {
    const qwen = @import("qwen35");
    const name = "TF_QWEN35_CHUNK_SOLO";
    const previous = if (std.c.getenv(name)) |v| try std.testing.allocator.dupeSentinel(u8, std.mem.span(v), 0) else null;
    defer if (previous) |v| {
        _ = env.setenv(name, v, 1);
        std.testing.allocator.free(v);
    } else {
        _ = env.unsetenv(name);
    };
    const loaded = try qwenRows("8192", 6000);
    try std.testing.expectEqual(@as(c_int, 0), env.setenv(name, "4096", 1));
    try std.testing.expectEqual(@as(u32, 6000), startup.rows(qwen.native, loaded));
    try std.testing.expectEqual(@as(u32, 6000), @import("cuda_snaps.zig").passRules(qwen.native, startup.rows(qwen.native, loaded)).chunk);
    const text = try line(qwen.native, loaded);
    defer std.testing.allocator.free(text);
    try std.testing.expectEqualStrings(expected(6000), text);
    const tiny = try qwenRows(null, 1);
    try std.testing.expectEqual(@as(u32, 32), startup.rows(qwen.native, tiny));
}
