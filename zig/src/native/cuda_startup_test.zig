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
        "1.0 GiB a stream", @as(f64, 20), @as(f64, 2), startup.rows(F, loaded), "",             "", "own glue kernels",
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
    try check(null, 1536);
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
    inline for (.{ nemo, @import("nemotron").omni, @import("glm_cuda").native, @import("qwen35").native, @import("qwen35").native_moe, @import("kolibri1").native }) |F| {
        const Loaded = @typeInfo(@typeInfo(@TypeOf(F.open)).@"fn".return_type.?).error_union.payload;
        try std.testing.expect(!@hasField(Loaded, "prompt_rows"));
        const loaded: Loaded = undefined;
        try std.testing.expectEqual(F.prompt_rows, startup.rows(F, loaded));
    }
}
