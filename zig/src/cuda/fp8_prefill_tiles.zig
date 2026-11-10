//! Explicit geometry opt-in for the native FP8 products; this module owns no data buffers.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const kernels = @import("kernels.zig");
const qmmf = @import("qmmf.zig");

pub const Mode = enum(u8) { baseline, tile128x64, tile128x128, tile64x64 };
pub const Geometry = struct { rows: u32, columns: u32, shared: u32 };
pub fn selected(raw: ?[]const u8) !Mode {
    const value = raw orelse return .baseline;
    if (std.mem.eql(u8, value, "0")) return .baseline;
    if (std.mem.eql(u8, value, "1")) return .tile128x64;
    if (std.mem.eql(u8, value, "2")) return .tile128x128;
    if (std.mem.eql(u8, value, "3")) return .tile64x64;
    return error.InvalidFp8PromptTile;
}
pub fn requested() !Mode {
    return selected(try @import("sys").environment.given("TF_QWEN_FP8_TILE"));
}
pub fn geometry(mode: Mode) Geometry {
    return switch (mode) {
        .baseline => .{ .rows = 64, .columns = 128, .shared = 0 },
        .tile128x64 => .{ .rows = 128, .columns = 64, .shared = 49152 },
        .tile128x128 => .{ .rows = 128, .columns = 128, .shared = 65536 },
        .tile64x64 => .{ .rows = 64, .columns = 64, .shared = 32768 },
    };
}
pub const symbols = [_][2][2][:0]const u8{
    .{ .{ "tf_fp8_tile128x64", "tf_fp8_tile128x64_f32" }, .{ "tf_fp8_tile128x64_row", "tf_fp8_tile128x64_row_f32" } },
    .{ .{ "tf_fp8_tile128x128", "tf_fp8_tile128x128_f32" }, .{ "tf_fp8_tile128x128_row", "tf_fp8_tile128x128_row_f32" } },
    .{ .{ "tf_fp8_tile64x64", "tf_fp8_tile64x64_f32" }, .{ "tf_fp8_tile64x64_row", "tf_fp8_tile64x64_row_f32" } },
};
pub const Tiles = struct {
    module: Module,
    mode: Mode,
    functions: [2][2]Function,

    pub fn load(d: *const Driver, mode: Mode) !Tiles {
        if (mode == .baseline) return error.InvalidFp8PromptTile;
        const image = if (@hasDecl(kernels, "fp8_prefill_tiles")) kernels.fp8_prefill_tiles else return error.BuiltWithoutFp8Tiles;
        return loadImage(d, mode, image);
    }
    fn loadImage(d: *const Driver, mode: Mode, image: []const u8) !Tiles {
        var module = try Module.load(d, image);
        errdefer module.unload();
        var functions: [2][2]Function = undefined;
        const names = symbols[@backingInt(mode) - 1];
        for (&functions, names) |*row, row_names| for (row, row_names) |*f, name| {
            f.* = try module.function(name);
            try f.allowDynamicShared(geometry(mode).shared);
        };
        return .{ .module = module, .mode = mode, .functions = functions };
    }
    pub fn unload(t: *Tiles) void {
        t.module.unload();
    }

    pub fn product(t: *const Tiles, s: Stream, x: u64, row_scales: u64, m: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out, row: bool) !void {
        var args: launch.Args = .{};
        for ([_]u64{ x, w.codes, out }) |value| args.add(value);
        for ([_]usize{ m, w.n, w.k, w.npad }) |value| args.add(@as(i32, @intCast(value)));
        args.add(@as(f32, if (row) 1 else w.act * w.scale));
        args.add(row_scales);
        args.add(if (row) w.scales else @as(u64, 0));
        const g = geometry(t.mode);
        try launch.launch(t.functions[@intFromBool(row)][@intFromBool(kind == .f32)], .{
            .grid = .{ .x = @intCast((w.n + g.columns - 1) / g.columns), .y = @intCast((m + g.rows - 1) / g.rows) },
            .block = .{ .x = 256 },
            .shared = g.shared,
        }, s, &args);
    }
};

test "FP8 prompt tile flag is strict and defaults to the unchanged baseline" {
    try std.testing.expectEqual(Mode.baseline, try selected(null));
    try std.testing.expectEqual(Mode.baseline, try selected("0"));
    try std.testing.expectEqual(Mode.tile128x64, try selected("1"));
    try std.testing.expectEqual(Mode.tile128x128, try selected("2"));
    try std.testing.expectEqual(Mode.tile64x64, try selected("3"));
    for ([_][]const u8{ "", "4", "01", " 1", "1 ", "true" }) |bad|
        try std.testing.expectError(error.InvalidFp8PromptTile, selected(bad));
    try std.testing.expectEqual(@as(u32, 65536), geometry(.tile128x128).shared);
}

test "FP8 prompt tile load opts in all symbols and cleans partial module initialization" {
    const abi = @import("abi.zig");
    const Fake = struct {
        var functions: usize = 0;
        var attributes: usize = 0;
        var unloaded: usize = 0;
        var fail: bool = false;
        var invalid: bool = false;
        fn load(module: *abi.Module, _: ?*const anyopaque) callconv(.c) abi.Result {
            module.* = @ptrFromInt(1);
            return abi.success;
        }
        fn function(out: *abi.Function, _: abi.Module, symbol: [*:0]const u8) callconv(.c) abi.Result {
            const name = symbols[1][functions / 2][functions % 2];
            invalid = invalid or !std.mem.eql(u8, name, std.mem.span(symbol));
            functions += 1;
            out.* = @ptrFromInt(functions);
            return abi.success;
        }
        fn attribute(_: abi.Function, attr: abi.FunctionAttribute, bytes: c_int) callconv(.c) abi.Result {
            invalid = invalid or attr != .max_dynamic_shared_size_bytes or bytes != 65536;
            attributes += 1;
            return if (fail and attributes == 2) abi.error_not_ready else abi.success;
        }
        fn unload(_: abi.Module) callconv(.c) abi.Result {
            unloaded += 1;
            return abi.success;
        }
    };
    var d: Driver = undefined;
    d.api.cuModuleLoadData = Fake.load;
    d.api.cuModuleGetFunction = Fake.function;
    d.api.cuFuncSetAttribute = Fake.attribute;
    d.api.cuModuleUnload = Fake.unload;
    const image: [16]u8 align(16) = @splat(0);
    for ([_]bool{ false, true }) |fail| {
        Fake.functions = 0;
        Fake.attributes = 0;
        Fake.unloaded = 0;
        Fake.invalid = false;
        Fake.fail = fail;
        if (fail) {
            try std.testing.expectError(error.NotReady, Tiles.loadImage(&d, .tile128x128, &image));
            try std.testing.expectEqual(@as(usize, 2), Fake.attributes);
        } else {
            var loaded = try Tiles.loadImage(&d, .tile128x128, &image);
            try std.testing.expectEqual(@as(usize, 4), Fake.attributes);
            try std.testing.expectEqual(@as(usize, 0), Fake.unloaded);
            loaded.unload();
        }
        try std.testing.expect(!Fake.invalid);
        try std.testing.expectEqual(@as(usize, 1), Fake.unloaded);
    }
}
