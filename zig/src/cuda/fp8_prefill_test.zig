//! Records the real shared-input wrapper's launches without a CUDA device.
const std = @import("std");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Native = @import("fp8_prefill.zig").Native;
const Function = @import("module.zig").Function;
const qmmf = @import("qmmf.zig");
const Affine4 = @import("qlinear.zig").Affine4;
const tiles = @import("fp8_prefill_tiles.zig");
const Prepared = @import("affine_fp8_prepared.zig").Prepared;
const Stream = @import("stream.zig").Stream;
const T = std.testing;

const Trace = struct {
    const Call = struct {
        symbol: usize,
        input: u64 = 0,
        codes: u64 = 0,
        scales: u64 = 0,
        sums: u64 = 0,
        weight: u64 = 0,
        output: u64 = 0,
        rows: i32 = 0,
        width: i32 = 0,
        stride: i32 = 0,
        grid_x: u32 = 0,
        grid_y: u32 = 0,
        shared: u32 = 0,
    };
    var calls: [16]Call = undefined;
    var count: usize = 0;
    var unloaded: [2]usize = undefined;
    var unload_count: usize = 0;
    var query_count: usize = 0;
    var busy: bool = false;
    fn query(_: abi.Stream) callconv(.c) abi.Result {
        query_count += 1;
        return if (busy) abi.error_not_ready else abi.success;
    }
    fn unload(m: abi.Module) callconv(.c) abi.Result {
        unloaded[unload_count] = @intFromPtr(m.?);
        unload_count += 1;
        return abi.success;
    }
    fn arg(comptime V: type, args: ?[*]?*anyopaque, at: usize) V {
        return @as(*const V, @ptrCast(@alignCast(args.?[at].?))).*;
    }
    fn launch(f: abi.Function, gx: c_uint, gy: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, shared: c_uint, _: abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
        const symbol = @intFromPtr(f.?);
        calls[count] = if (symbol == 1) .{
            .symbol = symbol,
            .input = arg(u64, args, 0),
            .codes = arg(u64, args, 1),
            .scales = arg(u64, args, 2),
            .output = 0,
            .rows = arg(i32, args, 3),
            .width = arg(i32, args, 4),
            .stride = arg(i32, args, 5),
        } else if (symbol == 4) .{
            .symbol = symbol,
            .input = arg(u64, args, 0),
            .codes = arg(u64, args, 1),
            .rows = arg(i32, args, 2),
            .width = arg(i32, args, 3),
            .stride = arg(i32, args, 4),
        } else if (symbol == 6) .{
            .symbol = symbol,
            .input = arg(u64, args, 0),
            .codes = arg(u64, args, 1),
            .scales = arg(u64, args, 2),
            .sums = arg(u64, args, 3),
            .rows = arg(i32, args, 4),
            .width = arg(i32, args, 5),
            .stride = arg(i32, args, 6),
        } else if (symbol == 7 or symbol == 8 or symbol == 21 or symbol == 22) .{
            .symbol = symbol,
            .codes = arg(u64, args, 0),
            .scales = arg(u64, args, 1),
            .sums = arg(u64, args, 2),
            .weight = arg(u64, args, 3),
            .output = arg(u64, args, 6),
            .rows = arg(i32, args, 7),
            .width = arg(i32, args, 9),
        } else if (symbol == 20) .{
            .symbol = symbol,
            .weight = arg(u64, args, 0),
            .codes = arg(u64, args, 1),
            .width = arg(i32, args, 2),
        } else if (symbol >= 10) .{
            .symbol = symbol,
            .codes = arg(u64, args, 0),
            .weight = arg(u64, args, 1),
            .output = arg(u64, args, 2),
            .rows = arg(i32, args, 3),
            .width = arg(i32, args, 5),
            .scales = arg(u64, args, 8),
        } else .{
            .symbol = symbol,
            .input = 0,
            .codes = arg(u64, args, 0),
            .scales = arg(u64, args, 1),
            .output = arg(u64, args, 4),
            .rows = arg(i32, args, 5),
            .width = arg(i32, args, 7),
            .stride = 0,
        };
        calls[count].grid_x = gx;
        calls[count].grid_y = gy;
        calls[count].shared = shared;
        count += 1;
        return abi.success;
    }
};
fn function(d: *const Driver, symbol: usize) Function {
    return .{ .d = d, .handle = @ptrFromInt(symbol) };
}
fn native(d: *const Driver) Native {
    return .{
        .mod = undefined,
        .quant = function(d, 4),
        .affine_quant = function(d, 6),
        .row_quant = function(d, 1),
        .dense = undefined,
        .affine = .{ function(d, 7), function(d, 8) },
        .row = .{ function(d, 2), function(d, 3) },
        .scratch = .{ .d = d, .ptr = 0x100000, .len = 65536, .borrowed = true },
        .max_rows = 129,
        .max_width = 256,
    };
}

fn optional(d: *const Driver, mode: tiles.Mode) tiles.Tiles {
    return .{
        .module = .{ .d = d, .handle = @ptrFromInt(101) },
        .mode = mode,
        .functions = .{ .{ function(d, 10), function(d, 11) }, .{ function(d, 12), function(d, 13) } },
    };
}
fn weight(n: u32, k: u32, base: u64) qmmf.Weight {
    return .{ .mode = .fp8g, .codes = base, .scales = base + 0x10000, .n = n, .npad = (n + 127) / 128 * 128, .k = k };
}

fn affine(n: usize, k: usize, base: u64) Affine4 {
    return .{ .w = base, .s = base + 0x10000, .b = base + 0x20000, .n = n, .npad = (n + 127) / 128 * 128, .k = k };
}

fn prepared(d: *const Driver, bytes: usize) Prepared {
    return .{ .module = .{ .d = d, .handle = @ptrFromInt(103) }, .prepare = function(d, 20), .products = .{ function(d, 21), function(d, 22) }, .buffer = .{ .d = d, .ptr = 0x800000, .len = bytes, .borrowed = true } };
}

test "FP8 prompt prepared affine shares quantization and repacks every matrix into one buffer" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var n = native(&d);
    const s: Stream = .{ .d = &d, .handle = null };
    const weights = [_]Affine4{ affine(129, 192, 0x400000), affine(65, 192, 0x500000) };
    const outs = [_]u64{ 0x600000, 0x700000 };
    n.max_rows = 4096;
    n.scratch.len = 1024 * 192 + 1024 * 3 * 8;
    n.affine_prepared = prepared(&d, 256 * 192);
    for ([_]qmmf.Out{ .bf16, .f32 }) |kind| {
        Trace.count = 0;
        try n.affine4Many(s, 0x200000, 192, 1024, &weights, &outs, kind);
        try T.expectEqual(@as(usize, 5), Trace.count);
        try T.expectEqual(@as(usize, 6), Trace.calls[0].symbol);
        for (weights, 0..) |w, i| {
            const pack = Trace.calls[1 + 2 * i];
            const product = Trace.calls[2 + 2 * i];
            try T.expectEqual(@as(usize, 20), pack.symbol);
            try T.expectEqual(w.w, pack.weight);
            try T.expectEqual(@as(u64, 0x800000), pack.codes);
            try T.expectEqual(@as(u32, 3), pack.grid_x);
            try T.expectEqual(@as(u32, @intCast(w.npad / 64)), pack.grid_y);
            try T.expectEqual(@as(usize, if (kind == .f32) 22 else 21), product.symbol);
            try T.expectEqual(pack.codes, product.weight);
            try T.expectEqual(Trace.calls[0].codes, product.codes);
            try T.expectEqual(Trace.calls[0].scales, product.scales);
            try T.expectEqual(Trace.calls[0].sums, product.sums);
        }
        try n.affine4Many(s, 0x280000, 192, 1024, &weights, &outs, kind);
        try T.expectEqual(@as(usize, 10), Trace.count);
        try T.expectEqual(@as(usize, 20), Trace.calls[6].symbol);
        try T.expectEqual(@as(usize, 20), Trace.calls[8].symbol);
    }
    Trace.count = 0;
    try n.affine4Many(s, 0x200000, 192, 65, &weights, &outs, .bf16);
    try T.expectEqual(@as(usize, 3), Trace.count);
    try T.expectEqual(@as(usize, 7), Trace.calls[1].symbol);
    n.affine_prepared.?.buffer.?.len = 256 * 192 - 1;
    Trace.count = 0;
    try n.affine4Many(s, 0x200000, 192, 1024, &weights, &outs, .bf16);
    try T.expectEqual(@as(usize, 4), Trace.count);
    try T.expectEqual(@as(usize, 7), Trace.calls[1].symbol);
    try T.expectEqual(@as(usize, 20), Trace.calls[2].symbol);
}

test "FP8 prompt prepared affine refuses busy stream and pool aliases before quantization" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    d.api.cuStreamQuery = Trace.query;
    var n = native(&d);
    n.max_rows = 4096;
    n.scratch.len = 1024 * 192 + 1024 * 3 * 8;
    n.affine_prepared = prepared(&d, 256 * 192);
    n.affine_prepared.?.last_stream = .{ .d = &d, .handle = @ptrFromInt(1) };
    const s: Stream = .{ .d = &d, .handle = @ptrFromInt(2) };
    const w = affine(129, 192, 0x400000);
    Trace.busy = true;
    Trace.query_count = 0;
    Trace.count = 0;
    try T.expectError(error.AffinePreparedBusy, n.affine4(s, 0x200000, 1024, w, 0x600000, .bf16));
    try T.expectEqual(@as(usize, 0), Trace.count);
    try T.expectEqual(@as(usize, 1), Trace.query_count);
    Trace.busy = false;
    try T.expectError(error.InvalidFp8PromptAlias, n.affine4(s, 0x800000, 1024, w, 0x600000, .bf16));
    try T.expectError(error.InvalidFp8PromptAlias, n.affine4(s, 0x200000, 1024, w, 0x800000, .bf16));
    try T.expectEqual(@as(usize, 0), Trace.count);
    try n.affine4(s, 0x200000, 1024, w, 0x600000, .bf16);
    try T.expectEqual(@as(usize, 3), Trace.count);
    n.affine_prepared = null;
    try n.reserveAffinePacked(256 * 192);
    try T.expectEqual(@as(usize, 0), n.affinePackedBytes());
}

test "FP8 prompt shared affine projections preserve each product and quantize only once" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var n = native(&d);
    const s: Stream = .{ .d = &d, .handle = null };
    const weights = [_]Affine4{ affine(129, 192, 0x400000), affine(65, 192, 0x500000) };
    const outs = [_]u64{ 0x600000, 0x700000 };
    for ([_]qmmf.Out{ .bf16, .f32 }) |kind| {
        Trace.count = 0;
        for (weights, outs) |w, out| try n.affine4(s, 0x200000, 65, w, out, kind);
        const first = Trace.calls[1];
        const second = Trace.calls[3];
        Trace.count = 0;
        try n.affine4Many(s, 0x200000, 192, 65, &weights, &outs, kind);
        try T.expectEqual(@as(usize, 3), Trace.count);
        try T.expectEqualDeep(first, Trace.calls[1]);
        try T.expectEqualDeep(second, Trace.calls[2]);
        try T.expectEqual(@as(usize, 6), Trace.calls[0].symbol);
        try T.expectEqual(n.scratch.ptr, Trace.calls[0].codes);
        try T.expectEqual(n.scratch.ptr + 65 * 192, Trace.calls[0].scales);
        try T.expectEqual(n.scratch.ptr + 65 * 192 + 65 * 3 * 4, Trace.calls[0].sums);
        try n.affine4Many(s, 0x300000, 195, 65, &weights, &outs, kind);
        try T.expectEqual(@as(usize, 6), Trace.count);
        try T.expectEqual(@as(u64, 0x300000), Trace.calls[3].input);
        try T.expectEqual(@as(i32, 195), Trace.calls[3].stride);
    }
}

test "FP8 prompt shared affine rejects aliases and incompatible geometry before quantization" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var n = native(&d);
    const s: Stream = .{ .d = &d, .handle = null };
    var weights = [_]Affine4{ affine(129, 192, 0x400000), affine(65, 192, 0x500000) };
    const outs = [_]u64{ 0x600000, 0x700000 };
    Trace.count = 0;
    weights[1].k = 128;
    try T.expectError(error.InvalidFp8PromptShape, n.affine4Many(s, 0x200000, 192, 65, &weights, &outs, .bf16));
    weights[1].k = 192;
    weights[1].npad = 127;
    try T.expectError(error.InvalidFp8PromptShape, n.affine4Many(s, 0x200000, 192, 65, &weights, &outs, .bf16));
    weights[1].npad = 128;
    for ([_]u64{ 0x200000, 0x100000, outs[0], weights[0].w, weights[1].s, weights[1].b }) |address|
        try T.expectError(error.InvalidFp8PromptAlias, n.affine4Many(s, 0x200000, 192, 65, &weights, &.{ address, outs[0] }, .bf16));
    weights[1].b = n.scratch.ptr;
    try T.expectError(error.InvalidFp8PromptAlias, n.affine4Many(s, 0x200000, 192, 65, &weights, &outs, .bf16));
    weights[1].b = 0x520000;
    n.scratch.len = 65 * 192;
    try T.expectError(error.InvalidFp8PromptShape, n.affine4Many(s, 0x200000, 192, 65, &weights, &outs, .bf16));
    try T.expectEqual(@as(usize, 0), Trace.count);
}

test "FP8 prompt shared rows quantize exactly once and reuse the same operand pointers" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var n = native(&d);
    const s: Stream = .{ .d = &d, .handle = null };
    const ws = [_]qmmf.Weight{ weight(129, 192, 0x400000), weight(65, 192, 0x500000) };
    const outs = [_]u64{ 0x600000, 0x700000 };
    for ([_]qmmf.Out{ .bf16, .f32 }) |kind| {
        Trace.count = 0;
        try n.rowScaledMany(s, 0x200000, 195, 65, &ws, &outs, kind);
        try T.expectEqual(@as(usize, 3), Trace.count);
        try T.expectEqual(@as(usize, 1), Trace.calls[0].symbol);
        try T.expectEqual(@as(u64, 0x200000), Trace.calls[0].input);
        try T.expectEqual(@as(i32, 195), Trace.calls[0].stride);
        for (Trace.calls[1..3], outs) |call, output| {
            try T.expectEqual(@as(usize, if (kind == .f32) 3 else 2), call.symbol);
            try T.expectEqual(Trace.calls[0].codes, call.codes);
            try T.expectEqual(Trace.calls[0].scales, call.scales);
            try T.expectEqual(output, call.output);
        }
        // A second input must always run another quantizer, even with identical dimensions and scratch.
        try n.rowScaledMany(s, 0x300000, 195, 65, &ws, &outs, kind);
        try T.expectEqual(@as(usize, 6), Trace.count);
        try T.expectEqual(@as(usize, 1), Trace.calls[3].symbol);
        try T.expectEqual(@as(u64, 0x300000), Trace.calls[3].input);
    }
}

test "FP8 prompt shared rows reject incompatible widths and aliasing before any launch" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    var n = native(&d);
    const s: Stream = .{ .d = &d, .handle = null };
    var ws = [_]qmmf.Weight{ weight(129, 192, 0x400000), weight(65, 192, 0x500000) };
    const outs = [_]u64{ 0x600000, 0x700000 };
    Trace.count = 0;
    ws[1].k = 128;
    try T.expectError(error.InvalidFp8PromptShape, n.rowScaledMany(s, 0x200000, 192, 65, &ws, &outs, .bf16));
    ws[1].k = 192;
    try T.expectError(error.InvalidFp8PromptAlias, n.rowScaledMany(s, 0x200000, 192, 65, &ws, &.{ 0x600000, 0x600002 }, .bf16));
    try T.expectError(error.InvalidFp8PromptAlias, n.rowScaledMany(s, 0x200000, 192, 65, &ws, &.{ 0x200000, 0x700000 }, .bf16));
    try T.expectError(error.InvalidFp8PromptAlias, n.rowScaledMany(s, 0x200000, 192, 65, &ws, &.{ 0x100000, 0x700000 }, .bf16));
    try T.expectError(error.InvalidFp8PromptAlias, n.rowScaledMany(s, 0x200000, 192, 65, &ws, &.{ ws[1].codes, 0x700000 }, .bf16));
    try T.expectEqual(@as(usize, 0), Trace.count);
}

test "FP8 prompt optional tiles reuse each quantizer and exact scratch addresses" {
    var d: Driver = undefined;
    d.api.cuLaunchKernel = Trace.launch;
    const s: Stream = .{ .d = &d, .handle = null };
    const ws = [_]qmmf.Weight{ weight(129, 192, 0x400000), weight(65, 192, 0x500000) };
    const outs = [_]u64{ 0x600000, 0x700000 };
    for ([_]tiles.Mode{ .tile128x64, .tile128x128, .tile64x64 }) |mode| {
        var n = native(&d);
        n.tiles = optional(&d, mode);
        const geometry = tiles.geometry(mode);
        for ([_]qmmf.Out{ .bf16, .f32 }) |kind| {
            Trace.count = 0;
            try n.rowScaledMany(s, 0x200000, 195, 65, &ws, &outs, kind);
            try T.expectEqual(@as(usize, 3), Trace.count);
            try T.expectEqual(@as(usize, 1), Trace.calls[0].symbol);
            for (Trace.calls[1..3], ws, outs) |call, w, out| {
                try T.expectEqual(@as(usize, if (kind == .bf16) 12 else 13), call.symbol);
                try T.expectEqual(n.scratch.ptr, call.codes);
                try T.expectEqual(Trace.calls[0].scales, call.scales);
                try T.expectEqual(w.codes, call.weight);
                try T.expectEqual(out, call.output);
                try T.expectEqual((w.n + geometry.columns - 1) / geometry.columns, call.grid_x);
                try T.expectEqual(@as(u32, (65 + geometry.rows - 1) / geometry.rows), call.grid_y);
                try T.expectEqual(geometry.shared, call.shared);
            }
            var w = ws[0];
            w.mode = .fp8;
            w.act = 0.125;
            Trace.count = 0;
            try n.matmul(s, 0x300000, 195, 65, w, outs[0], kind);
            try T.expectEqual(@as(usize, 2), Trace.count);
            try T.expectEqual(@as(usize, 4), Trace.calls[0].symbol);
            try T.expectEqual(@as(usize, if (kind == .bf16) 10 else 11), Trace.calls[1].symbol);
            try T.expectEqual(Trace.calls[0].codes, Trace.calls[1].codes);
            try T.expectEqual(@as(u64, 0), Trace.calls[1].scales);
            try T.expectEqual(n.scratch.ptr, Trace.calls[1].codes);
        }
        Trace.count = 0;
        try T.expectError(error.InvalidFp8PromptAlias, n.rowScaledMany(s, 0x200000, 195, 65, &ws, &.{ outs[0], outs[0] }, .bf16));
        try T.expectEqual(@as(usize, 0), Trace.count);
    }
}

test "FP8 prompt optional module has one owner and never frees borrowed scratch" {
    var d: Driver = undefined;
    d.api.cuModuleUnload = Trace.unload;
    for ([_]bool{ false, true }) |enabled| {
        var n = native(&d);
        n.mod = .{ .d = &d, .handle = @ptrFromInt(102) };
        if (enabled) n.tiles = optional(&d, .tile128x128);
        Trace.unload_count = 0;
        const scratch = n.scratch.ptr;
        n.unload();
        try T.expectEqual(@as(usize, if (enabled) 2 else 1), Trace.unload_count);
        if (enabled) try T.expectEqual(@as(usize, 101), Trace.unloaded[0]);
        try T.expectEqual(@as(usize, 102), Trace.unloaded[Trace.unload_count - 1]);
        try T.expectEqual(@as(u64, 0x100000), scratch);
    }
}
