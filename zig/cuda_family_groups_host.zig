//! Family adapters call the real affine launchers through an inert CUDA driver.
const std = @import("std");
const cuda = @import("cuda");
const gemma = @import("gemma4");
const spark = @import("spark25");
const fixture = @import("qwen35_verify_host.zig");
const Trace = fixture.Trace;
const t = std.testing;
const x = 0x10000;
const xs = 0x20000;
const out = 0x30000;
const weight: cuda.qlinear.Affine4 = .{ .w = 10, .s = 20, .b = 30, .n = 4096, .k = 8192, .npad = 4096 };
const rows_set = [_]usize{ 1, 4, 16, 17, 32, 33, 64, 65, 128, 129, 256 };
fn reset() void {
    Trace.count = 0;
    Trace.prompts = 0;
    Trace.grouped = 0;
}
fn check(rows: usize, parts: usize, tile: usize, grouped: bool, prompt: bool) !void {
    const per = (rows + tile - 1) / tile;
    try t.expectEqual(@as(usize, if (grouped and !prompt) per else 0), Trace.grouped);
    try t.expectEqual(@as(usize, if (prompt) parts else 0), Trace.prompts);
    try t.expectEqual(@as(usize, if (prompt) 0 else parts * per), Trace.count);
    for (Trace.calls[0..Trace.count], 0..) |call, i| {
        const at = (if (grouped) i / parts else i % per) * tile;
        const part = if (grouped) i % parts else i / per;
        try t.expectEqual(x + at * weight.k * 2, call.x);
        try t.expectEqual(xs + at * (weight.k / 64) * 4, call.xs);
        try t.expectEqual(out + part * 0x1000000 + at * weight.n * 2, call.out);
        try t.expectEqual(@as(c_int, @intCast(@min(tile, rows - at))), call.rows);
        try t.expectEqual(@as(c_int, @intCast(cuda.qlinear.splitK(weight.n, weight.k))), call.sk);
    }
}
test "GB10 family grouping: Gemma target QKV QK and MLP preserve native and prompt routes" {
    if (comptime !@hasDecl(gemma.forward.Model, "projectMany")) return error.TestExpectedEqual;
    for ([_]bool{ false, true }) |gb10| {
        var d: cuda.Driver = undefined;
        const e = fixture.engine(&d, gb10);
        var k: gemma.kernels.Kernels = undefined;
        k.affine = e.affine;
        const m: gemma.forward.Model = undefined;
        const ps = [_]gemma.weights.Proj{ .{ .a4 = weight }, .{ .a4 = weight }, .{ .a4 = weight } };
        for ([_]bool{ false, true }) |prompt| for (rows_set) |rows| for ([_]usize{ 2, 3 }) |parts| {
            reset();
            try m.projectMany(&k, e.stream, ps[0..parts], x, xs, (&[_]u64{ out, out + 0x1000000, out + 0x2000000 })[0..parts], rows, prompt);
            try check(rows, parts, k.affine.verifyRows(), gb10 and !prompt, prompt);
        };
    }
}
test "GB10 family grouping: Gemma assistant and DFlash retain the latter's prompt cutoff" {
    if (comptime !@hasDecl(gemma.assistant.Drafter, "projectMany") or !@hasDecl(gemma.dflash.Drafter, "projectMany")) return error.TestExpectedEqual;
    for ([_]bool{ false, true }) |gb10| {
        var d: cuda.Driver = undefined;
        const e = fixture.engine(&d, gb10);
        var k: gemma.kernels.Kernels = undefined;
        k.affine = e.affine;
        var assistant: gemma.assistant.Drafter = undefined;
        assistant.k = &k;
        assistant.s = e.stream;
        var dflash: gemma.dflash.Drafter = undefined;
        dflash.k = &k;
        dflash.s = e.stream;
        for (rows_set) |rows| {
            reset();
            try assistant.projectMany(&.{ weight, weight }, x, xs, &.{ out, out + 0x1000000 }, rows);
            try check(rows, 2, k.affine.verifyRows(), gb10, false);
            reset();
            try dflash.projectMany(&.{ weight, weight, weight }, x, xs, &.{ out, out + 0x1000000, out + 0x2000000 }, rows);
            try check(rows, 3, k.affine.verifyRows(), gb10, rows > k.affine.verifyRows());
        }
    }
}
test "GB10 family grouping: Gemma mixed bit widths keep separate projections" {
    if (comptime !@hasDecl(gemma.forward.Model, "projectMany")) return error.TestExpectedEqual;
    var d: cuda.Driver = undefined;
    const e = fixture.engine(&d, true);
    var k: gemma.kernels.Kernels = undefined;
    k.affine = e.affine;
    k.wide.gemv_fns = @splat(e.affine.gemv_fn);
    k.wide.gemv_blocks = @splat(1);
    k.wide.pdl = true;
    k.wide.split = null;
    const m: gemma.forward.Model = undefined;
    const other: cuda.qlinear.AffineN = .{ .w = 40, .s = 50, .b = 60, .n = weight.n, .k = weight.k, .npad = weight.npad, .bits = 6 };
    reset();
    try m.projectMany(&k, e.stream, &.{ .{ .a4 = weight }, .{ .an = other } }, x, xs, &.{ out, out + 0x1000000 }, 128, false);
    try t.expectEqual(@as(usize, 0), Trace.grouped);
    try t.expectEqual(@as(usize, 9), Trace.count);
}

test "GB10 family grouping: Spark MLP preserves prompts and other GPUs" {
    if (comptime !@hasDecl(spark.forward.Model, "projectMany")) return error.TestExpectedEqual;
    for ([_]bool{ false, true }) |gb10| {
        var d: cuda.Driver = undefined;
        const e = fixture.engine(&d, gb10);
        var k: spark.kernels.Kernels = undefined;
        k.affine = e.affine;
        const m: spark.forward.Model = undefined;
        const ps = [_]spark.weights.Proj{ .{ .a4 = weight }, .{ .a4 = weight } };
        for ([_]bool{ false, true }) |prompt| for (rows_set) |rows| {
            reset();
            try m.projectMany(&k, e.stream, &ps, x, xs, &.{ out, out + 0x1000000 }, rows, prompt);
            try check(rows, 2, k.affine.verifyRows(), gb10 and !prompt, prompt);
        };
        k.wide.gemv_fns = @splat(e.affine.gemv_fn);
        k.wide.gemv_blocks = @splat(1);
        k.wide.pdl = gb10;
        k.wide.split = null;
        const other: cuda.qlinear.AffineN = .{ .w = 40, .s = 50, .b = 60, .n = weight.n, .k = weight.k, .npad = weight.npad, .bits = 8 };
        reset();
        try m.projectMany(&k, e.stream, &.{ .{ .a4 = weight }, .{ .an = other } }, x, xs, &.{ out, out + 0x1000000 }, 128, false);
        try t.expectEqual(@as(usize, 0), Trace.grouped);
        try t.expectEqual(@as(usize, if (gb10) 9 else 16), Trace.count);
    }
}
