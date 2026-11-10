//! Shared affine4 quantization must equal separate calls and independent quantized FP64 products.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const c = core.contract.spec;
const Base = @import("contract_fp8_prompt.zig");
const Gpu = @import("check.zig").Gpu;

fn reference(item: *Base.Case) !void {
    const a = item.rig.gpu.gpa;
    const want = try a.alloc(f64, item.count());
    defer a.free(want);
    const magnitude = try a.alloc(f64, item.count());
    defer a.free(magnitude);
    try item.reference(want, magnitude);
    const out = try item.rig.outputBytes();
    const spec = Base.spec(item.config);
    for (want, magnitude, 0..) |value, mag, i| {
        const actual = c.read(spec.out, out, i);
        if (c.classOf(actual) != c.classOf(value)) return error.SharedFp8NonfiniteMismatch;
        if (std.math.isFinite(value) and @abs(actual - value) > c.bound(spec, item.shape, value, mag))
            return error.SharedFp8ReferenceMismatch;
    }
}

fn runCase(gpu: Gpu, stream: cuda.Stream, m: usize, k: usize, kind: cuda.qmmf.Out, exceptional: bool) !void {
    var first: Base.Case = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = .{ .kind = .affine4, .out = kind } };
    defer first.deinit();
    var second: Base.Case = .{ .rig = .{ .gpu = gpu, .s = stream }, .config = .{ .kind = .affine4, .out = kind } };
    defer second.deinit();
    const fill: core.contract.gates.Fill = if (exceptional) .nonfinite else .random;
    if (!try first.prepare(.{ .rows = m, .n = 129, .k = k }, fill, 0x6f8)) return error.SharedFp8ShapeRefused;
    if (!try second.prepare(.{ .rows = m, .n = 65, .k = k }, fill, 0x6f8)) return error.SharedFp8ShapeRefused;
    if (!std.mem.eql(u16, first.x, second.x)) return error.SharedFp8FixtureInputMismatch;
    const a = gpu.gpa;
    const first_inputs = try a.dupe(u8, try first.rig.inputBytes());
    defer a.free(first_inputs);
    const second_inputs = try a.dupe(u8, try second.rig.inputBytes());
    defer a.free(second_inputs);
    try first.run();
    try second.run();
    const first_bytes = try a.dupe(u8, try first.rig.outputBytes());
    defer a.free(first_bytes);
    const second_bytes = try a.dupe(u8, try second.rig.outputBytes());
    defer a.free(second_bytes);
    var weights = [_]cuda.qlinear.Affine4{ first.affine, second.affine };
    const outputs = [_]u64{ first.out, second.out };
    for (0..5) |_| {
        try first.native.?.affine4Many(stream, first.dx, k, m, &weights, &outputs, kind);
        try stream.synchronize();
        if (!std.mem.eql(u8, first_bytes, try first.rig.outputBytes()) or
            !std.mem.eql(u8, second_bytes, try second.rig.outputBytes())) return error.SharedFp8BaselineMismatch;
    }
    try reference(&first);
    try reference(&second);
    if (!std.mem.eql(u8, first_inputs, try first.rig.inputBytes()) or
        !std.mem.eql(u8, second_inputs, try second.rig.inputBytes())) return error.SharedFp8InputMutation;

    // A different input with the same shape must replace every code; restoring the first must restore its bits.
    const changed = try a.dupe(u16, first.x);
    defer a.free(changed);
    for (changed) |*bits| bits.* ^= 0x8000;
    const dx = try first.rig.input(std.mem.sliceAsBytes(changed));
    const changed_inputs = try a.dupe(u8, try first.rig.inputBytes());
    defer a.free(changed_inputs);
    const changed_q = try a.dupe(u8, first.q);
    defer a.free(changed_q);
    for (changed_q) |*code| code.* ^= 0x80;
    const original_first = first.q;
    const original_second = second.q;
    first.q = changed_q;
    second.q = changed_q;
    defer {
        first.q = original_first;
        second.q = original_second;
    }
    try first.native.?.affine4Many(stream, dx, k, m, &weights, &outputs, kind);
    try stream.synchronize();
    try reference(&first);
    try reference(&second);
    first.q = original_first;
    second.q = original_second;
    try first.native.?.affine4Many(stream, first.dx, k, m, &weights, &outputs, kind);
    try stream.synchronize();
    if (!std.mem.eql(u8, first_bytes, try first.rig.outputBytes()) or
        !std.mem.eql(u8, second_bytes, try second.rig.outputBytes())) return error.SharedFp8StaleInput;
    try std.testing.expectError(error.InvalidFp8PromptAlias, first.native.?.affine4Many(stream, first.dx, k, m, &weights, &.{ first.out, first.out }, kind));
    for ([_]u64{ first.dx, first.native.?.scratch.ptr, weights[0].w, weights[1].s, weights[1].b }) |address|
        try std.testing.expectError(error.InvalidFp8PromptAlias, first.native.?.affine4Many(stream, first.dx, k, m, &weights, &.{ address, second.out }, kind));
    weights[1].k = k - 64;
    try std.testing.expectError(error.InvalidFp8PromptShape, first.native.?.affine4Many(stream, first.dx, k, m, &weights, &outputs, kind));
    weights[1].k = k;
    if (!std.mem.eql(u8, first_bytes, try first.rig.outputBytes()) or
        !std.mem.eql(u8, second_bytes, try second.rig.outputBytes())) return error.SharedFp8RefusalWroteOutput;
    if (!std.mem.eql(u8, changed_inputs, try first.rig.inputBytes()) or
        !std.mem.eql(u8, second_inputs, try second.rig.inputBytes())) return error.SharedFp8InputMutation;
    if (!try first.rig.intact() or !try second.rig.intact()) return error.SharedFp8GuardWrite;
    std.debug.print("PASS shared affine FP8 M={d} K={d} out={t} exceptional={} separate bits, FP64, five repeats, changed input, guards\n", .{ m, k, kind, exceptional });
}

pub fn run(gpu: Gpu) !void {
    const sm = try gpu.ctx.capability();
    if (sm != 120 and sm != 121) return error.UnsupportedNativeFp8Prefill;
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var passed: usize = 0;
    for ([_][2]usize{ .{ 17, 192 }, .{ 65, 192 }, .{ 17, 5120 } }) |shape|
        for ([_]cuda.qmmf.Out{ .bf16, .f32 }) |kind| for ([_]bool{ false, true }) |exceptional| {
            try runCase(gpu, stream, shape[0], shape[1], kind, exceptional);
            passed += 1;
        };
    std.debug.print("SUMMARY shared affine FP8: {d} cases pass\n", .{passed});
}

test "FP8 prompt shared affine device contract compiles without running a GPU" {
    std.mem.doNotOptimizeAway(&run);
}
