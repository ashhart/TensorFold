//! The real GLM EXL3 caller and prompt launcher at the driver boundary, without a CUDA device.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm");
const t = std.testing;
const source = @embedFile("kernels/cuda/exl3_experts.cu");

const Record = struct {
    var copies: usize = 0;
    var up: usize = 0;
    var down: usize = 0;
    var decode: usize = 0;
    var pairs: [16]i32 = undefined;
    var experts: [16]i32 = undefined;
    var divisor: [16]i32 = undefined;
    var outputs: [16]u64 = undefined;
    var skip: [16]i32 = undefined;
    var routes: usize = 0;

    fn reset() void {
        copies = 0;
        up = 0;
        down = 0;
        decode = 0;
        routes = 0;
    }
    fn arg(comptime T: type, args: ?[*]?*anyopaque, index: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args.?[index].?))).*;
    }
    fn copy(_: *const cuda.abi.Memcpy2D, _: cuda.abi.Stream) callconv(.c) c_int {
        copies += 1;
        return 0;
    }
    fn launch(f: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, args: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        switch (@intFromPtr(f)) {
            1, 2 => {
                pairs[routes] = arg(i32, args, 1);
                experts[routes] = arg(i32, args, 2);
                routes += 1;
            },
            7 => {
                divisor[up] = arg(i32, args, 13);
                up += 1;
            },
            8 => {
                outputs[down] = arg(u64, args, 6);
                skip[down] = arg(i32, args, 9);
                down += 1;
            },
            9 => decode += 1,
            else => {},
        }
        return 0;
    }
    fn function(d: *cuda.Driver, id: usize) cuda.Function {
        return .{ .d = d, .handle = @ptrFromInt(id) };
    }
};

fn run(rows: usize, prompt: bool, shared: bool, count: usize) !void {
    Record.reset();
    var d: cuda.Driver = undefined;
    d.api.cuLaunchKernel = Record.launch;
    d.api.cuMemcpy2DAsync_v2 = Record.copy;
    var k: glm.exl3.Kernels = undefined;
    inline for (.{ "group", "rot_in", "gateup", "down_combine" }, .{ 9, 5, 13, 14 }) |field, id| @field(k, field) = Record.function(&d, id);
    k.grouped = .{ Record.function(&d, 10), Record.function(&d, 11), Record.function(&d, 12) };
    k.prompt = .{ .rotate = Record.function(&d, 6), .gateup = Record.function(&d, 7), .down = Record.function(&d, 8) };
    var ek: glm.experts.Kernels = undefined;
    inline for (.{ "plan_small", "plan_rows", "plan_rank", "plan_offsets", "plan_scatter", "plan_rank_rows", "plan_scatter_rows" }, .{ 1, 18, 2, 3, 4, 19, 20 }) |field, id| @field(ek.router, field) = Record.function(&d, id);
    const w = glm.exl3.Experts{ .gt = 101, .ut = 102, .dt = 103, .k2 = 104, .suh_g = 105, .suh_u = 106, .svh_g = 107, .svh_u = 108, .suh_d = 109, .svh_d = 110, .shx = shared };
    const sc = glm.exl3.Scratch{ .picks = 0x100000000, .xg = 0, .xu = 0, .z = 0 };
    const plan = cuda.grouped.Plan{ .members = 201, .items = 202, .counts = 203, .rank = 204, .hist = 205 };
    try glm.exl3.run(&k, &ek, .{ .d = &d, .handle = null }, w, sc, plan, 0x200000000, 512, 0x300000000, 9, rows, count, 512, 256, 7, prompt, 0x400000000, 0);
}

test "GLM EXL3 prompt: a whole pass makes one expert call and no 2D copies" {
    for ([_]bool{ false, true }) |shared| for ([_]usize{ 2048, 1, 17, 1024, 4096, 4102 }) |rows| {
        try run(rows, true, shared, 288);
        try t.expectEqual(@as(usize, 0), Record.copies);
        const calls = (rows + 4095) / 4096;
        try t.expectEqual(calls, Record.up);
        try t.expectEqual(calls, Record.down);
        try t.expectEqual(calls, Record.routes);
        try t.expectEqual(@as(usize, 0), Record.decode);
        for (0..calls) |j| {
            const n: usize = @min(rows - j * 4096, 4096);
            try t.expectEqual(@as(i32, @intCast(n * 9)), Record.pairs[j]);
            try t.expectEqual(@as(i32, 289), Record.experts[j]);
            try t.expectEqual(@as(i32, if (shared) 9 else 1), Record.divisor[j]);
            try t.expectEqual(@as(i32, 288), Record.skip[j]);
            try t.expectEqual(@as(u64, 0x400000000 + j * 4096 * 9 * 512 * 4), Record.outputs[j]);
        }
    };
}

test "GLM EXL3 prompt: decode retains compact 1024-row calls" {
    try run(2048, false, false, 288);
    try t.expectEqual(@as(usize, 4), Record.copies);
    try t.expectEqual(@as(usize, 2), Record.decode);
    try t.expectEqual(@as(usize, 0), Record.up);
    try t.expectEqual(@as(usize, 0), Record.down);
}

test "GLM EXL3 prompt: the direct shared-id route refuses an overflowing expert table before launch" {
    try t.expectError(error.Exl3Shape, run(2048, true, true, 1024));
    try t.expectEqual(@as(usize, 0), Record.routes);
    try t.expectEqual(@as(usize, 0), Record.up);
    try t.expectEqual(@as(usize, 0), Record.copies);
}

test "GLM EXL3 prompt: scratch names the rank width and rotation layout" {
    const MiB = 1024 * 1024;
    const D = 4096;
    const W_tp2 = 1024;
    const W_tp1 = 2048;
    for ([_]usize{ 2048, 4096 }, 1..) |rows, scale| {
        try t.expectEqual(@as(usize, 52 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp2, true, true)[0]);
        try t.expectEqual(@as(usize, 88 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp1, true, true)[0]);
        try t.expectEqual(@as(usize, 324 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp2, true, false)[0]);
        try t.expectEqual(@as(usize, 360 * MiB * scale), glm.exl3.scratchBytes(rows, D, W_tp1, true, false)[0]);
    }
    try t.expectEqual(glm.exl3.scratchBytes(4096, D, W_tp2, true, true), glm.exl3.scratchBytes(4102, D, W_tp2, true, true));
}

fn guarded(text: []const u8, name: []const u8, payload: []const u8) bool {
    const at = std.mem.indexOf(u8, text, name) orelse return false;
    const tail = text[at..];
    const end = std.mem.indexOf(u8, tail, "\n}") orelse return false;
    const body = tail[0..end];
    const guard = std.mem.indexOf(u8, body, "if (e == skip) return;") orelse return false;
    const load = std.mem.indexOf(u8, body, payload) orelse return false;
    const member = std.mem.indexOf(u8, body, "members[") orelse body.len;
    return guard < @min(load, member);
}

test "GLM EXL3 prompt: shared-id guards precede payload loads and late-guard controls are caught" {
    try t.expect(guarded(source, "tf_exl3_prompt_gu(", "tgateup<"));
    try t.expect(guarded(source, "tf_exl3_prompt_down(", "members[first"));
    const bad = try std.mem.replaceOwned(u8, t.allocator, source, "if (e == skip) return;", "const volatile int control_member = members[items[3 * item + 1]];\n    if (e == skip) return;");
    defer t.allocator.free(bad);
    try t.expectEqual(@as(usize, 2), std.mem.count(u8, bad, "if (e == skip) return;"));
    try t.expect(!guarded(bad, "tf_exl3_prompt_gu(", "tgateup<"));
    try t.expect(!guarded(bad, "tf_exl3_prompt_down(", "members[first"));
}
