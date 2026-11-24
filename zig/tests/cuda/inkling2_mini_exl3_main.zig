//! Run the Mini EXL3 checks with the serving images and without other families' contract fixtures.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const gates = @import("core").contract.gates;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2 and args.len != 4) return error.BadArguments;
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-exl3-contract"))
        return @import("contract_inkling2_mini_exl3.zig").run(gpu);
    if (args.len == 2 and std.mem.eql(u8, args[1], "inkling2-mini-regressions"))
        return regressions(gpu);
    if (args.len == 4 and std.mem.eql(u8, args[1], "inkling2-mini-exl3-teacher"))
        return @import("inkling2_mini_exl3_teacher.zig").run(gpu, args[2], args[3]);
    return error.BadArguments;
}

fn regressions(gpu: check.Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels: std.ArrayList(gates.Kernel) = .empty;
    defer kernels.deinit(gpu.gpa);
    const set = try @import("contract_inkling2.zig").Set.init(gpu, stream, gpu.gpa, &kernels);
    defer set.deinit();
    @import("contract_after.zig").keep(&kernels, &.{
        "inkling2 Mini NVFP4 decode",           "inkling2 Mini MXFP8 decode",
        "inkling2 loop tf_ink2_loop_gate_down", "inkling2 loop rank-64 gate",
    });
    var tally: gates.Tally = .{};
    try gates.runAll(gpu.gpa, kernels.items, 0x6b63, &tally);
    std.debug.print("MINI REGRESSIONS {d}/{d} kernels, {d}/{d} controls\n", .{
        tally.passed, tally.kernels, tally.caught, tally.controls,
    });
    try check.expect(tally.kernels == 3 and tally.controls == 3 and tally.problems() == 0, "Mini regression contract", .{});
}
