//! Isolated GPU entry point for the registered repetition gates; no checkpoint or native engine is loaded.
const std = @import("std");
const cuda = @import("cuda");

pub fn main(init: std.process.Init) !void {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var name: [256]u8 = undefined;
    std.debug.print("INFO repetition device: {s}, sm_{d}\n", .{ try ctx.name(&name), try ctx.capability() });
    try @import("sample_repetition_contract.zig").run(.{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io });
}
