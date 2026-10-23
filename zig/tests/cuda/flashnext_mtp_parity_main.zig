//! `tf-flashnext-mtp-parity <model dir> <ids file>`: several streams' MTP heads in one shared pass against each alone.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const tests = @import("flashnext_mtp_parity_tests.zig");

const usage =
    \\usage: tf-flashnext-mtp-parity <model dir> <ids file>
    \\  several streams' greedy MTP heads in one shared pass against each stream's own pass, bit for bit
    \\
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io };
    tests.run(gpu, args[1], args[2]) catch |err| {
        std.debug.print("FAIL flashnext mtp parity: {t}\n", .{err});
        return 1;
    };
    return 0;
}
