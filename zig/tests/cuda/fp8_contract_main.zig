const std = @import("std");
const cuda = @import("cuda");

pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    try @import("contract_fp8_prompt.zig").run(.{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io });
    try @import("contract_fp8_shared.zig").run(.{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io });
    return 0;
}
