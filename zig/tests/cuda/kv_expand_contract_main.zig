//! Focused native KV-expansion contract; compiling this entry does not open a GPU.
const std = @import("std");
const cuda = @import("cuda");

pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    try @import("contract_kv_expand.zig").run(.{
        .d = &driver,
        .ctx = &context,
        .gpa = init.gpa,
        .io = init.io,
    });
    return 0;
}
