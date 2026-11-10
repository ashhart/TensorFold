//! Focused entry for the mode-6 replacement contract; compiling it does not open a GPU.
const std = @import("std");
const cuda = @import("cuda");
pub fn main(init: std.process.Init) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    try @import("contract_attention_tuned.zig").run(
        .{ .d = &driver, .ctx = &context, .gpa = init.gpa, .io = init.io },
        cuda.kernels.prefill_attention,
    );
    return 0;
}
