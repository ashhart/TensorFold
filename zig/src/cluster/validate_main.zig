//! Host-only `tensorfold config validate`: the same argv as the product CLI, no GPU, no CUDA, no probes.
const std = @import("std");
const validate = @import("validate.zig");

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    return validate.main(init, argv[1..]);
}
