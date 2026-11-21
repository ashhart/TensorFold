//! tf-cluster: the cluster commands alone, the same as `tensorfold cluster ...`, `serve --cluster` and `node`.
const std = @import("std");
const cluster = @import("cluster");

pub fn main(init: std.process.Init) !u8 {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (cluster.validate.wants(@ptrCast(argv[1..]))) return cluster.validate.main(init, argv[1..]);
    return cluster.cli.main(init, argv[1..]);
}
