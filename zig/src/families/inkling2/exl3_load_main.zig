//! A loader-only canary: no serving listener, forward or drafter.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("exl3_config.zig");
const weights = @import("weights.zig");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("usage: tf-inkling2-exl3-load DIRECTORY RANK WORLD LAYERS\n", .{});
        return 2;
    }
    const rank = try std.fmt.parseInt(usize, args[2], 10);
    const world = try std.fmt.parseInt(usize, args[3], 10);
    const count = try std.fmt.parseInt(usize, args[4], 10);
    const share = try cfg.Share.init(rank, world);
    var driver = try cuda.Driver.open();
    defer driver.close();
    var context = try cuda.Context.init(&driver, 0);
    defer context.deinit();
    var loaded = try weights.load(init.gpa, init.io, &driver, args[1], share, count);
    defer loaded.deinit();
    std.debug.print("LOADED rank {d}/{d}: {d} layers, {d} owned device bytes\n", .{ rank, world, loaded.layer.len, loaded.bytes });
    return 0;
}
