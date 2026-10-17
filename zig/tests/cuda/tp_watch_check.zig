//! tf-tp-watch-check RANK WORLD fail|fatal|die|end: the last rank fails or dies and the watch stops every other.
const std = @import("std");
const cuda = @import("cuda");

const usage = "usage: NCCL_COMM_ID=127.0.0.1:PORT tf-tp-watch-check RANK WORLD fail|fatal|die|end\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const modes = [_][]const u8{ "fail", "fatal", "die", "end" };
    var known = false;
    if (args.len == 4) for (modes) |m| {
        known = known or std.mem.eql(u8, args[3], m);
    };
    if (!known) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const rank = try std.fmt.parseInt(usize, args[1], 10);
    const world = try std.fmt.parseInt(usize, args[2], 10);
    if (world < 2 or rank >= world) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const mode = args[3];
    try cuda.tp_watch.start(init.io, rank, world);
    std.debug.print("rank {d} of {d}: lines open, {s}\n", .{ rank, world, mode });
    const io = init.io;
    if (std.mem.eql(u8, mode, "end")) { // every rank ends on purpose, so no line closing stops one
        cuda.tp_watch.ended();
        std.Io.sleep(io, .fromMilliseconds(3000), .awake) catch {};
        return 0;
    }
    if (rank == world - 1) {
        std.Io.sleep(io, .fromMilliseconds(500), .awake) catch {};
        if (std.mem.eql(u8, mode, "fail")) {
            cuda.tp_watch.fail("injected failure");
            std.c._exit(3);
        }
        if (std.mem.eql(u8, mode, "fatal")) cuda.tp_watch.fatal("injected fatal failure");
        std.c._exit(4); // die: no record, the line just closes
    }
    std.Io.sleep(io, .fromMilliseconds(60_000), .awake) catch {}; // the watch ends this process first (exit 1)
    std.debug.print("rank {d}: still running after 60 s: no watch stopped it\n", .{rank});
    return 9;
}
