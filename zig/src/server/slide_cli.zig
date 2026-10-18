//! `tensorfold slide rollback FOLDER`: a folder --slide taught put back as it was before its first lesson.
const std = @import("std");
const api = @import("engine_api");

const usage = "usage: tensorfold slide rollback FOLDER\n";

/// Runs `slide` with the arguments after it; the process's exit status.
pub fn main(init: std.process.Init, argv: []const [:0]const u8) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    if (argv.len == 1 and (std.mem.eql(u8, argv[0], "--help") or std.mem.eql(u8, argv[0], "-h"))) {
        try std.Io.File.stdout().writeStreamingAll(io, usage);
        return 0;
    }
    if (argv.len != 2 or !std.mem.eql(u8, argv[0], "rollback") or argv[1].len == 0 or argv[1][0] == '-') {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const dir: []const u8 = argv[1];
    var why: []const u8 = "";
    const r = api.originals.rollback(io, dir, .{}, a, &why) catch |e| {
        std.debug.print("tensorfold: {s}\n", .{if (why.len > 0) why else @errorName(e)});
        return 1;
    };
    @import("save_guard").finish(io, dir) catch |e| {
        std.debug.print("tensorfold: rollback save marker: {s}\n", .{@errorName(e)});
        return 1;
    };
    var buf: [4 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(io, &buf);
    const w = &out.interface;
    const done = "tensorfold: {s} is as it was before its first lesson: {d} files ({d:.1} GB, {d} cloned) put " ++
        "back from {s}, each matching its SHA-256, in {d:.1} s\n";
    const gb = @as(f64, @floatFromInt(r.bytes)) / 1e9;
    try w.print(done, .{ dir, r.files, gb, r.cloned, api.originals.folder_name, r.seconds });
    for (r.removed) |name| try w.print("tensorfold: removed {s}, which a lesson added\n", .{name});
    for (r.graphs) |path| try w.print("tensorfold: removed the fact graph {s}\n", .{path});
    try w.flush();
    return 0;
}
