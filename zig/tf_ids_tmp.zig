//! TEMPORARY local aid (not part of any patch): text file -> token ids JSON for tf-dsv41-run.
//! usage: tf-ids TOKENIZER_JSON TEXT_FILE
const std = @import("std");
const tok = @import("tokenizer");

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 3) {
        std.debug.print("usage: tf-ids TOKENIZER_JSON TEXT_FILE\n", .{});
        return error.Usage;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], a, .limited(1 << 20));
    const t = try tok.loadTokenizer(io, a, args[1]);
    const ids = try t.encode(a, text);
    std.debug.print("[", .{});
    for (ids, 0..) |id, i| {
        if (i != 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{id});
    }
    std.debug.print("]\n", .{});
}
