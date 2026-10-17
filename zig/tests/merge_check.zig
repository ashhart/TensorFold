//! tf-merge-check DENSE ADAPTER OUT SHIPPED, or --quantizer DENSE SHIPPED MODULE...: the gates of `merge --quantize`.
const std = @import("std");
const merge = @import("lora_merge");
const Io = std.Io;

const usage = "usage: tf-merge-check DENSE ADAPTER OUT SHIPPED | tf-merge-check --quantizer DENSE SHIPPED MODULE...\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var out_buf: [4096]u8 = undefined;
    var out = Io.File.stdout().writerStreaming(init.io, &out_buf);
    defer out.interface.flush() catch {};
    const w = &out.interface;
    const gpa = init.gpa;
    const arena = init.arena.allocator();
    const run = if (args.len == 5 and !std.mem.eql(u8, args[1], "--quantizer"))
        merge.gate.check(gpa, arena, init.io, w, args[1], args[2], args[3], args[4])
    else if (args.len >= 5 and std.mem.eql(u8, args[1], "--quantizer"))
        merge.gate.quantizerCheck(gpa, arena, init.io, w, args[2], args[3], @ptrCast(args[4..]))
    else {
        try w.writeAll(usage);
        return 2;
    };
    const ok = run catch |e| {
        if (e != error.Refused) try w.print("tf-merge-check: {s}\n", .{@errorName(e)});
        return 1;
    };
    return if (ok) 0 else 1;
}
