//! tf-merge-check --quantizer DENSE SHIPPED MODULE...: the quantizer against shipped tensors from their dense source.
const std = @import("std");
const tools = @import("lora_merge");
const Io = std.Io;

const usage = "usage: tf-merge-check --quantizer DENSE SHIPPED MODULE...\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var out_buf: [4096]u8 = undefined;
    var out = Io.File.stdout().writerStreaming(init.io, &out_buf);
    defer out.interface.flush() catch {};
    const w = &out.interface;
    if (args.len < 5 or !std.mem.eql(u8, args[1], "--quantizer")) {
        try w.writeAll(usage);
        return 2;
    }
    const ok = tools.gate.quantizerCheck(init.gpa, init.arena.allocator(), init.io, w, args[2], args[3], @ptrCast(args[4..])) catch |e| {
        if (e != error.Refused) try w.print("tf-merge-check: {s}\n", .{@errorName(e)});
        return 1;
    };
    return if (ok) 0 else 1;
}
