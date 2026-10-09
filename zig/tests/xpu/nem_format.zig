//! Device-free format check: checkFormat accepts the MLX 4-bit Nemotron checkpoint, refuses NVFP4 / FP8-Mamba.

const std = @import("std");
const rt = @import("xpu").rt;
const ld = @import("xpu").loader;
const model = @import("nemotron_xpu").model;

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.Usage;
    const want_ok = std.mem.eql(u8, args[1], "supported");
    var bad: u32 = 0;
    for (args[2..]) |dir| {
        var r: rt.Runtime = undefined; // the loader only stores the pointer while it reads the headers
        var l = try ld.Loader.init(std.heap.page_allocator, &r, dir);
        const ok = if (model.checkFormat(&l)) true else |_| false;
        std.debug.print("{s}: {s} ({d} tensors)\n", .{ dir, if (ok) "supported" else "unsupported", l.map.count() });
        if (ok != want_ok) bad += 1;
    }
    std.debug.print("{s}\n", .{if (bad == 0) "format check ok" else "format check FAILED"});
    return if (bad == 0) 0 else 1;
}
