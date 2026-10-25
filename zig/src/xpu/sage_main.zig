//! Development runner of the Sage-A1.5 XPU forward (first form): greedy tokens and timings.
const std = @import("std");
const sage = @import("sage.zig");
const now = @import("qwen27.zig").now;

const alloc = std.heap.c_allocator;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;

pub fn main(init: std.process.Init) !void {
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 3) {
        std.debug.print("usage: xpu-sage MODEL_DIR IDS_FILE [STEPS]\n", .{});
        return;
    }
    const steps: usize = if (argv.len > 3) try std.fmt.parseInt(usize, argv[3], 10) else 32;
    const ctx: u32 = if (std.c.getenv("XPU_CTX")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 4096;
    var ids: std.ArrayList(i32) = .empty;
    const path = try std.fmt.allocPrintSentinel(alloc, "{s}", .{argv[2]}, 0);
    const fd = open(path.ptr, 0);
    if (fd < 0) return error.FileNotFound;
    const size: usize = @intCast(lseek(fd, 0, 2));
    const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
    var it = std.mem.tokenizeAny(u8, text[0..size], " \n,[]");
    while (it.next()) |w| try ids.append(alloc, try std.fmt.parseInt(i32, w, 10));
    const t_load = now();
    const r = try sage.Runner.init(argv[1], ctx);
    defer r.deinit();
    std.debug.print("loaded in {d:.1} s\n", .{now() - t_load});
    const t0 = now();
    try r.prompt(ids.items);
    const t1 = now();
    var out: std.ArrayList(i32) = .empty;
    try out.append(alloc, r.last);
    for (1..steps) |_| {
        try r.step(r.last);
        try out.append(alloc, r.last);
    }
    const t2 = now();
    std.debug.print("prompt: {d} tokens in {d:.3} s ({d:.0} tok/s)\n", .{ ids.items.len, t1 - t0, @as(f64, @floatFromInt(ids.items.len)) / (t1 - t0) });
    std.debug.print("decode: {d} tokens in {d:.3} s ({d:.1} tok/s)\n", .{ steps - 1, t2 - t1, @as(f64, @floatFromInt(steps - 1)) / (t2 - t1) });
    std.debug.print("tokens:", .{});
    for (out.items) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
}
