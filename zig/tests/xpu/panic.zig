//! Exit-path test (no model): holds device memory, ends via panic/error/ok/deinit-*; must exit within 10 s.
const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");

pub const panic = std.debug.FullPanic(rt.panicDrain);

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const mode = if (args.len > 1) args[1] else "ok";
    var r = try rt.open();
    const big = try r.alloc(1 << 30);
    _ = big;
    const host = try init.gpa.alloc(u8, 64 << 20);
    defer if (std.mem.startsWith(u8, mode, "deinit")) r.deinit();
    // host stays allocated: queued uploads read it at execution time
    @memset(host, 7);
    const dst = try r.alloc(host.len);
    for (0..32) |_| try r.upload(dst, host);
    std.debug.print("holding {d:.2} GB, mode {s}\n", .{ @as(f64, @floatFromInt(@import("xpu").rt.alloc_now)) / 1e9, mode });
    if (std.mem.eql(u8, mode, "panic")) @panic("deliberate panic with device memory held");
    if (std.mem.endsWith(u8, mode, "error")) return error.Deliberate;
    return 0;
}
