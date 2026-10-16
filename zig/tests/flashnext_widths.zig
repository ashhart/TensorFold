//! tf-flashnext-widths: Flash Next's products at every oQ width against float64, each row's bits in every window.
const std = @import("std");
const mtl = @import("metal");
const lib = @import("flashnext_widths_lib.zig");
const lane = @import("flashnext_widths_lane.zig");
const hc = @import("flashnext_widths_hc.zig");
const moe = @import("flashnext_widths_moe.zig");
const prompt = @import("flashnext_widths_prompt.zig");
const rows = @import("flashnext_widths_rows.zig");
const fz = @import("tensorfold").flashnext_replay;

pub fn main(init: std.process.Init) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    const args = try init.minimal.args.toSlice(lib.a);
    const only: ?[]const u8 = if (args.len > 1) args[1] else null; // a class name runs that class alone
    std.debug.print("device {s}, tensor units {}\n", .{ device.name(), device.tensorUnits() });
    if (only != null and std.mem.eql(u8, only.?, "rows-time")) return rows.time(device, queue); // timing, not a check
    var t = lib.Tally{};
    if (only == null or std.mem.eql(u8, only.?, "lane")) try lane.run(&t, device, queue);
    if (only == null or std.mem.eql(u8, only.?, "rows")) try rows.run(&t, device, queue);
    const recorded = [_][]const u8{ "hc", "moe", "prompt", "rows-hc" };
    if (only == null or
        for (recorded) |c| (if (std.mem.eql(u8, only.?, c)) break true) else false)
    { // the recorded kernels the any-width ones come from
        const r = try lib.a.create(fz.Run);
        r.* = .{ .arena = lib.a, .device = device, .queue = queue, .serial = true };
        try r.compileChecked();
        if (only == null or std.mem.eql(u8, only.?, "hc")) try hc.run(&t, r);
        if (only == null or std.mem.eql(u8, only.?, "rows-hc")) try hc.runRows(&t, r);
        if (only == null or std.mem.eql(u8, only.?, "moe")) try moe.run(&t, r);
        if (only == null or std.mem.eql(u8, only.?, "prompt")) try prompt.run(&t, r);
    }
    if (t.failed > 0) {
        std.debug.print("FAIL {d} of {d} checks; the largest difference is {d:.4} of its float64 bound\n", .{ t.failed, t.checks, t.worst });
        return error.CheckFailed;
    }
    std.debug.print("PASS {d} checks: every product within {d:.4} of its float64 bound, every row's bits equal in every window\n", .{ t.checks, t.worst });
}
