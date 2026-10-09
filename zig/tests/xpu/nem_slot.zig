//! Regression: forwards queued back to back without sync give the bits of the same forwards synced one by one.
const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;
const ld = @import("xpu").loader;
const cfgm = @import("nemotron_xpu").cfg;
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);

const gpa = std.heap.page_allocator;
const toks = [_]u32{ 1784, 8961, 1307, 5498, 1395, 3149, 9111, 87539, 4990, 4244, 23322, 6610, 1261, 2142, 1500, 2500, 3500, 4500, 5500, 6500, 7500, 8500, 9500, 10500 };

fn same(a: []const f32, b: []const f32) bool {
    for (a, b) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) return false;
    return true;
}

pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    const cfg = try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    var m = try model.Model.load(gpa, &r, &l, cfg.value, 256);
    var w = try nw.Win.init(gpa, &m);
    const vocab = cfg.value.vocab_size;
    const a = try gpa.alloc(f32, vocab);
    const b = try gpa.alloc(f32, vocab);
    var ok = true;

    // Model.forward: queued vs synced
    for ([_]bool{ true, false }) |queued| {
        try w.reset(&m);
        for (toks, 0..) |t, i| {
            try m.forward(t, i + 1 == toks.len);
            if (!queued) try r.sync();
        }
        try m.fetchLogits(if (queued) a else b);
    }
    const plain_same = same(a, b);
    std.debug.print("Model.forward, {d} tokens queued without a sync vs synced: {s}\n", .{ toks.len, if (plain_same) "identical" else "DIFFERENT" });
    ok = ok and plain_same;

    // Win.forward: one-row windows from a slot that is clobbered right after the call
    for ([_]bool{ true, false }) |queued| {
        try w.reset(&m);
        for (toks, 0..) |t, i| {
            var slot = [1]u32{t};
            try w.forward(&m, &slot, true, if (i + 1 == toks.len) 1 else 0);
            slot[0] = 0; // a late read of the caller's slice would see this
            if (!queued) try r.sync();
        }
        try r.sync();
        try w.fetchRowLogits(&m, if (queued) a else b);
    }
    const win_same = same(a, b);
    std.debug.print("Win.forward, {d} one-row windows from a clobbered slot vs synced: {s}\n", .{ toks.len, if (win_same) "identical" else "DIFFERENT" });
    ok = ok and win_same;
    if (!ok) return error.Mismatch;
    std.debug.print("slot ok\n", .{});
}
