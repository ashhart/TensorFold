//! Nemotron-H prefill in 32..512-row chunks vs one row at a time: logits diff, top-1, chunk invariance, tokens/s.

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const stop = @import("xpu").stop;
const ld = @import("xpu").loader;
const cfgm = @import("nemotron_xpu").cfg;
const model = @import("nemotron_xpu").model;
const nw = @import("nemotron_xpu").win;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn top1(v: []const f32) u32 {
    var b: u32 = 0;
    for (v, 0..) |x, i| if (x > v[b]) {
        b = @intCast(i);
    };
    return b;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    const n: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 1024;
    var chunks: std.ArrayList(u32) = .empty;
    for (args[@min(args.len, 3)..]) |a| try chunks.append(gpa, try std.fmt.parseInt(u32, a, 10));
    if (chunks.items.len == 0) try chunks.appendSlice(gpa, &.{ 32, 64, 256, 512 });
    var big: u32 = 16;
    for (chunks.items) |c| big = @max(big, c);
    nw.default_rows = big;
    const cfg = try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    var m = try model.Model.load(gpa, &r, &l, cfg.value, @intCast(n + 64));
    m.bf16_logits = true;
    var w = try nw.Win.init(gpa, &m);
    const vocab = cfg.value.vocab_size;
    // a deterministic pseudo-text of valid token ids
    const ids = try gpa.alloc(u32, n);
    var prng = std.Random.DefaultPrng.init(42);
    for (ids) |*x| x.* = 1000 + prng.random().uintLessThan(u32, 60000);
    const base = try gpa.alloc(f32, vocab);
    try w.reset(&m);
    const t0 = nowNs();
    for (0..n) |i| {
        try nw.stopCheck(&r);
        try w.forward(&m, ids[i .. i + 1], true, if (i + 1 == n) 1 else 0);
        if (i % 64 == 63) try r.sync();
    }
    try w.fetchRowLogits(&m, base);
    const dt0 = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
    const base_top = top1(base);
    std.debug.print("row by row: {d} tokens {d:.0} ms ({d:.1} tokens/s), top-1 {d}\n", .{ n, dt0, @as(f64, @floatFromInt(n)) / dt0 * 1e3, base_top });
    const logits = try gpa.alloc(f32, vocab);
    const prev = try gpa.alloc(f32, vocab);
    var have_prev = false;
    for (chunks.items) |c| {
        try w.reset(&m);
        const t1 = nowNs();
        var i: usize = 0;
        while (i < n) {
            try nw.stopCheck(&r);
            const rows = @min(c, n - i);
            try w.forward(&m, ids[i .. i + rows], true, if (i + rows == n) 1 else 0);
            i += rows;
        }
        try w.fetchRowLogits(&m, logits);
        const dt = @as(f64, @floatFromInt(nowNs() - t1)) / 1e6;
        var worst: f32 = 0;
        var mx: f32 = 0;
        for (logits, base) |a, b| {
            worst = @max(worst, @abs(a - b));
            mx = @max(mx, @abs(b));
        }
        const same_prev = have_prev and std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(prev));
        std.debug.print("chunk {d:>4}: {d:>7.0} ms ({d:>7.1} tokens/s), last-token logits: max|diff| {e:.2} ({e:.2} of max|logit|), top-1 {s}, identical to previous chunk size: {s}\n", .{ c, dt, @as(f64, @floatFromInt(n)) / dt * 1e3, worst, worst / mx, if (top1(logits) == base_top) "same" else "DIFFERENT", if (!have_prev) "-" else if (same_prev) "yes" else "no" });
        @memcpy(prev, logits);
        have_prev = true;
        if (std.c.getenv("NEM_PROF") != null) { // one more window of c rows at the end of the context with every launch timed (sync round trip included)
            try w.reset(&m);
            nw.prof_on = true;
            @import("nemotron_xpu").pf.prof_on = true;
            try w.forward(&m, ids[0..c], true, 0);
            try r.sync();
            nw.prof_on = false;
            @import("nemotron_xpu").pf.prof_on = false;
            std.debug.print("profile of one window of {d} rows:\n", .{c});
            nw.printProf(&w, &m, 1);
            @import("nemotron_xpu").pf.printProf(&w.mpf.?);
            try w.profClear();
        }
    }
}
