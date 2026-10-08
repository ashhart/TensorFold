//! Row invariance of the Nemotron-H window forward: fp32 logits of each row of R-token windows vs the tokens singly.

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

fn compareRow(got: []const f32, want: []const f32, worst: *f32) bool {
    var same = true;
    for (got, want) |g, w| {
        if (@as(u32, @bitCast(g)) != @as(u32, @bitCast(w))) {
            same = false;
            worst.* = @max(worst.*, @abs(g - w));
        }
    }
    return same;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    stop.install();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    const n: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 40;
    const cfg = try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var r = try rt.open();
    defer r.deinit();
    var l = try ld.Loader.init(gpa, &r, dir);
    var m = try model.Model.load(gpa, &r, &l, cfg.value, 512);
    var w = try nw.Win.init(gpa, &m);
    const vocab = cfg.value.vocab_size;
    if (args.len > 3 and std.mem.eql(u8, args[3], "--cost")) { // time of a window of R rows (verify-window shape: logits for every row) vs one plain step
        try w.reset(&m);
        var toks: [16]u32 = undefined;
        for (&toks, 0..) |*x, j| x.* = 1000 + @as(u32, @intCast(j)) * 37;
        for (0..4) |_| try m.forward(1234, true);
        try r.sync();
        var best: u64 = std.math.maxInt(u64);
        for (0..10) |_| {
            m.pos = 8;
            const t0 = nowNs();
            try m.forward(1234, true);
            _ = try m.argmax();
            best = @min(best, nowNs() - t0);
        }
        const plain_ns = best;
        std.debug.print("plain step: {d:.2} ms\n", .{@as(f64, @floatFromInt(best)) / 1e6});
        for ([_]u32{ 1, 2, 4, 8, 16 }) |R| {
            best = std.math.maxInt(u64);
            for (0..10) |_| {
                m.pos = 8;
                const t0 = nowNs();
                try w.forward(&m, toks[0..R], false, R);
                var out: [16]i32 = undefined;
                try w.rowArgmax(&m, &out);
                best = @min(best, nowNs() - t0);
            }
            std.debug.print("window R={d:>2}: {d:.2} ms ({d:.2}x a plain step)\n", .{ R, @as(f64, @floatFromInt(best)) / 1e6, @as(f64, @floatFromInt(best)) / @as(f64, @floatFromInt(plain_ns)) });
        }
        if (std.c.getenv("NEM_PROF") != null) {
            for ([_]u32{ 1, 8 }) |R| {
                nw.prof_on = true;
                m.pos = 8;
                try w.forward(&m, toks[0..R], false, R);
                try r.sync();
                nw.prof_on = false;
                std.debug.print("profile of one window, R={d}:\n", .{R});
                nw.printProf(&w, &m, 1);
                try w.profClear();
            }
        }
        return;
    }
    if (args.len > 3 and std.mem.eql(u8, args[3], "--tiny")) { // two rows, in place, NEM_LAYERS layers: no baseline (a first run of new kernels)
        try w.reset(&m);
        try w.forward(&m, &[_]u32{ 1, 3087 }, true, 2);
        try r.sync();
        const lg = try gpa.alloc(f32, 2 * vocab);
        try w.fetchRowLogits(&m, lg);
        std.debug.print("tiny window ok: logits row 0 [0..3] {d:.3} {d:.3} {d:.3} {d:.3}\n", .{ lg[0], lg[1], lg[2], lg[3] });
        return;
    }
    // a realistic token sequence: the greedy continuation of a short prompt, one token at a time
    const tokens = try gpa.alloc(u32, n);
    const prompt = [_]u32{ 1, 3087, 1044, 1032, 7456 };
    @memcpy(tokens[0..prompt.len], &prompt);
    const base = try gpa.alloc(f32, n * vocab);
    try w.reset(&m);
    var next: u32 = 0;
    for (0..n) |t| {
        try nw.stopCheck(&r);
        if (t >= prompt.len) tokens[t] = next;
        try w.forward(&m, tokens[t .. t + 1], true, 1); // the one-token path is a window of one row
        var one: [1]i32 = undefined;
        try w.rowArgmax(&m, &one);
        next = @intCast(one[0]);
        try w.fetchRowLogits(&m, base[t * vocab ..][0..vocab]);
    }
    std.debug.print("baseline: {d} tokens one at a time (windows of one row)\n", .{n});
    { // and the single-row kernels of model.zig: the difference of the logits is the rounding of the two dot-product orders
        try w.reset(&m);
        const lg = try gpa.alloc(f32, vocab);
        var worst: f32 = 0;
        var mx: f32 = 0;
        for (0..n) |t| {
            try m.forward(tokens[t], true);
            try m.fetchLogits(lg);
            for (lg, base[t * vocab ..][0..vocab]) |a, b| {
                worst = @max(worst, @abs(a - b));
                mx = @max(mx, @abs(b));
            }
        }
        std.debug.print("window-of-one vs model.zig single-row kernels: max |logit diff| {e:.2} ({e:.2} of max |logit|)\n", .{ worst, worst / mx });
    }
    const logits = try gpa.alloc(f32, 16 * vocab);
    const commit_pattern = [_]u32{ 8, 3, 1, 5, 8, 2, 7, 4 };
    var failures: u32 = 0;
    for ([_]u32{ 2, 3, 4, 8, 16, 100 }) |mode| {
        try w.reset(&m);
        var i: usize = 0;
        var win: usize = 0;
        var bad_rows: u32 = 0;
        var rows_seen: u32 = 0;
        var worst: f32 = 0;
        const t0 = nowNs();
        while (i < n) : (win += 1) {
            try nw.stopCheck(&r);
            const rows: usize = @min(if (mode == 100) @as(usize, 8) else mode, n - i);
            try w.forward(&m, tokens[i .. i + rows], mode != 100, @intCast(rows));
            try w.fetchRowLogits(&m, logits);
            for (0..rows) |k| {
                rows_seen += 1;
                if (!compareRow(logits[k * vocab ..][0..vocab], base[(i + k) * vocab ..][0..vocab], &worst)) bad_rows += 1;
            }
            var c: usize = rows;
            if (mode == 100) {
                c = @min(commit_pattern[win % commit_pattern.len], rows);
                try w.commit(&m, @intCast(c));
            }
            i += c;
        }
        try r.sync();
        const dt = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
        std.debug.print("{s} R={d}: {d} rows compared, {d} differ bitwise (worst abs {e:.2}), {d:.1} ms total\n", .{ if (mode == 100) "pending+commit" else "inplace       ", if (mode == 100) @as(u32, 8) else mode, rows_seen, bad_rows, worst, dt });
        if (bad_rows != 0) failures += 1;
    }
    if (failures != 0) return error.RowsDiffer;
    std.debug.print("rows ok\n", .{});
}
