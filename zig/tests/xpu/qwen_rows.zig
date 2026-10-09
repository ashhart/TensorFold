//! Row invariance of the multi-row window forward on the whole model vs one token at a time. usage: ...-test CKPT [N]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qg = @import("qwen_xpu").gguf;

const gpa = std.heap.page_allocator;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Counts rows whose logits differ bitwise from the baseline row and the worst absolute difference.
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
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    const n: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 40;
    var r = try rt.open();
    defer r.deinit();
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var m = try model.Model.load(gpa, &r, &l, cfg.value, 512);
    const vocab = cfg.value.text_config.vocab_size;
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);

    // a realistic token sequence: the greedy continuation of a short prompt
    const tokens = try gpa.alloc(u32, n);
    const prompt = [_]u32{ 760, 6511, 314, 9338, 369 };
    @memcpy(tokens[0..prompt.len], &prompt);
    const base = try gpa.alloc(f32, n * vocab);
    var next: u32 = 0;
    for (0..n) |t| {
        if (t >= prompt.len) tokens[t] = next;
        try m.forward(tokens[t], true);
        next = try m.argmax();
        try m.fetchLogits(base[t * vocab ..][0..vocab]);
    }
    std.debug.print("baseline: {d} tokens one at a time\n", .{n});

    const logits = try gpa.alloc(f32, 16 * vocab);
    const commit_pattern = [_]u32{ 8, 3, 1, 5, 8, 2, 7, 4 };
    var failures: u32 = 0;
    for ([_]u32{ 2, 3, 4, 8, 16, 100 }) |mode| {
        // mode < 100: in-place windows of `mode` rows; 100: R = 8 windows, state pending then committed
        try m.reset(zeros);
        var i: usize = 0;
        var win: usize = 0;
        var bad_rows: u32 = 0;
        var rows_seen: u32 = 0;
        var worst: f32 = 0;
        const t0 = nowNs();
        while (i < n) : (win += 1) {
            const want_rows: usize = if (mode == 100) 8 else mode;
            const rows = @min(want_rows, n - i);
            try m.forwardRows(tokens[i .. i + rows], @intCast(rows), mode != 100);
            try m.fetchRowLogits(logits);
            for (0..rows) |k| {
                rows_seen += 1;
                if (!compareRow(logits[k * vocab ..][0..vocab], base[(i + k) * vocab ..][0..vocab], &worst)) bad_rows += 1;
            }
            var c: usize = rows;
            if (mode == 100) {
                c = @min(commit_pattern[win % commit_pattern.len], rows);
                try m.commitRows(@intCast(c));
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
