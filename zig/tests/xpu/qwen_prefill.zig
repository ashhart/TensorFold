//! Chunked prompt prefill vs token-by-token decode: last-token logits and tokens/s. usage: ...-test CKPT IDS_FILE

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qw = @import("qwen_xpu").win;
const qg = @import("qwen_xpu").gguf;

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
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.Usage;
    const dir = args[1];
    const ids_text = try ld.readFile(gpa, "/", args[2][1..]);
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.tokenizeAny(u8, ids_text, ", \n");
    while (it.next()) |t| try ids.append(gpa, try std.fmt.parseInt(u32, t, 10));
    var chunks: std.ArrayList(u32) = .empty;
    var profile = false;
    var expect_exact = false;
    var nobase = false; // skip the token-by-token baseline (long prompts: timing only)
    for (args[3..]) |a| {
        if (std.mem.eql(u8, a, "profile")) profile = true else if (std.mem.eql(u8, a, "exact")) expect_exact = true else if (std.mem.eql(u8, a, "nobase")) nobase = true else try chunks.append(gpa, try std.fmt.parseInt(u32, a, 10));
    }
    if (chunks.items.len == 0) try chunks.appendSlice(gpa, &.{ 16, 64, 256, 512 });
    var big: u32 = 16;
    for (chunks.items) |c| big = @max(big, c);
    qw.default_rows = big;

    var r = try rt.open();

    defer r.deinit();
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var m = try model.Model.load(gpa, &r, &l, cfg.value, @intCast(ids.items.len + 64));
    const vocab = cfg.value.text_config.vocab_size;
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);
    const n = ids.items.len;

    const base = try gpa.alloc(f32, vocab);
    @memset(base, 1);
    const t0 = nowNs();
    for (ids.items[0 .. if (nobase) 0 else n], 0..) |tok, i| {
        try m.forward(tok, i + 1 == n);
        if (i + 1 < n) try r.sync();
    }
    if (!nobase) try m.fetchLogits(base);
    const dt0 = @as(f64, @floatFromInt(nowNs() - t0)) / 1e6;
    const base_top = top1(base);
    std.debug.print("token by token: {d} tokens {d:.0} ms ({d:.1} tokens/s), top-1 {d}\n", .{ n, dt0, @as(f64, @floatFromInt(n)) / dt0 * 1e3, base_top });

    const logits = try gpa.alloc(f32, vocab);
    const prev = try gpa.alloc(f32, vocab);
    var have_prev = false;
    for (chunks.items) |c| {
        try m.reset(zeros);
        qw.prof_on = profile;
        const t1 = nowNs();
        var i: usize = 0;
        while (i < n) {
            const rows = @min(c, n - i);
            try m.forwardRows(ids.items[i .. i + rows], if (i + rows == n) 1 else 0, true);
            i += rows;
        }
        var one: [1]i32 = undefined;
        try m.rowArgmax(&one);
        try m.fetchRowLogits(logits);
        const dt = @as(f64, @floatFromInt(nowNs() - t1)) / 1e6;
        var worst: f32 = 0;
        var mx: f32 = 0;
        for (logits, base) |a, b| {
            worst = @max(worst, @abs(a - b));
            mx = @max(mx, @abs(b));
        }
        var same_prev = false;
        if (have_prev) same_prev = std.mem.eql(u8, std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(prev));
        std.debug.print("chunk {d:>4}: {d:>7.0} ms ({d:>7.1} tokens/s), last-token logits: max|diff| {e:.2} ({e:.2} of max|logit|), top-1 {s}, identical to previous chunk size: {s}\n", .{ c, dt, @as(f64, @floatFromInt(n)) / dt * 1e3, worst, worst / mx, if (@as(u32, @intCast(one[0])) == base_top) "same" else "DIFFERENT", if (!have_prev) "-" else if (same_prev) "yes" else "no" });
        if (expect_exact and c <= 16 and worst != 0) return error.WindowsNotExact;
        if (profile) {
            m.win.printProfile(n);
            qw.prof_on = false;
        }
        @memcpy(prev, logits);
        have_prev = true;
    }
}
