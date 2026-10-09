//! Regression: queued single-token forwards without a sync match synced ones. usage: ...-test CKPT [N_TOKENS]
const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cfgm = @import("qwen_xpu").config;
const model = @import("qwen_xpu").model;
const qg = @import("qwen_xpu").gguf;

const gpa = std.heap.page_allocator;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) return error.Usage;
    const dir = args[1];
    const n: usize = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 14;
    var r = try rt.open();
    defer r.deinit();
    const is_gguf = std.mem.endsWith(u8, dir, ".gguf");
    var l = if (is_gguf) try ld.Loader.initBare(gpa, &r) else try ld.Loader.init(gpa, &r, dir);
    const cfg: std.json.Parsed(cfgm.Config) = if (is_gguf) blk: {
        const meta = try qg.readHeader(gpa, &l, dir);
        break :blk .{ .arena = undefined, .value = try qg.config(gpa, &l, meta) };
    } else try cfgm.parse(gpa, try ld.readFile(gpa, dir, "config.json"));
    var m = try model.Model.load(gpa, &r, &l, cfg.value, 256);
    const vocab = cfg.value.text_config.vocab_size;
    const zeros = try gpa.alloc(u8, model.state_bytes);
    @memset(zeros, 0);
    const toks = [_]u32{ 760, 6511, 314, 9338, 369, 1784, 8961, 1307, 5498, 1395, 3149, 9111, 4990, 4244, 23322, 6610, 1261, 2142 };
    if (n > toks.len) return error.Usage;
    const a = try gpa.alloc(f32, vocab);
    const b = try gpa.alloc(f32, vocab);
    var ok = true;
    for ([_]bool{ true, false }) |queued| {
        try m.reset(zeros);
        for (0..n) |t| {
            try m.forward(toks[t], t + 1 == n);
            if (!queued) try r.sync();
        }
        try r.sync();
        try m.fetchLogits(if (queued) a else b);
        try r.sync();
    }
    var diff: usize = 0;
    for (a, b) |x, y| {
        if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) diff += 1;
    }
    if (diff != 0) ok = false;
    std.debug.print("{d} forwards queued without a sync vs synced one by one: {d} of {d} logits differ -> {s}\n", .{ n, diff, vocab, if (ok) "ok" else "FAIL" });
    return if (ok) 0 else 1;
}
