//! LFM3-VL on one picture without the server: tower, image rows spliced into the prompt, prefill, greedy reply (twice).

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rig.zig");
const lfm3 = @import("lfm3_xpu");
const img = @import("lfm3_image").image;

fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) / 1e6;
}

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    xpu.stop.install();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) {
        std.debug.print("skipped: usage tf-xpu-lfm3_vl-test MODEL_DIR PROMPT_JSON IMAGE MAX_TOKENS [REPLY_TOKENS] [OUT_JSON]\n", .{});
        return;
    }
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .unlimited);
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, bytes, .{});
    const vals = doc.value.object.get("ids").?.array.items;
    const at: u32 = @intCast(doc.value.object.get("at").?.integer);
    const p = try img.prepare(io, gpa, try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(64 << 20)), try std.fmt.parseInt(u32, args[4], 10));
    // the template's one image token, expanded to the picture's rows
    const ids = try gpa.alloc(u32, vals.len - 1 + p.tokens);
    for (ids, 0..) |*o, i| o.* = if (i < at) @intCast(vals[i].integer) else if (i < at + p.tokens) 125030 else @intCast(vals[i + 1 - p.tokens].integer);
    const reply_len: usize = if (args.len > 5) try std.fmt.parseInt(usize, args[5], 10) else 256;
    if (at + p.tokens > ids.len or ids[at] != 125030 or ids[at + p.tokens - 1] != 125030) return error.PromptRowsDiffer;
    var r = try rt.open();
    defer r.deinit();
    var c = try lfm3.config.Config.read(gpa, io, args[1]);
    defer c.deinit();
    var l = try xpu.loader.Loader.init(gpa, &r, args[1]);
    var m = try lfm3.model.Model.load(gpa, &r, &l, &c, 8192, 128);
    const P = p.gh * p.gw;
    const tower = try lfm3.vision.Tower.load(gpa, &r, &l, P);
    const rows = try l.empty(@as(usize, p.tokens) * lfm3.vision.out_width * 2);
    std.debug.print("{d} x {d} patches, {d} image rows at prompt position {d} of {d}; {d} MiB on the device\n", .{ p.gh, p.gw, p.tokens, at, ids.len, l.total >> 20 });
    var first: []u32 = &.{};
    var split: [2]f64 = undefined;
    tower.split = &split;
    for (0..3) |run| {
        try r.sync();
        const t0 = now();
        try tower.image(p.pixels, p.gh, p.gw, rows, null);
        try r.sync();
        const t1 = now();
        const images = [_]lfm3.model.Image{.{ .rows = rows, .at = at, .n = p.tokens }};
        m.images = &images;
        try m.reset();
        try m.prefill(ids, null);
        m.images = &.{};
        var out = try gpa.alloc(u32, reply_len);
        out[0] = try m.argmax();
        const t2 = now();
        var n: usize = 1;
        while (n < reply_len and out[n - 1] != 124900) : (n += 1) {
            try m.forward(out[n - 1 .. n], true, null);
            out[n] = try m.argmax();
        }
        const t3 = now();
        out = out[0..n];
        std.debug.print("run {d}: tower {d:.0} ms, projector {d:.0} ms (both {d:.0}), text prefill {d:.0} ms, first token {d:.0} ms, {d} tokens at {d:.1} tok/s\n", .{ run + 1, split[0], split[1], t1 - t0, t2 - t1, t2 - t0, n, @as(f64, @floatFromInt(n - 1)) / ((t3 - t2) / 1e3) });
        if (run == 0) first = out else std.debug.print("  reply {s} to run 1\n", .{if (std.mem.eql(u32, first, out)) "identical" else "DIFFERENT"});
    }
    std.debug.print("device allocations at the end: {d} MiB (Level Zero, this process)\n", .{xpu.rt.alloc_now >> 20});
    var w: std.Io.Writer.Allocating = .init(gpa);
    try std.json.Stringify.value(.{ .reply = first }, .{}, &w.writer);
    if (args.len > 6) try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[6], .data = w.written() });
    std.debug.print("{s}\n", .{w.written()});
}
