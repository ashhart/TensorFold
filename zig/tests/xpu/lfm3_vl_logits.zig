//! LFM3-VL's logits with a picture in context: every row of a teacher-forced prompt, for #1502's compare_logits.py.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rig.zig");
const lfm3 = @import("lfm3_xpu");
const img = @import("lfm3_image").image;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
pub fn main(init: std.process.Init) !void {
    xpu.stop.install();
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 5) {
        std.debug.print("skipped: usage tf-xpu-lfm3_vl_logits-test MODEL_DIR TOKENS_JSON IMAGE OUT_DIR [PATCHES_F32]\n", .{});
        return;
    }
    const doc = try std.json.parseFromSlice(std.json.Value, gpa, try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .unlimited), .{});
    const vals = doc.value.object.get("tokens").?.array.items;
    const at: u32 = @intCast(doc.value.object.get("image_start").?.integer);
    const ids = try gpa.alloc(u32, vals.len);
    for (vals, ids) |v, *o| o.* = @intCast(v.integer);
    const p = try img.prepare(io, gpa, try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(64 << 20)), 4096);
    // PATCHES_F32: the tower's input from a file (fp32 [patches][1536], the picture's grid), e.g. a receipt's patches
    if (args.len > 5) {
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, args[5], gpa, .limited(256 << 20));
        if (raw.len != p.pixels.len * 4) return error.PatchesDiffer;
        @memcpy(std.mem.sliceAsBytes(p.pixels), raw);
    }
    for (ids[at .. at + p.tokens]) |t| if (t != 125030) return error.PromptRowsDiffer;
    if (ids[at + p.tokens] == 125030) return error.PromptRowsDiffer;
    var r = try rt.open();
    defer r.deinit();
    var c = try lfm3.config.Config.read(gpa, io, args[1]);
    defer c.deinit();
    var l = try xpu.loader.Loader.init(gpa, &r, args[1]);
    var m = try lfm3.model.Model.load(gpa, &r, &l, &c, 4096, 128);
    const tower = try lfm3.vision.Tower.load(gpa, &r, &l, p.gh * p.gw);
    const rows = try l.empty(@as(usize, p.tokens) * lfm3.vision.out_width * 2);
    try tower.image(p.pixels, p.gh, p.gw, rows, null);
    const images = [_]lfm3.model.Image{.{ .rows = rows, .at = at, .n = p.tokens }};
    m.images = &images;
    try m.reset();
    // teacher-forced: the prompt in windows of up to 16 rows, each row's logits (verified windows' path)
    const V = lfm3.config.vocab;
    const logits = try gpa.alloc(f32, ids.len * V);
    var i: usize = 0;
    while (i < ids.len) {
        const n = @min(ids.len - i, lfm3.model.window_rows);
        try m.forwardSegs(&.{.{ .q = m.seq, .ids = ids[i .. i + n] }}, .rows, false, null);
        try m.rowLogits(logits[i * V .. (i + n) * V]);
        i += n;
    }
    var b: [256]u8 = undefined;
    const out = args[4];
    // logits as bf16 (round to nearest even), the form compare_logits.py reads
    const lb = try gpa.alloc(u16, logits.len);
    for (lb, logits) |*o, v| {
        var u: u32 = @bitCast(v);
        u +%= 0x7fff + ((u >> 16) & 1);
        o.* = @truncate(u >> 16);
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&b, "{s}/xpu_logits.bf16", .{out}), .data = std.mem.sliceAsBytes(lb) });
    const host = try gpa.alloc(u16, @as(usize, p.tokens) * lfm3.vision.out_width);
    try r.sync();
    try r.download(std.mem.sliceAsBytes(host), rows);
    try r.sync();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&b, "{s}/xpu_rows.bf16", .{out}), .data = std.mem.sliceAsBytes(host) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&b, "{s}/pixels.f32", .{out}), .data = std.mem.sliceAsBytes(p.pixels) });
    std.debug.print("{d} x {d} patches, {d} image rows at {d}; {d} teacher-forced rows of logits in {s}\n", .{ p.gh, p.gw, p.tokens, at, ids.len, out });
}
