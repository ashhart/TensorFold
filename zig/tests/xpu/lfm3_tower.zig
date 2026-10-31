//! LFM3-VL's tower and projector on one picture: patches, every layer's rows and the image rows, for vision_check.py.

const std = @import("std");
const xpu = @import("xpu");
const rt = @import("rig.zig");
const lfm3 = @import("lfm3_xpu");
const img = @import("lfm3_image").image;
const Tower = lfm3.vision.Tower;

/// #1502's synthetic 300 x 217 image test picture ("match the processor float32 pixel fixture"): 14 x 20 patches.
fn synthetic(a: std.mem.Allocator) !img.Prepared {
    const rgb = try a.alloc(u8, 217 * 300 * 3);
    defer a.free(rgb);
    for (0..217) |y| for (0..300) |x| for (0..3) |c| {
        rgb[(y * 300 + x) * 3 + c] = @intCast((x * 7 + y * 13 + c * 50 + (x * y) % 31) % 256);
    };
    return img.prepareRgb(a, .{ .pixels = rgb, .width = 300, .height = 217 }, 0, 1 << 20);
}

const Dump = struct {
    io: std.Io,
    dir: []const u8,
    prefix: []const u8,
    fn take(ctx: *anyopaque, stage: i32, rows: []const u16, cols: u32) void {
        _ = cols;
        const d: *Dump = @ptrCast(@alignCast(ctx));
        var b: [256]u8 = undefined;
        const name = if (stage < 0)
            std.fmt.bufPrint(&b, "{s}/{s}embedding_norm.bf16", .{ d.dir, d.prefix }) catch return
        else if (stage == lfm3.vision.depth)
            std.fmt.bufPrint(&b, "{s}/{s}rows.bf16", .{ d.dir, d.prefix }) catch return
        else
            std.fmt.bufPrint(&b, "{s}/{s}layer_{d:0>2}.bf16", .{ d.dir, d.prefix, @as(u32, @intCast(stage)) }) catch return;
        std.Io.Dir.cwd().writeFile(d.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(rows) }) catch {};
    }
};

/// Milliseconds on the monotonic clock.
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
    if (args.len < 3) {
        std.debug.print("skipped: usage tf-xpu-lfm3_tower-test MODEL_DIR OUT_DIR [IMAGE MAX_TOKENS]\n", .{});
        return;
    }
    const out = args[2];
    const p = if (args.len > 4) blk: {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[3], gpa, .limited(64 << 20));
        break :blk try img.prepare(io, gpa, bytes, try std.fmt.parseInt(u32, args[4], 10));
    } else try synthetic(gpa);
    const P = p.gh * p.gw;
    std.debug.print("picture: {d} x {d} patches ({d}), {d} image rows\n", .{ p.gh, p.gw, P, p.tokens });
    var b: [256]u8 = undefined;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&b, "{s}/pixels.f32", .{out}), .data = std.mem.sliceAsBytes(p.pixels) });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.bufPrint(&b, "{s}/grid.json", .{out}), .data = try std.fmt.bufPrint(b[128..], "{{\"gh\": {d}, \"gw\": {d}}}\n", .{ p.gh, p.gw }) });
    var r = try rt.open();
    defer r.deinit();
    var l = try xpu.loader.Loader.init(gpa, &r, args[1]);
    const t0 = now();
    const t = try Tower.load(gpa, &r, &l, (P + 3) / 4 * 4);
    defer t.deinit();
    std.debug.print("tower loaded in {d:.0} ms, {d} MiB on the device with scratch for {d} patches\n", .{ now() - t0, t.bytes >> 20, P });
    const rows = try l.empty(@as(usize, p.tokens) * lfm3.vision.out_width * 2);
    var dump: Dump = .{ .io = io, .dir = out, .prefix = "xpu_" };
    try t.image(p.pixels, p.gh, p.gw, rows, .{ .ctx = &dump, .f = Dump.take });
    for (0..3) |i| {
        try r.sync();
        const t1 = now();
        try t.image(p.pixels, p.gh, p.gw, rows, null);
        try r.sync();
        std.debug.print("  run {d}: tower and projector {d:.1} ms\n", .{ i + 1, now() - t1 });
    }
    // planted control: the same rows with the grid passed transposed (gw x gh) must not match the reference
    var ctl: Dump = .{ .io = io, .dir = out, .prefix = "transposed_" };
    try t.image(p.pixels, p.gw, p.gh, rows, .{ .ctx = &ctl, .f = Dump.take });
    std.debug.print("lfm3_tower: dumps in {s}\n", .{out});
}
