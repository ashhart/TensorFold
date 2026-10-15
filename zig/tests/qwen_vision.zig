//! tf-qwen-vision: a Qwen-VL checkpoint's tower on picture files; each image's patches and merged rows written out.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const vl = tf.qwen_vl;

const usage =
    \\tf-qwen-vision MODEL_DIR OUT_DIR TOKENS PICTURE...
    \\Each picture prepared natively under a TOKENS budget and run through the tower twice: OUT_DIR gets NAME.pixels.f32
    \\(the patches), NAME.rows.f32 (the merged rows, bf16 widened) and a meta.json entry (grid, tokens, ms, repeat_equal).
    \\A NAME.ref.f32 already in OUT_DIR (a float32 reference's rows) is compared: each row's relative L2 error, worst and mean.
;

fn read(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ?[]const u8 {
    const path = std.fs.path.join(a, &.{ dir, name }) catch return null;
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch null;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len < 5) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    const dir = args[1];
    const out = args[2];
    const tokens = try std.fmt.parseInt(u32, args[3], 10);
    const config = read(a, io, dir, "config.json") orelse return error.NoConfig;
    const c = (try vl.Config.parse(a, config, read(a, io, dir, "preprocessor_config.json"), read(a, io, dir, "processor_config.json"))) orelse return error.NoVisionTower;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const queue = try device.queue();
    const t0 = std.Io.Clock.awake.now(io);
    const tower = try tf.qwen_vl_tower.Tower.load(std.heap.page_allocator, io, device, dir, c, tokens);
    defer tower.deinit();
    const load_ms = @as(f64, @floatFromInt(t0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
    std.debug.print("{s} on {s}: tower loaded in {d:.0} ms ({d:.2} GiB with scratch for {d} tokens)\n", .{ dir, device.name(), load_ms, @as(f64, @floatFromInt(tower.bytes)) / (1 << 30), tokens });
    const width: usize = c.out_hidden;
    const rows = try device.buffer(@as(usize, tokens) * width * 2, mtl.ResourceOptions.shared);
    defer rows.deinit();
    var meta: std.ArrayList(u8) = .empty;
    try meta.appendSlice(a, "{");
    for (args[4..], 0..) |pic, n| {
        const bytes = std.Io.Dir.cwd().readFileAlloc(io, pic, a, .limited(64 << 20)) catch |err| {
            std.debug.print("{s}: {t}\n", .{ pic, err });
            return 1;
        };
        const p = try vl.geometry.prepare(io, a, bytes, c, tokens);
        const count = @as(usize, p.tokens) * width;
        const first = try a.alloc(u16, count);
        var ms: f64 = 0;
        var same = true;
        for (0..2) |pass| {
            const s0 = std.Io.Clock.awake.now(io);
            try tower.image(queue, p.pixels, p.gh, p.gw, .{ .buf = rows });
            ms = @as(f64, @floatFromInt(s0.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
            const got = rows.slice(u16, count);
            if (pass == 0) @memcpy(first, got) else same = std.mem.eql(u16, first, got);
        }
        const wide = try a.alloc(f32, count);
        for (wide, first) |*w, b| w.* = @bitCast(@as(u32, b) << 16);
        const name = std.fs.path.stem(pic);
        const ref = compare(a, io, out, name, wide, width);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.pixels.f32", .{ out, name }), .data = std.mem.sliceAsBytes(p.pixels) });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/{s}.rows.f32", .{ out, name }), .data = std.mem.sliceAsBytes(wide) });
        try meta.print(a, "{s}\"{s}\": {{\"gh\": {d}, \"gw\": {d}, \"tokens\": {d}, \"hash\": \"{x}\", \"ms\": {d:.1}, \"repeat_equal\": {}, \"ref_worst\": {d:.6}, \"ref_mean\": {d:.6}}}", .{ if (n > 0) ", " else "", name, p.gh, p.gw, p.tokens, p.hash, ms, same, ref[0], ref[1] });
        std.debug.print("{s}: grid {d} x {d}, {d} tokens, tower {d:.1} ms, second pass bit-equal {}, against the reference worst {d:.6} mean {d:.6}\n", .{ name, p.gh, p.gw, p.tokens, ms, same, ref[0], ref[1] });
        if (!same) return 1;
    }
    try meta.appendSlice(a, "}\n");
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = try std.fmt.allocPrint(a, "{s}/meta.json", .{out}), .data = meta.items });
    return 0;
}

/// Each row's relative L2 error against OUT/NAME.ref.f32, worst and mean; -1 without a reference of this shape.
fn compare(a: std.mem.Allocator, io: std.Io, out: []const u8, name: []const u8, rows: []const f32, width: usize) [2]f64 {
    const path = std.fmt.allocPrint(a, "{s}/{s}.ref.f32", .{ out, name }) catch return .{ -1, -1 };
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30)) catch return .{ -1, -1 };
    if (bytes.len != rows.len * 4) return .{ -1, -1 };
    const ref: []align(1) const f32 = std.mem.bytesAsSlice(f32, bytes);
    var worst: f64 = 0;
    var sum: f64 = 0;
    for (0..rows.len / width) |r| {
        var d: f64 = 0;
        var e: f64 = 0;
        for (rows[r * width ..][0..width], ref[r * width ..][0..width]) |x, y| {
            d += (@as(f64, x) - y) * (@as(f64, x) - y);
            e += @as(f64, y) * y;
        }
        const rel = @sqrt(d) / @max(@sqrt(e), 1e-30);
        worst = @max(worst, rel);
        sum += rel;
    }
    return .{ worst, sum / @as(f64, @floatFromInt(rows.len / width)) };
}
