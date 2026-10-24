//! MiniMax-M3.1-Flash's image tower alone (its shard, not the 200 GiB model): an image's patches and text rows.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ks = @import("kernel_sources");
const fl = tf.minimax_m3;
const ck = tf.checkpoint_metal;

const usage =
    \\tf-minimax-vision <model> <image> <out dir>   pixels.bin ([P, 1176] fp32, merge order), grid.txt (h w),
    \\                                              rows.bin ([P / 4, 6144] bf16): tools/vision_ref.py's outputs
    \\
;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("{s}", .{usage});
        return error.ExpectedModelImageOut;
    }
    const gpa = init.gpa;
    const io = init.io;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var c = ck.Checkpoint.init(gpa);
    defer c.deinit();
    const shard = try std.fs.path.joinZ(gpa, &.{ args[1], fl.vision.shard });
    defer gpa.free(shard);
    try c.addFile(device, shard, "");
    const lib = try mtl.Library.fromSource(device, ks.minimax_m3_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
    defer lib.deinit();
    const widen = try mtl.Pipeline.init(device, lib, "mm_widen", false);
    defer widen.deinit();
    const narrow = try mtl.Pipeline.init(device, lib, "mm_narrow", false);
    defer narrow.deinit();
    const v = try fl.vision.Vision.load(gpa, device, &c, widen, narrow);
    defer v.deinit();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(64 << 20));
    defer gpa.free(bytes);
    const p = try fl.vision.prepare(io, gpa, bytes, fl.vision.max_tokens);
    defer gpa.free(p.pixels);
    const rows = try device.buffer(@as(usize, p.tokens) * fl.vision.hidden * 2, mtl.ResourceOptions.shared);
    defer rows.deinit();
    var t0: u64 = 0;
    for (0..2) |pass| { // the second pass is timed: the first pages the weights in
        if (pass == 1) t0 = now();
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        try v.encode(e, p.pixels, p.gh, p.gw, .{ .buf = rows });
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("the tower failed: {s}", .{msg});
            return error.GpuFailed;
        }
    }
    const ms = @as(f64, @floatFromInt(now() - t0)) / 1e6;
    var dir = try std.Io.Dir.cwd().createDirPathOpen(io, args[3], .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = "pixels.bin", .data = std.mem.sliceAsBytes(p.pixels) });
    try dir.writeFile(io, .{ .sub_path = "rows.bin", .data = rows.slice(u8, @as(usize, p.tokens) * fl.vision.hidden * 2) });
    var text: [32]u8 = undefined;
    try dir.writeFile(io, .{ .sub_path = "grid.txt", .data = try std.fmt.bufPrint(&text, "{d} {d}\n", .{ p.gh, p.gw }) });
    std.debug.print("minimax vision: {d} x {d} patches, {d} tokens in {d:.1} ms\n", .{ p.gh, p.gw, p.tokens, ms });
}

fn now() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
