//! tf-moss-image-decoder: MOSS-Image's image decoder on CUDA, a latents file to the raw float32 picture.
const std = @import("std");
const mi = @import("moss_image");

const usage =
    \\tf-moss-image-decoder VAE_DIR LATENTS OUT WIDTH HEIGHT
    \\VAE_DIR is the OpenMOSS-Team/MOSS-Image-Base folder's vae/; LATENTS the transformer's output (height/16 x width/16
    \\rows of 128 float32, as tf-moss-image writes it). The decoder runs in float32, as the pipeline decodes, and OUT holds
    \\its picture before the clamp: [height, width, 3] float32. Its timing and the picture's range are printed.
;

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 6) {
        std.debug.print("{s}\n", .{usage});
        return 2;
    }
    const width = try std.fmt.parseInt(u32, args[4], 10);
    const height = try std.fmt.parseInt(u32, args[5], 10);
    const cwd = std.Io.Dir.cwd();
    const lat_bytes = try cwd.readFileAlloc(io, args[2], a, .limited(1 << 30));
    const latents = try a.alloc(f32, lat_bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(latents), lat_bytes[0 .. latents.len * 4]);
    const picture = try a.alloc(f32, @as(usize, width) * @as(usize, height) * 3);
    const loading = mi.seconds();
    const d = try mi.Decoder.load(std.heap.page_allocator, io, args[1]);
    defer d.deinit();
    const load_s = mi.seconds() - loading;
    var buf: [256]u8 = undefined;
    std.debug.print("{s}: {d} latent values, {d}x{d} picture; loaded in {d:.2} s\n", .{ d.device(&buf), latents.len, width, height, load_s });
    const t = try d.decode(latents, width, height, picture);
    var low: f32 = std.math.inf(f32);
    var high: f32 = -std.math.inf(f32);
    var total: f64 = 0;
    for (picture) |v| {
        low = @min(low, v);
        high = @max(high, v);
        total += @abs(v);
    }
    std.debug.print("decoded in {d:.3} s ({d:.3} s on the GPU); the picture runs {d:.4} to {d:.4}, mean |value| {d:.4}\n", .{ t.wall, t.gpu, low, high, total / @as(f64, @floatFromInt(picture.len)) });
    try cwd.writeFile(io, .{ .sub_path = args[3], .data = std.mem.sliceAsBytes(picture) });
    return 0;
}
