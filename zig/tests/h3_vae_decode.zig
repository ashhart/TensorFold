//! tf-h3-vae-decode: the H3 video VAE decoder against the render's latents,
//! writing RGB frames. Parity target: Python vae/reference/vae_video.py.
const std = @import("std");
const mtl = @import("metal");
const vae = @import("vae");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fclose(file: *anyopaque) c_int;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;

fn writeBytes(gpa_alloc: std.mem.Allocator, values: []const u8, comptime fmt: []const u8, args: anytype) !void {
    const path = try gpa_alloc.allocSentinel(u8, std.fmt.count(fmt, args), 0);
    _ = std.fmt.bufPrint(path, fmt, args) catch unreachable;
    const file = fopen(path.ptr, "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(values.ptr, 1, values.len, file) != values.len) return error.CannotWrite;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) {
        std.debug.print("usage: tf-h3-vae-decode pipeline_dir latents.f32 depth height width out_prefix\n", .{});
        return error.BadArguments;
    }
    const gpa = init.gpa;
    const device = try mtl.Device.init();

    const latents_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(1 << 30));
    defer gpa.free(latents_bytes);
    const depth = try std.fmt.parseInt(usize, args[3], 10);
    const height = try std.fmt.parseInt(usize, args[4], 10);
    const width = try std.fmt.parseInt(usize, args[5], 10);
    const latents = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, latents_bytes)));
    if (latents.len != 24 * depth * height * width) return error.BadLatentsSize;

    var decoder = try vae.Decoder.load(gpa, init.io, device, args[1]);
    defer decoder.deinit();
    const pixels = try decoder.frames(gpa, latents, depth, height, width);
    defer gpa.free(pixels);

    try writeBytes(gpa, pixels, "{s}.rgb", .{args[6]});
    std.debug.print("decoded {d} frames {d}x{d}\n", .{ depth * 4, height * 16, width * 16 });
}
