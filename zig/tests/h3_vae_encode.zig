//! H3 video VAE CNN encoder: Metal conv validation against CPU.
const std = @import("std");
const enc = @import("vae_encoder");
const mtl = @import("metal");
const vit = @import("vae");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, nmemb: usize, stream: *anyopaque) usize;
extern "c" fn fclose(stream: *anyopaque) c_int;
extern "c" fn getenv(name: [*:0]const u8) ?[*:0]u8;

fn loadF32(gpa: std.mem.Allocator, path: []const u8) ![]f32 {
    var path_buf: [512]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&path_buf, "{s}", .{path});
    path_buf[formatted.len] = 0;
    const file = fopen(@ptrCast(&path_buf), "rb") orelse return error.CannotOpen;
    defer _ = fclose(file);
    _ = fseek(file, 0, 2);
    const size = ftell(file);
    _ = fseek(file, 0, 0);
    const n: usize = @intCast(@divExact(size, @as(c_long, 4)));
    const out = try gpa.alloc(f32, n);
    const raw = try gpa.alloc(u8, @intCast(size));
    defer gpa.free(raw);
    _ = fread(raw.ptr, 1, @intCast(size), file);
    for (0..n) |i| out[i] = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
    return out;
}

extern "c" fn fseek(stream: *anyopaque, offset: c_long, whence: c_int) c_int;
extern "c" fn ftell(stream: *anyopaque) c_long;
extern "c" fn fread(ptr: [*]u8, size: usize, nmemb: usize, stream: *anyopaque) usize;

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-h3-vae-encode pipeline_dir weights_dir", .{});
        return;
    }
    const device = try mtl.Device.init();
    const queue = try device.queue();
    // Load decoder lib for its pipelines (encoder kernel is in same source)
    const decoder = try vit.Decoder.load(gpa, init.io, device, args[1]);
    defer decoder.vit_checkpoint.deinit();
    const pipeline = decoder.p[8]; // h3_vae_enc_conv

    // synthetic input (3, 1, 8, 8)
    var input: [3 * 1 * 8 * 8]f32 = undefined;
    for (0..3) |c| for (0..1) |t| for (0..8) |hh| for (0..8) |ww| {
        input[((c * 1) + t) * 64 + hh * 8 + ww] = @floatFromInt(c * 64 + hh * 8 + ww);
    };
    // conv weights: 4 output channels, kernel (1,3,3,3) causal
    const c_out = 4;
    var wgt: [4 * 3 * 27]f32 = undefined;
    for (0..wgt.len) |i| wgt[i] = @floatFromInt(i % 7);
    var bias: [4]f32 = .{ 0.1, 0.2, 0.3, 0.4 };

    // CPU reference
    const cpu_out = try enc.conv3d(gpa, &input, &wgt, &bias, 3, c_out, 1, 8, 8, 3, 3, 3, 1, 1, 1, true, 1, 0, 0);
    defer gpa.free(cpu_out);

    // Metal
    const metal_out = try enc.conv3dMetal(device, queue, pipeline, gpa, &input, &wgt, &bias, 3, c_out, 1, 8, 8, true, 1, 0, 0);
    defer gpa.free(metal_out);

    var max_diff: f32 = 0;
    for (0..cpu_out.len) |i| {
        const d = @abs(cpu_out[i] - metal_out[i]);
        if (d > max_diff) max_diff = d;
    }
    std.debug.print("METAL-CONV max diff vs CPU: {d:.6} (len {d})", .{ max_diff, cpu_out.len });
    for (0..@min(8, cpu_out.len)) |i| {
        std.debug.print("  [{d}] cpu={d:.4} metal={d:.4}", .{ i, cpu_out[i], metal_out[i] });
    }
}
