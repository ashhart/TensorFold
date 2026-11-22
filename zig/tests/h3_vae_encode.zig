//! H3 video VAE CNN encoder: full forward with Metal conv.
const std = @import("std");
const enc = @import("vae_encoder");
const mtl = @import("metal");
const vit = @import("vae");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, nmemb: usize, stream: *anyopaque) usize;
extern "c" fn fclose(stream: *anyopaque) c_int;
extern "c" fn fseek(stream: *anyopaque, offset: c_long, whence: c_int) c_int;
extern "c" fn ftell(stream: *anyopaque) c_long;
extern "c" fn fread(ptr: [*]u8, size: usize, nmemb: usize, stream: *anyopaque) usize;

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

fn fileExists(path: []const u8) bool {
    var path_buf: [512]u8 = undefined;
    const formatted = std.fmt.bufPrint(&path_buf, "{s}", .{path}) catch return false;
    path_buf[formatted.len] = 0;
    const file = fopen(@ptrCast(&path_buf), "rb") orelse return false;
    _ = fclose(file);
    return true;
}

fn saveF32(gpa: std.mem.Allocator, path: []const u8, data: []const f32) !void {
    _ = gpa;
    var path_buf: [512]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&path_buf, "{s}", .{path});
    path_buf[formatted.len] = 0;
    const file = fopen(@ptrCast(&path_buf), "wb") orelse return error.CannotWrite;
    _ = fwrite(@ptrCast(data.ptr), 4, data.len, file);
    _ = fclose(file);
}

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: tf-h3-vae-encode pipeline_dir weights_dir input.f32 T H W", .{});
        return;
    }
    const device = try mtl.Device.init();
    const decoder = try vit.Decoder.load(gpa, init.io, device, args[1]);
    defer decoder.vit_checkpoint.deinit();
    _ = decoder.p[8]; // h3_vae_enc_conv (Metal conv — future)

    const wdir = args[2];
    const in_path = args[3];
    const t_in = try std.fmt.parseInt(usize, args[4], 10);
    const h_in = try std.fmt.parseInt(usize, args[5], 10);
    const w_in = try std.fmt.parseInt(usize, args[6], 10);

    const x = try loadF32(gpa, in_path);
    std.debug.print("input loaded: {d} floats ({d}x{d}x{d}x{d})", .{ x.len, 3, t_in, h_in, w_in });

    var encoder = enc.Encoder.init(.{});
    var w = enc.AllWeights{
        .conv_in_w = undefined, .conv_in_b = undefined,
        .levels = undefined,
        .norm_out_w = undefined, .norm_out_b = undefined,
        .conv_out_w = undefined, .conv_out_b = undefined,
        .quant_w = undefined, .quant_b = undefined,
    };
    var path_buf: [512]u8 = undefined;
    w.conv_in_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_conv_in_weight.f32", .{wdir}));
    w.conv_in_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_conv_in_bias.f32", .{wdir}));
    for (0..6) |level| {
        w.levels[level] = .{ .blocks = undefined };
        for (0..2) |bi| {
            w.levels[level].blocks[bi] = .{
                .norm1_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_norm1_weight.f32", .{ wdir, level, bi })),
                .norm1_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_norm1_bias.f32", .{ wdir, level, bi })),
                .conv1_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_conv1_weight.f32", .{ wdir, level, bi })),
                .conv1_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_conv1_bias.f32", .{ wdir, level, bi })),
                .norm2_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_norm2_weight.f32", .{ wdir, level, bi })),
                .norm2_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_norm2_bias.f32", .{ wdir, level, bi })),
                .conv2_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_conv2_weight.f32", .{ wdir, level, bi })),
                .conv2_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_conv2_bias.f32", .{ wdir, level, bi })),
                .nin_w = null, .nin_b = null,
            };
            const nin_path = try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_nin_shortcut_weight.f32", .{ wdir, level, bi });
            if (fileExists(nin_path)) {
                w.levels[level].blocks[bi].nin_w = try loadF32(gpa, nin_path);
                const nb_path = try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_nin_shortcut_bias.f32", .{ wdir, level, bi });
                w.levels[level].blocks[bi].nin_b = try loadF32(gpa, nb_path);
            }
        }
        if (level < 4) {
            w.levels[level].down_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_downsample_conv_weight.f32", .{ wdir, level }));
            w.levels[level].down_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_downsample_conv_bias.f32", .{ wdir, level }));
        }
    }
    w.norm_out_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_norm_out_weight.f32", .{wdir}));
    w.norm_out_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_norm_out_bias.f32", .{wdir}));
    w.conv_out_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_conv_out_weight.f32", .{wdir}));
    w.conv_out_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/encoder_conv_out_bias.f32", .{wdir}));
    w.quant_w = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/quant_conv_weight.f32", .{wdir}));
    w.quant_b = try loadF32(gpa, try std.fmt.bufPrint(&path_buf, "{s}/quant_conv_bias.f32", .{wdir}));

    std.debug.print("encoding {d}x{d}x{d} (CPU conv)...", .{ t_in, h_in, w_in });
    const moments = try encoder.forward(gpa, x, t_in, h_in, w_in, &w);
    const oh = h_in / 16;
    const ow = w_in / 16;
    var msum: f64 = 0;
    for (0..24 * t_in * oh * ow) |i| msum += moments[i];
    std.debug.print("mean-of-mean: {d:.6}", .{msum / @as(f64, @floatFromInt(24 * t_in * oh * ow))});
    try saveF32(gpa, "moments.f32", moments);
    std.debug.print("moments written", .{});
}


