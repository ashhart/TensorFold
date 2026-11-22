//! H3 video VAE CNN encoder: full forward on synthetic weights.
const std = @import("std");
const enc = @import("vae_encoder");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fread(ptr: [*]u8, size: usize, nmemb: usize, stream: *anyopaque) usize;
extern "c" fn fclose(stream: *anyopaque) c_int;
extern "c" fn fwrite(ptr: [*]const u8, size: usize, nmemb: usize, stream: *anyopaque) usize;
extern "c" fn fseek(stream: *anyopaque, offset: c_long, whence: c_int) c_int;
extern "c" fn ftell(stream: *anyopaque) c_long;

fn loadF32(gpa: std.mem.Allocator, path: []const u8) ![]f32 {
    var path_buf: [512]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&path_buf, "{s}", .{path});
    path_buf[formatted.len] = 0;
    const pz: [*:0]const u8 = @ptrCast(&path_buf);
    const file = fopen(pz, "rb") orelse return error.CannotOpen;
    defer _ = fclose(file);
    // get size
    const seek_end: c_int = 2;
    const seek_set: c_int = 0;
    _ = fseek(file, 0, seek_end);
    const size = ftell(file);
    _ = fseek(file, 0, seek_set);
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

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.page_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print("usage: tf-h3-vae-encode weights_dir [T H W]", .{});
        return;
    }
    const wdir = args[1];
    var t_in: usize = 3;
    var h_in: usize = 64;
    var w_in: usize = 64;
    if (args.len >= 5) {
        t_in = try std.fmt.parseInt(usize, args[2], 10);
        h_in = try std.fmt.parseInt(usize, args[3], 10);
        w_in = try std.fmt.parseInt(usize, args[4], 10);
    }

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
                .nin_w = null,
                .nin_b = null,
            };
            // nin_shortcut if exists
            const nin_path = try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_nin_shortcut_weight.f32", .{ wdir, level, bi });
            if (fileExists(nin_path)) {
                w.levels[level].blocks[bi].nin_w = try loadF32(gpa, nin_path);
                const nb_path = try std.fmt.bufPrint(&path_buf, "{s}/encoder_down_{d}_block_{d}_nin_shortcut_bias.f32", .{ wdir, level, bi });
                w.levels[level].blocks[bi].nin_b = try loadF32(gpa, nb_path);
            }
        }
        // downsample — level 0..3 have it, 4-5 don't
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

    // synthetic input: (3, T, H, W) gradient
    const x = try gpa.alloc(f32, 3 * t_in * h_in * w_in);
    for (0..3) |c| {
        for (0..t_in) |t| {
            for (0..h_in) |hh| {
                for (0..w_in) |ww| {
                    x[((c * t_in) + t) * h_in * w_in + hh * w_in + ww] =
                        @as(f32, @floatFromInt(c + t + hh + ww)) / 100.0;
                }
            }
        }
    }

    std.debug.print("encoding {d}x{d}x{d}...", .{ t_in, h_in, w_in });
    const moments = try encoder.forward(gpa, x, t_in, h_in, w_in, &w);
    // moments: (48, T, H/16, W/16)
    const oh = h_in / 16;
    const ow = w_in / 16;
    var mean_sum: f64 = 0;
    for (0..24) |c| {
        for (0..t_in) |t| {
            for (0..oh) |hh| {
                for (0..ow) |ww| {
                    mean_sum += moments[((c * t_in) + t) * oh * ow + hh * ow + ww];
                }
            }
        }
    }
    std.debug.print("mean-of-mean: {d:.6}", .{mean_sum / @as(f64, @floatFromInt(24 * t_in * oh * ow))});
    // write moments
    {
        var pz_buf: [512]u8 = undefined;
        const formatted = try std.fmt.bufPrint(&pz_buf, "moments.f32", .{});
        pz_buf[formatted.len] = 0;
        const file = fopen(@ptrCast(&pz_buf), "wb") orelse return error.CannotWrite;
        _ = fwrite(@ptrCast(moments.ptr), 4, moments.len, file);
        _ = fclose(file);
    }
    std.debug.print("moments written to moments.f32", .{});
}
