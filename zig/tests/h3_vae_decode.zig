//! tf-h3-vae-decode: the H3 video VAE decoder against the render's latents,
//! writing RGB frames. Parity target: Python vae/reference/vae_video.py.
const std = @import("std");
const mtl = @import("metal");
const vae = @import("vae");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fread(ptr: *anyopaque, size: usize, count: usize, file: *anyopaque) usize;
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
    const gpa = std.heap.page_allocator;
    const device = try mtl.Device.init();

    const latents_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(1 << 30));
    defer gpa.free(latents_bytes);
    const depth = try std.fmt.parseInt(usize, args[3], 10);
    const height = try std.fmt.parseInt(usize, args[4], 10);
    const width = try std.fmt.parseInt(usize, args[5], 10);
    const latents = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, latents_bytes)));
    if (latents.len != 24 * depth * height * width) return error.BadLatentsSize;

    std.debug.print("loaded latents, decoding...", .{});
    var decoder = try vae.Decoder.load(gpa, init.io, device, args[1]);
    // denormalise the latents the way the transformer left them: z*std + mean
    const mean24 = decoder.config.latents_mean;
    const std24 = decoder.config.latents_std;
    const denorm = try gpa.alloc(f32, latents.len);
    defer gpa.free(denorm);
    // file layout is (C, F, H, W): denormalise per channel
    const fhw = depth * height * width;
    for (0..24) |c| {
        for (0..fhw) |i| {
            denorm[c * fhw + i] = latents[c * fhw + i] * std24[c] + mean24[c];
        }
    }
    defer decoder.deinit();
    std.debug.print("decoder loaded, calling frames", .{});
    {
        const R = 4;
        const K = 2048;
        const N = 8;
        const xb = try device.buffer(R * K * 4, mtl.ResourceOptions.shared);
        defer xb.deinit();
        @memset(xb.slice(f32, R * K), 0.25);
        const wb = try device.buffer(N * K * 4, mtl.ResourceOptions.shared);
        defer wb.deinit();
        const wh = wb.slice(f32, N * K);
        @memset(wh, 0.0);
        for (0..N) |n| wh[n * K + n] = 1.0;
        const bb2 = try device.buffer(N * 4, mtl.ResourceOptions.shared);
        defer bb2.deinit();
        @memset(bb2.slice(f32, N), 0.5);
        const yb = try device.buffer(R * N * 4, mtl.ResourceOptions.shared);
        defer yb.deinit();
        const q2 = try device.queue();
        const cb2 = q2.commandBuffer();
        const et = cb2.compute(.concurrent);
        et.setPipeline(decoder.p[0]);
        et.setBuffer(xb, 0, 0);
        et.setBuffer(wb, 0, 1);
        et.setBuffer(bb2, 0, 2);
        et.setBuffer(yb, 0, 3);
        et.setValue([4]u32{ @intCast(R), @intCast(K), @intCast(N), 0 }, 4);
        et.dispatchThreads(mtl.Size.of(N, R, 1), mtl.Size.of(64, 1, 1));
        et.end();
        cb2.commit(); cb2.wait();
        const yh = yb.slice(f32, R * N);
        std.debug.print("MATMUL-TEST y = {d} {d} {d} {d}", .{ yh[0], yh[1], yh[2], yh[3] });
    {
        // RMSNORM unit test: X = tokens from tokens-dump rows 0..3, D=2048
        const tz: [*:0]const u8 = "tokens-dump.f32";
        const tf = fopen(tz, "rb") orelse return error.CannotOpen;
        {
            const n: usize = 4357 * 2048;
            const rows_n = n / 2048;
            const xb2 = try device.buffer(n * 4, mtl.ResourceOptions.shared);
            defer xb2.deinit();
            _ = fread(xb2.slice(f32, n).ptr, 4, n, tf);
            _ = fclose(tf);
            const yb3u = try device.buffer(n * 4, mtl.ResourceOptions.shared);
            defer yb3u.deinit();
            const wb3u = try device.buffer(2048 * 4, mtl.ResourceOptions.shared);
            defer wb3u.deinit();
            @memcpy(wb.slice(f32, 2048), decoder.blocks[0].norm1_w);
            const q3 = try device.queue();
            const cb3 = q3.commandBuffer();
            const e3 = cb3.compute(.concurrent);
            e3.setPipeline(decoder.p[1]);
            e3.setBuffer(xb2, 0, 0);
            e3.setBuffer(wb3u, 0, 1);
            e3.setBuffer(yb3u, 0, 2);
            e3.setValue([1]u32{ 2048 }, 3);
            e3.dispatchGroups(mtl.Size.of(rows_n, 1, 1), mtl.Size.of(256, 1, 1));
            e3.end();
            cb3.commit(); cb3.wait();
            const yh3 = yb3u.slice(f32, n);
                        const xh3 = xb2.slice(f32, n);
            std.debug.print("RMSNORM-TEST x0={d:.4} x1={d:.4} w0={d:.4} y0={d:.4} y1={d:.4}", .{ xh3[0], xh3[1], wb3u.slice(f32, 2048)[0], yh3[0], yh3[1] });
        {
            const xb3u = try device.buffer(2 * 2048 * 4, mtl.ResourceOptions.shared);
            defer xb3u.deinit();
            const xbu2 = try device.buffer(2 * 2048 * 4, mtl.ResourceOptions.shared);
            defer xbu2.deinit();
            const xh4 = xbu2.slice(f32, 2 * 2048);
            @memset(xh4, 2.0);
            const wbu2 = try device.buffer(2048 * 4, mtl.ResourceOptions.shared);
            defer wbu2.deinit();
            @memset(wbu2.slice(f32, 2048), 1.0);
            const ybu2 = try device.buffer(2 * 2048 * 4, mtl.ResourceOptions.shared);
            defer ybu2.deinit();
            const q4u = try device.queue();
            const cb4u = q4u.commandBuffer();
            const e4u = cb4u.compute(.concurrent);
            e4u.setPipeline(decoder.p[1]);
            e4u.setBuffer(xbu2, 0, 0);
            e4u.setBuffer(wbu2, 0, 1);
            e4u.setBuffer(ybu2, 0, 2);
            e4u.setValue([1]u32{ 2048 }, 3);
            e4u.dispatchGroups(mtl.Size.of(2, 1, 1), mtl.Size.of(256, 1, 1));
            e4u.end();
            cb4u.commit(); cb4u.wait();
            const yvu = ybu2.slice(f32, 2 * 2048);
            std.debug.print("RMSNORM-UNIT y[0]={d:.4} y[1]={d:.4} y[2048]={d:.4} (expect 1.0)", .{ yvu[0], yvu[1], yvu[2048] });
        }
        }
    }
    }
    std.debug.print("MEAN24[0]={d:.4} STD24[0]={d:.4} denorm[0]={d:.4} latents[0]={d:.4}", .{ mean24[0], std24[0], denorm[0], latents[0] });
    const pixels = try decoder.frames(gpa, denorm, depth, height, width);
    std.debug.print("frames done", .{});
    defer gpa.free(pixels);

    try writeBytes(gpa, pixels, "{s}.rgb", .{args[6]});
    std.debug.print("decoded {d} frames {d}x{d}\n", .{ depth * 4, height * 16, width * 16 });
}
