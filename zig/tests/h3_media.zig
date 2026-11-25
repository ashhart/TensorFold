//! tf-h3-media: DiT rows -> RGB frames (video VAE) + stereo WAV (audio VAE), pure Zig.
//! Input: PREFIX.video.f32 (rows of 96), PREFIX.audio.f32 (rows of 32) from tf-h3-dit.
//! Usage: tf-h3-media fl2va_dir prefix lat_depth lat_height lat_width out_prefix
const std = @import("std");
const mtl = @import("metal");
const vae = @import("vae");
const vae_audio = @import("vae_audio");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fread(ptr: *anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;

fn writeBytes(gpa_alloc: std.mem.Allocator, values: []const u8, path: []const u8) !void {
    const p = try gpa_alloc.allocSentinel(u8, path.len, 0);
    defer gpa_alloc.free(p);
    @memcpy(p[0..path.len], path);
    const file = fopen(p.ptr, "wb") orelse return error.CannotWrite;
    defer _ = fclose(file);
    if (fwrite(values.ptr, 1, values.len, file) != values.len) return error.CannotWrite;
}

fn readF32(gpa_alloc: std.mem.Allocator, io: std.Io, path: []const u8) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa_alloc, .limited(1 << 31));
    errdefer gpa_alloc.free(bytes);
    if (bytes.len % 4 != 0) return error.BadFloats;
    return @as([]f32, @alignCast(std.mem.bytesAsSlice(f32, bytes)));
}

fn loadMeanStd(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, mean: *[32]f32, sstd: *[32]f32) !void {
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const p = try std.fmt.allocPrintSentinel(arena, "{s}/config.json", .{dir}, 0);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, p, arena, .limited(1 << 20));
    const parsed = try std.json.parseFromSlice(std.json.Value, arena, bytes, .{});
    const o = parsed.value.object;
    const ma = o.get("latents_mean").?.array;
    const sa = o.get("latents_std").?.array;
    for (ma.items, 0..) |v, i| mean[i] = if (v == .float) @floatCast(v.float) else @floatFromInt(v.integer);
    for (sa.items, 0..) |v, i| sstd[i] = if (v == .float) @floatCast(v.float) else @floatFromInt(v.integer);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 7) {
        std.debug.print("usage: tf-h3-media fl2va_dir prefix lat_depth lat_height lat_width out_prefix", .{});
        return error.BadArguments;
    }
    const gpa = std.heap.page_allocator;
    const device = try mtl.Device.init();
    const depth = try std.fmt.parseInt(usize, args[3], 10);
    const height = try std.fmt.parseInt(usize, args[4], 10);
    const width = try std.fmt.parseInt(usize, args[5], 10);

    // video: rows (F * H/2 * W/2, 96) -> (C, F, H, W) channel-major
    {
        var pbuf: [4096]u8 = undefined;
        _ = &pbuf;
        const vpath = try std.fmt.allocPrintSentinel(gpa, "{s}.video.f32", .{args[2]}, 0);
        const rows = try readF32(gpa, init.io, vpath);
        const rw: usize = 96; // 24 channels * patch (1,2,2)
        if (rows.len % rw != 0) return error.BadVideoRows;
        const nrows = rows.len / rw;
        if (nrows != depth * (height / 2) * (width / 2)) return error.BadVideoRowGrid;
        const latents = try gpa.alloc(f32, rows.len);
        defer gpa.free(latents);
        const hw2 = (height / 2) * (width / 2);
        const fhw = depth * height * width;
        for (0..nrows) |r| {
            const f = r / hw2;
            const hw = r % hw2;
            const h = hw / (width / 2);
            const w = hw % (width / 2);
            for (0..24) |c| {
                for (0..2) |ph| {
                    for (0..2) |pw| {
                        const v = rows[r * rw + c * 4 + ph * 2 + pw];
                        latents[c * fhw + f * height * width + (h * 2 + ph) * width + (w * 2 + pw)] = v;
                    }
                }
            }
        }
        var decoder = try vae.Decoder.load(gpa, init.io, device, args[1]);
        defer decoder.deinit();
        const mean24 = decoder.config.latents_mean;
        const std24 = decoder.config.latents_std;
        const denorm = try gpa.alloc(f32, latents.len);
        defer gpa.free(denorm);
        for (0..24) |c| {
            for (0..fhw) |i| denorm[c * fhw + i] = latents[c * fhw + i] * std24[c] + mean24[c];
        }
        const pixels = try decoder.frames(gpa, denorm, depth, height, width);
        defer gpa.free(pixels);
        var obuf: [4096]u8 = undefined;
        _ = &obuf;
        const opath = try std.fmt.allocPrintSentinel(gpa, "{s}.rgb", .{args[6]}, 0);
        try writeBytes(gpa, pixels, opath);
        std.debug.print("video: {d} frames {d}x{d}", .{ depth * 4, height * 16, width * 16 });
    }

    // audio: rows (2T, 32) b-major -> (2, T, 32) -> BigVGAN -> WAV
    {
        var pbuf: [4096]u8 = undefined;
        _ = &pbuf;
        const apath = try std.fmt.allocPrintSentinel(gpa, "{s}.audio.f32", .{args[2]}, 0);
        const rows = try readF32(gpa, init.io, apath);
        const vae_ch: usize = 32;
        if (rows.len % (2 * vae_ch) != 0) return error.BadAudioRows;
        const t = rows.len / (2 * vae_ch);
        var mean: [32]f32 = undefined;
        var sstd: [32]f32 = undefined;
        const adir = try std.fmt.allocPrintSentinel(gpa, "{s}/audio_vae", .{args[1]}, 0);
        try loadMeanStd(gpa, init.io, adir, &mean, &sstd);
        const av = try vae_audio.AudioVAE.load(gpa, device, adir);
        const t_out = t * 800;
        const wave = try gpa.alloc(f32, 2 * t_out);
        defer gpa.free(wave);
        @memset(wave, 0);
        for (0..2) |b_| {
            const zb = try gpa.alloc(f32, vae_ch * t);
            defer gpa.free(zb);
            for (0..vae_ch) |c| {
                for (0..t) |tt| {
                    zb[c * t + tt] = rows[(b_ * t + tt) * vae_ch + c] * sstd[c] + mean[c];
                }
            }
            const w = try av.decodeOne(gpa, zb, t);
            defer gpa.free(w);
            if (w.len != t_out) return error.BadWaveLength;
            @memcpy(wave[b_ * t_out ..][0..t_out], w);
        }
        var peak: f32 = 0;
        for (wave) |v| {
            const a = @abs(v);
            if (a > peak) peak = a;
        }
        const gain: f32 = if (peak > 1.0) 1.0 / peak else 1.0;
        const nsamp = 2 * t_out / 2; // interleaved stereo frames
        const data_len: u32 = @intCast(nsamp * 4);
        const riff_len: u32 = 36 + data_len;
        var wbuf: [4096]u8 = undefined;
        _ = &wbuf;
        const wpath = try std.fmt.allocPrintSentinel(gpa, "{s}.wav", .{args[6]}, 0);
        const wf = fopen(wpath.ptr, "wb") orelse return error.CannotWrite;
        defer _ = fclose(wf);
        _ = fwrite("RIFF".ptr, 1, 4, wf);
        _ = fwrite(std.mem.asBytes(&riff_len).ptr, 1, 4, wf);
        _ = fwrite("WAVE".ptr, 1, 4, wf);
        _ = fwrite("fmt ".ptr, 1, 4, wf);
        const fmt_len: u32 = 16;
        const fmt_tag: u16 = 1;
        const ch: u16 = 2;
        const rate: u32 = 32000;
        const byte_rate: u32 = 32000 * 4;
        const block: u16 = 4;
        const bits: u16 = 16;
        _ = fwrite(std.mem.asBytes(&fmt_len).ptr, 1, 4, wf);
        _ = fwrite(std.mem.asBytes(&fmt_tag).ptr, 1, 2, wf);
        _ = fwrite(std.mem.asBytes(&ch).ptr, 1, 2, wf);
        _ = fwrite(std.mem.asBytes(&rate).ptr, 1, 4, wf);
        _ = fwrite(std.mem.asBytes(&byte_rate).ptr, 1, 4, wf);
        _ = fwrite(std.mem.asBytes(&block).ptr, 1, 2, wf);
        _ = fwrite(std.mem.asBytes(&bits).ptr, 1, 2, wf);
        _ = fwrite("data".ptr, 1, 4, wf);
        _ = fwrite(std.mem.asBytes(&data_len).ptr, 1, 4, wf);
        for (0..nsamp) |i| {
            const l: i16 = @intFromFloat(std.math.clamp(wave[2 * i] * gain, -1.0, 1.0) * 32767.0);
            const r: i16 = @intFromFloat(std.math.clamp(wave[2 * i + 1] * gain, -1.0, 1.0) * 32767.0);
            _ = fwrite(std.mem.asBytes(&l).ptr, 1, 2, wf);
            _ = fwrite(std.mem.asBytes(&r).ptr, 1, 2, wf);
        }
        std.debug.print("audio: {d} stereo samples @32k, peak {d:.3}", .{ nsamp, peak });
    }
}
