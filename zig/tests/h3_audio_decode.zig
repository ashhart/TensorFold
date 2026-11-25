//! tf-h3-audio-decode: decode audio latents (2 x 32 x T) to stereo WAV via the pure-Zig
//! BigVGAN audio VAE decoder. Usage: tf-h3-audio-decode <model_dir> <latents.f32> <out.wav>
const std = @import("std");
const mtl = @import("metal");
const vae = @import("vae_audio");

extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque;
extern "c" fn fwrite(ptr: *const anyopaque, size: usize, count: usize, file: *anyopaque) usize;
extern "c" fn fclose(file: *anyopaque) c_int;

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-h3-audio-decode model_dir latents.f32 out.wav\n", .{});
        return error.BadArguments;
    }
    const gpa = std.heap.page_allocator;
    const device = try mtl.Device.init();

    const model_dir = args[1];
    const latents_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], gpa, .limited(1 << 30));
    defer gpa.free(latents_bytes);
    const latents = @as([]const f32, @alignCast(std.mem.bytesAsSlice(f32, latents_bytes)));
    const vae_ch: usize = 32;
    const batch = latents.len / vae_ch / 207; // expect 2
    const T = latents.len / vae_ch / batch;
    std.debug.print("latents: {d} floats => {d} x {d} x {d}\n", .{ latents.len, batch, vae_ch, T });

    var mean: [32]f32 = undefined;
    var sstd: [32]f32 = undefined;
    std.debug.print("loading mean/std...\n", .{});
    try loadMeanStd(gpa, init.io, model_dir, &mean, &sstd);
    std.debug.print("loading audio vae...\n", .{});

    const av = try vae.AudioVAE.load(gpa, device, model_dir);

    std.debug.print("vae loaded\n", .{});
    const T_out = T * 800;
    const wave = try gpa.alloc(f32, batch * T_out);
    @memset(wave, 0);
    for (0..batch) |b_| {
        const zb = try gpa.alloc(f32, vae_ch * T);
        defer gpa.free(zb);
        for (0..vae_ch) |c| {
            for (0..T) |t| {
                const v = latents[(b_ * T + t) * vae_ch + c];
                zb[c * T + t] = v * sstd[c] + mean[c];
            }
        }
        const w = try av.decodeOne(gpa, zb, T);
        defer gpa.free(w);
        if (w.len != T_out) return error.BadWaveLength;
        @memcpy(wave[b_ * T_out ..][0..T_out], w);
    }

    var peak: f32 = 0;
    for (wave) |v| {
        const a = @abs(v);
        if (a > peak) peak = a;
    }
    const gain: f32 = if (peak > 1.0) 1.0 / peak else 1.0;
    const nsamp = batch * T_out / 2; // frames (interleaved stereo)
    const data_len: u32 = @intCast(nsamp * 4);
    const riff_len: u32 = 36 + data_len;
    const wf = fopen(args[3], "wb") orelse return error.CannotWrite;
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
    {
        const rf = fopen("/tmp/zig-raw.f32", "wb") orelse return error.CannotWrite;
        _ = fwrite(wave.ptr, 4, wave.len, rf);
        _ = fclose(rf);
    }
    for (0..nsamp) |i| {
        const l: i16 = @intFromFloat(std.math.clamp(wave[2 * i] * gain, -1.0, 1.0) * 32767.0);
        const r: i16 = @intFromFloat(std.math.clamp(wave[2 * i + 1] * gain, -1.0, 1.0) * 32767.0);
        _ = fwrite(std.mem.asBytes(&l).ptr, 1, 2, wf);
        _ = fwrite(std.mem.asBytes(&r).ptr, 1, 2, wf);
    }
    _ = fclose(wf);
    std.debug.print("peak {d:.3} frames {d} wrote\n", .{ peak, nsamp });
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
