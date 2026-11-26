//! tf-h3-audio-decode: audio latents to stereo WAV via the pure-Zig BigVGAN decoder. Args: dir latents out.
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
    const nsamp = T_out; // stereo frames: L from the first plane, R from the second
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
    writeStereo(wave, nsamp, gain, wf);
    _ = fclose(wf);
    std.debug.print("peak {d:.3} frames {d} wrote\n", .{ peak, nsamp });
}

/// Interleave the planar wave buffer, L from the first plane, R from the second.
fn writeStereo(wave: []const f32, nsamp: usize, gain: f32, wf: *anyopaque) void {
    for (0..nsamp) |i| {
        const l: i16 = @intFromFloat(std.math.clamp(wave[i] * gain, -1.0, 1.0) * 32767.0);
        const r: i16 = @intFromFloat(std.math.clamp(wave[nsamp + i] * gain, -1.0, 1.0) * 32767.0);
        _ = fwrite(std.mem.asBytes(&l).ptr, 1, 2, wf);
        _ = fwrite(std.mem.asBytes(&r).ptr, 1, 2, wf);
    }
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

test "stereo output interleaves the two planes, not one plane twice" {
    // planar buffer: [L0, L1, L2, R0, R1, R2] must give frames (L0, R0), not (L0, L1)
    const wave = [_]f32{ 0.5, -0.5, 0.25, 0.75, -0.75, 0.125 };
    var frames: [3][2]i16 = undefined;
    for (0..3) |i| {
        frames[i][0] = @intFromFloat(std.math.clamp(wave[i], -1.0, 1.0) * 32767.0);
        frames[i][1] = @intFromFloat(std.math.clamp(wave[3 + i], -1.0, 1.0) * 32767.0);
    }
    try std.testing.expectEqual(@as(i16, 16383), frames[0][0]);
    try std.testing.expectEqual(@as(i16, 24575), frames[0][1]);
    try std.testing.expectEqual(@as(i16, -16383), frames[1][0]);
    try std.testing.expectEqual(@as(i16, 4095), frames[2][1]);
    // the old bug read channel two from the first plane, giving L1 instead of R0
    try std.testing.expect(frames[0][1] != frames[1][0]);
}
