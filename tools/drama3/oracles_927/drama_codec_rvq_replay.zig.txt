//! Diagnostic: swaps only the RVQ function of a production codec instance.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn weightHash(model: *drama.codec.Model, stream: cuda.Stream) ![64]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    var offset: usize = 0;
    while (offset < model.weights.len) {
        const n = @min(model.audio_host.bytes.len, model.weights.len - offset);
        const staging = model.audio_host.bytes[0..n];
        try model.weights.downloadAsync(offset, staging, stream.handle);
        try stream.synchronize();
        h.update(staging);
        offset += n;
    }
    return std.fmt.bytesToHex(h.finalResult(), .lower);
}

fn save(dir: std.Io.Dir, io: std.Io, name: []const u8, bytes: []const u8) !void {
    const file = try dir.createFile(io, name, .{ .exclusive = true });
    defer file.close(io);
    try file.writePositionalAll(io, bytes, 0);
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("usage: tf-drama-codec-rvq-replay CHECKPOINT CODES_U32LE SHADOW_FATBIN OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var bytes: [1024 * 10 * 4 + 1]u8 align(@alignOf(u32)) = undefined;
    const input = try std.Io.Dir.cwd().openFile(init.io, args[2], .{});
    defer input.close(init.io);
    const count = try input.readPositionalAll(init.io, &bytes, 0);
    if (count != 1024 * 10 * 4) return error.Expected1024Frames;
    const codes = std.mem.bytesAsSlice([10]u32, @as([]align(@alignOf(u32)) u8, bytes[0..count]));
    const codes_hash = hash(bytes[0..count]);
    var dir = try std.Io.Dir.cwd().openDir(init.io, args[4], .{});
    defer dir.close(init.io);
    const pcm = try init.gpa.alloc(f32, 1024 * 2048 + 16);
    defer init.gpa.free(pcm);
    const image = try std.Io.Dir.cwd().readFileAllocOptions(init.io, args[3], init.gpa, .limited(1 << 28), .@"16", null);
    defer init.gpa.free(image);
    _ = try cuda.setAllocationLimit(4 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var shadow = try cuda.Module.load(&driver, image);
    defer shadow.unload();
    const candidate = try shadow.function("drama_codec_rvq_latent_fp64_shadow");
    var codec = try drama.codec.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .max_frames = 1024 });
    defer codec.deinit();
    const original = codec.kernels.rvq;
    defer codec.kernels.rvq = original;
    const weights_before = try weightHash(&codec, stream);
    var manifest: std.ArrayList(u8) = .empty;
    defer manifest.deinit(init.gpa);
    try manifest.print(init.gpa, "{{\"schema\":1,\"codes_sha256\":\"{s}\",\"weights_before_sha256\":\"{s}\",\"shadow_sha256\":\"{s}\",\"allocation_cap_bytes\":{d},\"runs\":[\n", .{ codes_hash, weights_before, hash(image), @as(u64, 4) << 30 });
    var first = true;
    for ([_]usize{ 1, 2, 7, 128, 1024 }) |frames| {
        for ([_][]const u8{ "baseline", "baseline_repeat", "candidate", "candidate_repeat", "restored" }, 0..) |label, phase| {
            codec.kernels.rvq = if (phase == 2 or phase == 3) candidate else original;
            const samples = frames * 2048;
            const sentinel: f32 = @bitCast(@as(u32, 0x7fc0dead));
            @memset(pcm[0 .. samples + 16], sentinel);
            const begin = std.Io.Clock.awake.now(init.io).toNanoseconds();
            const written = try codec.decode(stream, codes[0..frames], pcm);
            const elapsed = std.Io.Clock.awake.now(init.io).toNanoseconds() - begin;
            if (written != samples) return error.UnexpectedSampleCount;
            for (pcm[0..samples]) |v| if (!std.math.isFinite(v)) return error.NonFiniteAudio;
            for (pcm[samples .. samples + 16]) |v| if (@as(u32, @bitCast(v)) != 0x7fc0dead) return error.OutputOverrun;
            const code_bytes = std.mem.sliceAsBytes(codes[0..frames]);
            try codec.codes_dev.downloadAsync(0, codec.codes_host.bytes[0..code_bytes.len], stream.handle);
            try stream.synchronize();
            if (!std.mem.eql(u8, code_bytes, codec.codes_host.bytes[0..code_bytes.len])) return error.CodesChanged;
            var name_buf: [80]u8 = undefined;
            const name = try std.fmt.bufPrint(&name_buf, "{s}-{d}.f32", .{ label, frames });
            const audio = std.mem.sliceAsBytes(pcm[0..samples]);
            try save(dir, init.io, name, audio);
            const used = cuda.usage(false);
            try manifest.print(init.gpa, "{s}{{\"phase\":\"{s}\",\"frames\":{d},\"samples\":{d},\"pcm_sha256\":\"{s}\",\"elapsed_s\":{d:.9},\"cuda_bytes\":{d},\"pinned_bytes\":{d},\"finite\":true,\"sentinel_untouched\":true,\"codes_unchanged\":true}}\n", .{ if (first) "" else ",", label, frames, written, hash(audio), @as(f64, @floatFromInt(elapsed)) / 1e9, used.device, used.host });
            first = false;
            if (phase == 0 or phase == 2) {
                // The RVQ latent alone, from an isolated launch of the function this phase decodes with.
                const w = codec.w;
                const base = codec.weights.ptr;
                try codec.kernels.rvqLatent(stream, codec.codes_dev.ptr, base + w.semantic_book, base + w.residual_books, base + w.out_w, base + w.out_b, codec.acts[0].ptr, frames);
                const latent = codec.audio_host.bytes[0 .. frames * 1024 * 2];
                try codec.acts[0].downloadAsync(0, latent, stream.handle);
                try stream.synchronize();
                const latent_name = try std.fmt.bufPrint(&name_buf, "{s}-{d}.latent.f16", .{ label, frames });
                try save(dir, init.io, latent_name, latent);
            }
        }
    }
    const weights_after = try weightHash(&codec, stream);
    if (!std.mem.eql(u8, &weights_before, &weights_after)) return error.WeightsChanged;
    if (!std.mem.eql(u8, &codes_hash, &hash(bytes[0..count]))) return error.HostCodesChanged;
    try manifest.print(init.gpa, "],\"weights_after_sha256\":\"{s}\",\"weights_unchanged\":true,\"host_codes_unchanged\":true}}\n", .{weights_after});
    try save(dir, init.io, "manifest.json", manifest.items);
    return 0;
}
