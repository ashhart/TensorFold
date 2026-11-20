const std = @import("std");
const sam = @import("sam3");
const mtl = @import("metal");

pub const Sample = struct {
    frame_index: usize,
    seed: bool,
    wall_seconds: f64,
    stages: sam.profile.Frame,
};

pub const Summary = struct {
    frames: usize,
    wall_seconds: f64,
    frames_per_second: f64,
    p50_seconds: f64,
    p95_seconds: f64,
};

pub fn summarize(a: std.mem.Allocator, samples: []const Sample) !?Summary {
    if (samples.len == 0) return null;
    const times = try a.alloc(f64, samples.len);
    defer a.free(times);
    var total: f64 = 0;
    for (samples, times) |sample, *time| {
        if (!std.math.isFinite(sample.wall_seconds) or sample.wall_seconds <= 0) return error.InvalidFrameTiming;
        time.* = sample.wall_seconds;
        total += time.*;
    }
    std.mem.sort(f64, times, {}, std.sort.asc(f64));
    return .{
        .frames = samples.len,
        .wall_seconds = total,
        .frames_per_second = @as(f64, @floatFromInt(samples.len)) / total,
        .p50_seconds = times[(times.len - 1) / 2],
        .p95_seconds = times[(times.len * 95 + 99) / 100 - 1],
    };
}

pub fn hashFile(io: std.Io, path: []const u8) ![64]u8 {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const len = try file.length(io);
    if (len == 0) return hash(&.{});
    var map = try file.createMemoryMap(io, .{ .len = std.math.cast(usize, len) orelse return error.FileTooLarge, .protection = .{ .read = true, .write = false }, .populate = false });
    defer map.destroy(io);
    return hash(map.memory);
}

pub fn hash(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

pub const Identity = struct {
    checkpoint_sha256: [64]u8,
    frames_tensor_sha256: [64]u8,
    prompts_sha256: [64]u8,
    executable_sha256: [64]u8,

    pub fn load(a: std.mem.Allocator, io: std.Io, checkpoint: []const u8, frames: []const u8, seeds: []const sam.session.Seed) !Identity {
        const executable = try std.process.executablePathAlloc(io, a);
        defer a.free(executable);
        const prompt_json = try std.json.Stringify.valueAlloc(a, seeds, .{});
        defer a.free(prompt_json);
        return .{
            .checkpoint_sha256 = try hashFile(io, checkpoint),
            .frames_tensor_sha256 = hash(frames),
            .prompts_sha256 = hash(prompt_json),
            .executable_sha256 = try hashFile(io, executable),
        };
    }
};

pub fn write(a: std.mem.Allocator, io: std.Io, path: []const u8, identity: Identity, loaded: *sam.Model, load_seconds: f64, samples: []const Sample, warmup: usize, parameters: anytype, masks_hash: [64]u8) !void {
    if (warmup >= samples.len) return error.InvalidWarmup;
    const process_info = mtl.objc.msg(mtl.objc.Id, mtl.objc.class("NSProcessInfo"), "processInfo", .{});
    const json = try std.json.Stringify.valueAlloc(a, .{
        .schema = 1,
        .model = loaded.config.version.name(),
        .precision = @tagName(loaded.precision),
        .checkpoint_sha256 = identity.checkpoint_sha256[0..],
        .frames_tensor_sha256 = identity.frames_tensor_sha256[0..],
        .prompts_sha256 = identity.prompts_sha256[0..],
        .executable_sha256 = identity.executable_sha256[0..],
        .masks_sha256 = masks_hash[0..],
        .zig_version = @import("builtin").zig_version_string,
        .gpu = std.mem.span(loaded.gpu.device.name()),
        .os = mtl.objc.osVersion(),
        .physical_ram_bytes = mtl.objc.msg(u64, process_info, "physicalMemory", .{}),
        .parameters = parameters,
        .timing_scope = "session API only; excludes checkpoint load, hashes, output and comparisons",
        .load_seconds_including_checkpoint_metadata_and_metal_library = load_seconds,
        .load_and_seed_seconds = load_seconds + samples[0].wall_seconds,
        .cold_seed = samples[0],
        .propagation = try summarize(a, samples[1..]),
        .warmup_frames = warmup,
        .after_warmup = try summarize(a, samples[1 + warmup ..]),
        .samples = samples,
        .peak_metal_bytes = loaded.gpu.peak_bytes,
        .pipeline_compile_seconds = loaded.gpu.pipeline_compile_seconds,
        .parameter_prepare_seconds = loaded.store.parameter_prepare_seconds,
        .checkpoint_read_seconds = loaded.store.checkpoint_read_seconds,
        .process_memory = @import("process_memory").read(),
    }, .{ .whitespace = .indent_2 });
    defer a.free(json);
    const file = try std.Io.Dir.cwd().createFile(io, path, .{ .exclusive = true });
    defer file.close(io);
    try file.writeStreamingAll(io, json);
    try file.writeStreamingAll(io, "\n");
}

test "frame statistics retain a slow tail and exclude absent propagation" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(?Summary, null), try summarize(a, &.{}));
    var samples: [20]Sample = undefined;
    for (&samples, 0..) |*sample, i| sample.* = .{ .frame_index = i, .seed = false, .wall_seconds = if (i < 18) 1 else 100, .stages = .{} };
    const summary = (try summarize(a, &samples)).?;
    try std.testing.expectEqual(@as(f64, 218), summary.wall_seconds);
    try std.testing.expectEqual(@as(f64, 1), summary.p50_seconds);
    try std.testing.expectEqual(@as(f64, 100), summary.p95_seconds);
    samples[0].wall_seconds = std.math.nan(f64);
    try std.testing.expectError(error.InvalidFrameTiming, summarize(a, &samples));
}
