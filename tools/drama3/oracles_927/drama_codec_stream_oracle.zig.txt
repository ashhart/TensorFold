//! Whole and chunked codec decodes of 1024 frame codes, each chunk over NaN-poisoned scratch; GPU lock is the caller's.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

const Allocator = std.mem.Allocator;

const frames = 1024;
const samples_per_frame = 2048;
const lengths = [_]usize{ 1, 2, 7, 128, 1024 };

const Schedule = struct { name: []const u8, chunks: []const usize };

/// The host's policy (1, then 4s), one frame at a time, the gate lengths as chunk ends, one call, irregular sizes.
fn schedules(a: Allocator) ![5]Schedule {
    var host: std.ArrayList(usize) = .empty;
    var done: usize = 1;
    try host.append(a, 1);
    while (done < frames) : (done += host.items[host.items.len - 1]) try host.append(a, @min(4, frames - done));
    var irregular: std.ArrayList(usize) = .empty;
    var state: u32 = 20261015;
    done = 0;
    while (done < frames) : (done += irregular.items[irregular.items.len - 1]) {
        state = state *% 1664525 +% 1013904223;
        try irregular.append(a, @min(1 + (state >> 8) % 300, frames - done));
    }
    const every = try a.alloc(usize, frames);
    @memset(every, 1);
    return .{
        .{ .name = "host-1-then-4", .chunks = host.items },
        .{ .name = "every-frame", .chunks = every },
        .{ .name = "gate-lengths", .chunks = &.{ 1, 1, 5, 121, frames - 128 } },
        .{ .name = "one-call", .chunks = &.{frames} },
        .{ .name = "irregular", .chunks = irregular.items },
    };
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-codec-stream-oracle CHECKPOINT CODES_U32LE OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var bytes: [frames * 10 * 4 + 1]u8 align(@alignOf(u32)) = undefined;
    const input = try std.Io.Dir.cwd().openFile(io, args[2], .{});
    defer input.close(io);
    if (try input.readPositionalAll(io, &bytes, 0) != frames * 10 * 4) return error.Expected1024Frames;
    const code_bytes = bytes[0 .. frames * 10 * 4];
    const codes = std.mem.bytesAsSlice([10]u32, @as([]align(@alignOf(u32)) u8, code_bytes));
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(code_bytes, &digest, .{});
    var dir = try std.Io.Dir.cwd().openDir(io, args[3], .{});
    defer dir.close(io);

    const pcm = try init.gpa.alloc(f32, frames * samples_per_frame);
    defer init.gpa.free(pcm);
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var codec = try drama.codec.Model.init(init.gpa, io, &ctx, args[1], "", .{ .max_frames = frames });
    defer codec.deinit();

    var name: [64]u8 = undefined;
    for (lengths) |n| {
        if (try codec.decode(stream, codes[0..n], pcm) != n * samples_per_frame) return error.UnexpectedSampleCount;
        const file = try std.fmt.bufPrint(&name, "whole-{d}.f32", .{n});
        try dir.writeFile(io, .{ .sub_path = file, .data = std.mem.sliceAsBytes(pcm[0 .. n * samples_per_frame]), .flags = .{ .exclusive = true } });
    }
    const plan = try schedules(a);
    for (plan) |s| {
        @memset(pcm, std.math.nan(f32));
        codec.begin();
        var end: usize = 0;
        const began = std.Io.Clock.awake.now(io).toNanoseconds();
        for (s.chunks) |n| {
            for (codec.acts) |b| try b.fill8(0xff, stream.handle); // FP16 0xffff is NaN
            try codec.audio_dev.fill8(0xff, stream.handle);
            end += n;
            if (try codec.advance(stream, codes[0..end], pcm) != end * samples_per_frame) return error.UnexpectedSampleCount;
        }
        if (end != frames) return error.IncompleteSchedule;
        const elapsed = std.Io.Clock.awake.now(io).toNanoseconds() - began;
        const file = try std.fmt.bufPrint(&name, "chunked-{s}.f32", .{s.name});
        try dir.writeFile(io, .{ .sub_path = file, .data = std.mem.sliceAsBytes(pcm), .flags = .{ .exclusive = true } });
        std.debug.print("schedule={s} chunks={d} elapsed_s_including_poison={d:.6}\n", .{ s.name, s.chunks.len, @as(f64, @floatFromInt(elapsed)) / 1e9 });
    }

    var manifest: std.ArrayList(u8) = .empty;
    try manifest.print(a, "{{\"codes_sha256\": \"{s}\", \"frames\": {d}, \"lengths\": [1, 2, 7, 128, 1024], \"schedules\": [", .{ std.fmt.bytesToHex(digest, .lower), frames });
    for (plan, 0..) |s, i| {
        try manifest.print(a, "{s}\n  {{\"name\": \"{s}\", \"chunks\": [", .{ if (i == 0) "" else ",", s.name });
        for (s.chunks, 0..) |n, j| try manifest.print(a, "{s}{d}", .{ if (j == 0) "" else ", ", n });
        try manifest.appendSlice(a, "]}");
    }
    try manifest.appendSlice(a, "\n]}\n");
    try dir.writeFile(io, .{ .sub_path = "manifest.json", .data = manifest.items, .flags = .{ .exclusive = true } });
    return 0;
}
