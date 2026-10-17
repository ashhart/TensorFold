//! Decodes fixed code prefixes (1024 rows of ten u32) through the production codec to f32 PCM.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-codec-oracle CHECKPOINT CODES_U32LE OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var bytes: [1024 * 10 * 4 + 1]u8 align(@alignOf(u32)) = undefined;
    const input = try std.Io.Dir.cwd().openFile(init.io, args[2], .{});
    defer input.close(init.io);
    const count = try input.readPositionalAll(init.io, &bytes, 0);
    if (count != 1024 * 10 * 4) return error.Expected1024Frames;
    const codes = std.mem.bytesAsSlice([10]u32, @as([]align(@alignOf(u32)) u8, bytes[0..count]));
    var output_dir = try std.Io.Dir.cwd().openDir(init.io, args[3], .{});
    defer output_dir.close(init.io);
    const pcm = try init.gpa.alloc(f32, 1024 * 2048);
    defer init.gpa.free(pcm);
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var codec = try drama.codec.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .max_frames = 1024 });
    defer codec.deinit();
    for ([_]usize{ 1, 2, 7, 128, 1024 }) |frames| {
        const begin = std.Io.Clock.awake.now(init.io).toNanoseconds();
        const samples = try codec.decode(stream, codes[0..frames], pcm);
        const elapsed = std.Io.Clock.awake.now(init.io).toNanoseconds() - begin;
        if (samples != frames * 2048) return error.UnexpectedSampleCount;
        for (pcm[0..samples]) |sample| if (!std.math.isFinite(sample)) return error.NonFiniteAudio;
        var name_buffer: [64]u8 = undefined;
        const name = try std.fmt.bufPrint(&name_buffer, "native-{d}.f32", .{frames});
        try output_dir.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(pcm[0..samples]) });
        const used = cuda.usage(false);
        std.debug.print("frames={d} samples={d} elapsed_s={d:.6} cuda_bytes={d} pinned_bytes={d}\n", .{
            frames, samples, @as(f64, @floatFromInt(elapsed)) / 1e9, used.device, used.host,
        });
    }
    return 0;
}
