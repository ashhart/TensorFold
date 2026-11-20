const std = @import("std");
const fixture = @import("checkpoint_fixture");
const st = @import("safetensors");

fn byteCount(width: usize, height: usize, frames: usize) !usize {
    if (width == 0 or height == 0 or frames == 0) return error.EmptyFrames;
    const pixels = try std.math.mul(usize, width, height);
    const bytes = try std.math.mul(usize, try std.math.mul(usize, pixels, 3), frames);
    if (bytes > 512 * 1024 * 1024) return error.FramesTooLarge;
    return bytes;
}

fn pack(a: std.mem.Allocator, raw: []const u8, width: usize, height: usize, frames: usize) ![]u8 {
    if (raw.len != try byteCount(width, height, frames)) return error.RawFrameLength;
    return fixture.image(a, &.{.{ .name = "frames", .shape = &.{ frames, height, width, 3 }, .bytes = raw }});
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 2 and std.mem.eql(u8, args[1], "--help")) {
        std.debug.print("tf-sam3-frames INPUT.rgb OUTPUT.safetensors WIDTH HEIGHT FRAME_COUNT\nPacks tightly interleaved RGB24 without changing pixels; refuses existing output and inputs above 512 MiB.\n", .{});
        return;
    }
    if (args.len != 6) return error.ExpectedRawOutputWidthHeightCount;
    const width = try std.fmt.parseInt(usize, args[3], 10);
    const height = try std.fmt.parseInt(usize, args[4], 10);
    const frames = try std.fmt.parseInt(usize, args[5], 10);
    const expected = try byteCount(width, height, frames);
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(expected + 1));
    const bytes = try pack(a, raw, width, height, frames);
    const file = try std.Io.Dir.cwd().createFile(init.io, args[2], .{ .exclusive = true });
    defer file.close(init.io);
    try file.writeStreamingAll(init.io, bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(raw, &digest, .{});
    std.debug.print("frames [{d},{d},{d},3], RGB24 SHA256 {s}\n", .{ frames, height, width, std.fmt.bytesToHex(digest, .lower) });
}

test "RGB24 packing preserves frame order, channels and safetensors geometry" {
    const a = std.testing.allocator;
    const raw = [_]u8{ 1, 2, 3, 4, 5, 6, 201, 202, 203, 254, 255, 0 };
    const bytes = try pack(a, &raw, 2, 1, 2);
    defer a.free(bytes);
    const header_len: usize = @intCast(std.mem.readInt(u64, bytes[0..8], .little));
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const header = try st.parseHeader(arena.allocator(), bytes[8..][0..header_len], bytes.len - 8 - header_len);
    const frames = header.get("frames").?;
    try std.testing.expectEqual(st.DType.u8, frames.dtype);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1, 2, 3 }, frames.shape[0..frames.rank]);
    try std.testing.expectEqualSlices(u8, &raw, bytes[8 + header_len ..]);
}

test "RGB24 packing refuses empty, overflowing, truncated and extra data" {
    const a = std.testing.allocator;
    try std.testing.expectError(error.EmptyFrames, pack(a, &.{}, 0, 1, 1));
    try std.testing.expectError(error.Overflow, byteCount(std.math.maxInt(usize), 2, 1));
    try std.testing.expectError(error.FramesTooLarge, byteCount(16384, 16384, 1));
    try std.testing.expectError(error.RawFrameLength, pack(a, &.{ 1, 2 }, 1, 1, 1));
    try std.testing.expectError(error.RawFrameLength, pack(a, &.{ 1, 2, 3, 4 }, 1, 1, 1));
}
