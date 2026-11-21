//! Native teacher-forced Inkling 2 Mini logits; one raw little-endian F32 vocabulary row per input token.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) return error.ExpectedModelTokensOutputChunk;
    const chunk = try std.fmt.parseInt(u32, args[4], 10);
    if (chunk == 0) return error.InvalidChunkSize;
    comptime std.debug.assert(@import("builtin").target.cpu.arch.endian() == .little);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cwd = std.Io.Dir.cwd();
    const bytes = try cwd.readFileAlloc(init.io, args[2], init.gpa, .limited(4 << 20));
    defer init.gpa.free(bytes);
    const a = try tf.npy.parse(bytes);
    if (!std.mem.eql(u8, a.descr, "<u4") or a.rank != 1 or a.count() == 0) return error.ExpectedU32TokenVector;
    const owned_ids = if (@intFromPtr(a.data.ptr) % @alignOf(u32) != 0) try init.gpa.alloc(u32, a.count()) else null;
    defer if (owned_ids) |owned| init.gpa.free(owned);
    const ids: []const u32 = if (owned_ids) |owned| blk: {
        @memcpy(std.mem.sliceAsBytes(owned), a.data);
        break :blk owned;
    } else std.mem.bytesAsSlice(u32, @as([]align(@alignOf(u32)) const u8, @alignCast(a.data)));
    const rows = @min(chunk, @as(u32, @intCast(ids.len)));
    const model = try tf.inkling2_metal.Model.load(init.gpa, init.io, args[1], .{ .capacity = @intCast(ids.len), .rows = rows });
    defer model.deinit();
    var state = try model.newState();
    defer state.deinit();
    const output = try cwd.createFile(init.io, args[3], .{});
    defer output.close(init.io);
    const vocab = model.config.text.unpadded_vocab;
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(ids.len - at, rows);
        try model.forward(&.{.{ .state = &state, .ids = ids[at..][0..n], .logit_rows = @intCast(n) }});
        try output.writeStreamingAll(init.io, model.logits.contents()[0 .. n * vocab * @sizeOf(f32)]);
        at += n;
    }
    std.debug.print("native Inkling 2 Mini forward: {d} rows x {d} vocabulary, little-endian f32\n", .{ ids.len, vocab });
}
