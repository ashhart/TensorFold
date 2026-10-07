//! One token through the captured _embed cubin, compared with the host dequant of the same row.

const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");

const embed = flash.embed;

fn shardPath(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) ![]u8 {
    const index = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(index);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, index, gpa, .limited(1 << 26));
    defer gpa.free(text);
    const parsed = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer parsed.deinit();
    const file = parsed.value.object.get("weight_map").?.object.get(embed.weight_name).?.string;
    return std.fs.path.join(gpa, &.{ dir, file });
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 4) {
        std.debug.print("usage: flashnext-embed <model-dir> <kernel-dir> <token>\n", .{});
        return 2;
    }
    const token = try std.fmt.parseInt(u32, args[3], 10);
    const path = try shardPath(gpa, io, args[1]);
    defer gpa.free(path);
    var mapped = try embed.Mapped.open(gpa, io, path);
    defer mapped.close();
    const weight = mapped.weight;
    const scales = mapped.scales;
    const biases = mapped.biases;

    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var table = try embed.Table.upload(&driver, weight, scales, biases);
    defer table.deinit();
    if (token >= table.rows) return error.UnexpectedTensor;
    const host = try gpa.alloc(u16, table.dims);
    defer gpa.free(host);
    try embed.dequant(weight.bytes, scales.bytes, biases.bytes, table.dims, token, host);

    var set = try cuda.aot.Set.load(gpa, io, &driver, ctx.device, args[2]);
    defer set.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var ids = try cuda.DeviceBuffer.fromHost(&driver, std.mem.asBytes(&token));
    defer ids.free();
    var out = try cuda.DeviceBuffer.alloc(&driver, table.dims * 2);
    defer out.free();
    try flash.Tri.embed(.{ .set = &set, .s = stream }, ids.ptr, table.w.ptr, table.s.ptr, table.b.ptr, out.ptr, table.dims, 1, 1);

    const gpu = try gpa.alloc(u8, table.dims * 2);
    defer gpa.free(gpu);
    try stream.synchronize();
    try out.download(0, gpu);

    var mismatch: usize = 0;
    for (host, 0..) |want, i| {
        const got = std.mem.readInt(u16, gpu[2 * i ..][0..2], .little);
        if (got != want) mismatch += 1;
    }
    std.debug.print("token {d} rows {d} dims {d} mismatch {d} head", .{ token, table.rows, table.dims, mismatch });
    for (0..@min(8, table.dims)) |i| std.debug.print(" {x:0>4}", .{std.mem.readInt(u16, gpu[2 * i ..][0..2], .little)});
    std.debug.print("\n", .{});
    if (mismatch != 0) return 1;

    const streams: usize = 4;
    var wide = try cuda.DeviceBuffer.alloc(&driver, streams * table.dims * 2);
    defer wide.free();
    try flash.Tri.embed(.{ .set = &set, .s = stream }, ids.ptr, table.w.ptr, table.s.ptr, table.b.ptr, wide.ptr, table.dims, streams, 1);
    const wide_bytes = try gpa.alloc(u8, streams * table.dims * 2);
    defer gpa.free(wide_bytes);
    try stream.synchronize();
    try wide.download(0, wide_bytes);
    var copies_differ = false;
    for (1..streams) |s| {
        if (!std.mem.eql(u8, wide_bytes[0 .. table.dims * 2], wide_bytes[s * table.dims * 2 ..][0 .. table.dims * 2])) copies_differ = true;
    }
    std.debug.print("embed copies {d} identical {}\n", .{ streams, !copies_differ });
    return if (copies_differ) 1 else 0;
}
