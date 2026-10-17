//! Teacher-forces fixed rows through the production tower; exports final hidden, last codebook head and KV cache.
const std = @import("std");
const builtin = @import("builtin");
const cuda = @import("cuda");
const drama = @import("drama");
const rows = 128;
const width = 2048;
const kv_width = 512;
const Frame = drama.tower.Frame;

fn view(buffer: cuda.DeviceBuffer, offset: usize, len: usize) cuda.DeviceBuffer {
    std.debug.assert(offset <= buffer.len and len <= buffer.len - offset);
    return .{ .d = buffer.d, .ptr = buffer.ptr + offset, .len = len, .borrowed = true };
}

fn readExact(io: std.Io, dir: std.Io.Dir, name: []const u8, bytes: []u8) !void {
    const file = try dir.openFile(io, name, .{});
    defer file.close(io);
    if (try file.length(io) != bytes.len or try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.BadFixtureLength;
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 4) {
        std.debug.print("usage: tf-drama-tower-oracle CHECKPOINT FIXTURE_DIRECTORY OUTPUT_DIRECTORY\n", .{});
        return 2;
    }
    if (builtin.cpu.arch.endian() != .little) return error.UnsupportedHostEndian;
    var fixture = try std.Io.Dir.cwd().openDir(init.io, args[2], .{});
    defer fixture.close(init.io);
    var out = try std.Io.Dir.cwd().openDir(init.io, args[3], .{});
    defer out.close(init.io);
    const hidden = try init.gpa.alloc(u16, rows * width);
    defer init.gpa.free(hidden);
    var frames: [rows + 1]Frame = undefined;
    try readExact(init.io, fixture, "hidden.bf16", std.mem.sliceAsBytes(hidden));
    try readExact(init.io, fixture, "frames.u32le", std.mem.sliceAsBytes(&frames));
    const result = try init.gpa.alloc(u16, rows * width);
    defer init.gpa.free(result);
    const repeated = try init.gpa.alloc(u16, rows * width);
    defer init.gpa.free(repeated);
    const cache_row = try init.gpa.alloc(u16, rows * kv_width);
    defer init.gpa.free(cache_row);
    var logits: [4096]u16 = undefined;
    _ = try cuda.setAllocationLimit(54 * (1 << 30));
    var driver = try cuda.Driver.open();
    defer driver.close();
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var stream = try cuda.Stream.init(&driver, true);
    defer stream.deinit();
    var tower = try drama.tower.Model.init(init.gpa, init.io, &ctx, args[1], "", .{ .context = rows, .max_rows = rows });
    defer tower.deinit();
    var device_hidden = try cuda.DeviceBuffer.alloc(&driver, rows * width * 2);
    defer device_hidden.free();
    try device_hidden.upload(0, std.mem.sliceAsBytes(hidden));
    const p = tower.impl;
    var name_buffer: [128]u8 = undefined;
    for ([_]usize{ 1, 7, rows }) |n| {
        try tower.reset(stream, &.{}, 0);
        _ = try tower.setInput(stream, frames[0..n]);
        try tower.prefill(stream, view(device_hidden, 0, n * width * 2), frames[1 .. n + 1]);
        try p.output_hidden.download(0, std.mem.sliceAsBytes(result[0 .. n * width]));
        var name = try std.fmt.bufPrint(&name_buffer, "native-{d}-hidden.bf16", .{n});
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(result[0 .. n * width]) });
        // The one-row final head uses generate()'s cuBLASLt plan and weights; no sampling.
        try p.plans.items[5].run(p.output_hidden.ptr + (n - 1) * width * 2, p.heads[8].ptr, p.head_logits.ptr, p.workspace.ptr, p.workspace.len, stream.handle);
        try stream.synchronize();
        try p.head_logits.download(0, std.mem.sliceAsBytes(&logits));
        name = try std.fmt.bufPrint(&name_buffer, "native-{d}-head9.bf16", .{n});
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(&logits) });
        for (0..13) |layer| for ([_][]const u8{ "keys", "values" }, 0..) |kind, which| {
            const offset = (layer * 2 + which) * rows * kv_width * 2;
            try p.cache.download(offset, std.mem.sliceAsBytes(cache_row[0 .. n * kv_width]));
            name = try std.fmt.bufPrint(&name_buffer, "native-{d}-layer-{d}-{s}.bf16", .{ n, layer, kind });
            try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(cache_row[0 .. n * kv_width]) });
        };
        try tower.reset(stream, &.{}, 0);
        _ = try tower.setInput(stream, frames[0..n]);
        try tower.prefill(stream, view(device_hidden, 0, n * width * 2), frames[1 .. n + 1]);
        try p.output_hidden.download(0, std.mem.sliceAsBytes(repeated[0 .. n * width]));
        if (!std.mem.eql(u16, result[0 .. n * width], repeated[0 .. n * width])) return error.TowerResetNotReproducible;
        // GEMM row counts may change low BF16 bits: chunked output is exported for comparison, not gated bitwise.
        try tower.reset(stream, &.{}, 0);
        for (0..n) |r| {
            _ = try tower.setInput(stream, frames[r .. r + 1]);
            try tower.prefill(stream, view(device_hidden, r * width * 2, width * 2), frames[r + 1 .. r + 2]);
            try p.output_hidden.download(0, std.mem.sliceAsBytes(repeated[r * width .. (r + 1) * width]));
        }
        name = try std.fmt.bufPrint(&name_buffer, "native-{d}-chunked-hidden.bf16", .{n});
        try out.writeFile(init.io, .{ .sub_path = name, .data = std.mem.sliceAsBytes(repeated[0 .. n * width]) });
        const used = cuda.usage(false);
        std.debug.print("tower_rows={d} reset_bit_exact=true cuda_bytes={d} pinned_bytes={d}\n", .{ n, used.device, used.host });
    }
    return 0;
}
