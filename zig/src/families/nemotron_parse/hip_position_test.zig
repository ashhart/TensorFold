//! Position interpolation against the actual pinned HF helper, independently generated before native execution.
const std = @import("std");
const hip = @import("hip");
const bf = @import("core").bf16_round;
const a = std.testing.allocator;
const io = std.testing.io;
const Cell = struct { height: usize, width: usize, gh: usize, gw: usize };
const Manifest = struct { hidden: usize, prefix: usize, cells: []Cell };

fn read(dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
}

fn upload(r: *const hip.Runtime, dir: []const u8, name: []const u8) !hip.DeviceBuffer {
    const bytes = try read(dir, name);
    defer a.free(bytes);
    return hip.DeviceBuffer.fromHost(r, bytes);
}

test "Parse HIP positions match HF content-normalized geometry (TF_PARSE_POSITION_DIR)" {
    const dir = @import("sys").getenv("TF_PARSE_POSITION_DIR") orelse return error.SkipZigTest;
    const text = try read(dir, "manifest.json");
    defer a.free(text);
    const doc = try std.json.parseFromSlice(Manifest, a, text, .{});
    defer doc.deinit();
    const m = doc.value;
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var module = try hip.Module.load(&r, @import("parse_gpu").bytes);
    defer module.unload();
    const k = try hip.vision.Kernels.load(module);
    var table = try upload(&r, dir, "table.bin");
    defer table.free();
    var cls = try upload(&r, dir, "cls.bin");
    defer cls.free();
    for (m.cells, 0..) |cell, index| {
        var name: [64]u8 = undefined;
        var patches = try upload(&r, dir, try std.fmt.bufPrint(&name, "patches-{d}.bin", .{index}));
        defer patches.free();
        const oh = (cell.height + 15) / 16;
        const ow = (cell.width + 15) / 16;
        var field = try hip.DeviceBuffer.alloc(&r, oh * ow * m.hidden * 4);
        defer field.free();
        var mid = try hip.DeviceBuffer.alloc(&r, 128 * ow * m.hidden * 4);
        defer mid.free();
        const count = (cell.gh * cell.gw + m.prefix) * m.hidden;
        var out = try hip.DeviceBuffer.alloc(&r, count * 2);
        defer out.free();
        const table_tensor: hip.ops.Tensor = .{ .ptr = table.base(), .kind = .bf16 };
        if (@min(oh, ow) == 1) {
            try k.resize(s, table_tensor, field.base(), .{ 128, 128 }, .{ oh, ow }, m.hidden, .area);
        } else {
            try k.resize(s, table_tensor, mid.base(), .{ 128, 128 }, .{ 128, ow }, m.hidden, .horizontal);
            try k.resize(s, .{ .ptr = mid.base(), .kind = .f32 }, field.base(), .{ 128, ow }, .{ oh, ow }, m.hidden, .vertical);
        }
        try k.position(s, .{ .ptr = patches.base(), .kind = .bf16 }, cls.base(), field.base(), .{ .ptr = out.base(), .kind = .bf16 }, .{ cell.gh, cell.gw }, .{ cell.height, cell.width }, m.hidden, m.prefix);
        try s.synchronize();
        const want = try read(dir, try std.fmt.bufPrint(&name, "expected-{d}.bin", .{index}));
        defer a.free(want);
        const got = try a.alloc(u16, count);
        defer a.free(got);
        try out.download(0, std.mem.sliceAsBytes(got));
        try std.testing.expectEqual(count * 2, want.len);
        var differences: usize = 0;
        var worst: f64 = 0;
        for (got, 0..) |v, i| {
            const expected = std.mem.readInt(u16, want[i * 2 ..][0..2], .little);
            differences += @intFromBool(v != expected);
            worst = @max(worst, @abs(@as(f64, bf.value(v)) - bf.value(expected)));
        }
        std.debug.print("Parse positions {d}x{d}: {d}/{d} byte differences, max abs {d:.9}\n", .{ cell.height, cell.width, differences, count, worst });
        try std.testing.expectEqual(@as(usize, 0), differences);
    }
}
