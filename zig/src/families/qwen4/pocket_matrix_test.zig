//! Widened checkpoint bytes, not floating reconstructions, define the immutable base projections.
const std = @import("std");
const st = @import("core").safetensors;
const cfg = @import("config.zig");
const pm = @import("pocket_matrix.zig");
const a = std.testing.allocator;

test "Qwen4 learning: packed gate/up base rows and down base columns retain every original byte" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/native-dense.json"), &p);
    defer c.deinit();
    c.learned = 64;
    c.quant.mode = .mlx;
    c.quant.default = .{ .format = .mlx4, .group = 64 };
    var arena: std.heap.ArenaAllocator = .init(a);
    defer arena.deinit();
    const mem = arena.allocator();
    inline for (std.enums.values(pm.Role)) |r| {
        const rows = if (r == .down) c.hidden else c.intermediate + c.learned;
        const width = if (r == .down) c.intermediate + c.learned else c.hidden;
        const name = "model.layers.0.mlp." ++ (if (r == .gate) "gate_proj" else if (r == .up) "up_proj" else "down_proj");
        var parts: [3]st.Tensor = undefined;
        for (&parts, 0..) |*part, index| {
            const div: usize = if (index == 0) 8 else 64;
            const dtype: st.DType = if (index == 0) .u32 else .bf16;
            const stride = width / div * dtype.size();
            const data = try mem.alloc(u8, rows * stride);
            for (data, 0..) |*v, i| v.* = @truncate(i *% 31 +% index * 19);
            const saved = try mem.dupe(u8, data);
            part.* = .{ .dtype = dtype, .rank = 2, .shape = .{ rows, width / div, 1, 1, 1 }, .bytes = data };
            const base = try pm.base(mem, &c, name, part.*);
            const out_rows = if (r == .down) rows else c.intermediate;
            const out_stride = if (r == .down) c.intermediate / div * dtype.size() else stride;
            try std.testing.expectEqual(out_rows * out_stride, base.len);
            for (0..out_rows) |row| try std.testing.expectEqualSlices(u8, saved[row * stride ..][0..out_stride], base[row * out_stride ..][0..out_stride]);
            try std.testing.expectEqualSlices(u8, saved, data);
            var broken = part.*;
            broken.shape[1] += 1;
            try std.testing.expectError(error.Qwen4PocketShape, pm.base(mem, &c, name, broken));
            broken = part.*;
            broken.bytes = data[0 .. data.len - 1];
            try std.testing.expectError(error.Qwen4PocketShape, pm.base(mem, &c, name, broken));
        }
        const codes = @constCast(parts[0].bytes);
        @memset(codes, 0x11);
        for (parts[1..]) |part| {
            const bytes = @constCast(part.bytes);
            var i: usize = 0;
            while (i < bytes.len) : (i += 2) std.mem.writeInt(u16, bytes[i..][0..2], if (bytes.ptr == parts[1].bytes.ptr) 0x3f80 else 0, .little);
        }
        const decoded = try mem.alloc(f32, c.learned * c.hidden);
        try pm.decode(&c, r, parts, decoded);
        for (decoded) |value| try std.testing.expectEqual(@as(f32, 1), value);
        std.mem.writeInt(u16, @constCast(parts[1].bytes)[parts[1].bytes.len - 2 ..][0..2], 0x7fc0, .little);
        try std.testing.expectError(error.Qwen4PocketNonfinite, pm.decode(&c, r, parts, decoded));
    }
}

test "Qwen4 learning: an untaught tensor returns its original view without allocation or a format restriction" {
    var p: cfg.Problem = .{};
    var c = try cfg.parse(a, @embedFile("fixtures/native-dense.json"), &p);
    defer c.deinit();
    const bytes = [_]u8{ 0x81, 0, 2, 3 };
    const tensor: st.Tensor = .{ .dtype = .bf16, .rank = 2, .shape = .{ 1, 2, 1, 1, 1 }, .bytes = &bytes };
    const out = try pm.base(std.testing.failing_allocator, &c, "model.layers.0.mlp.down_proj", tensor);
    try std.testing.expectEqual(bytes[0..].ptr, out.ptr);
    try std.testing.expectEqualSlices(u8, &bytes, out);
}
