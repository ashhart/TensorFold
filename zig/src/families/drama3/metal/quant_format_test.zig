const std = @import("std");
const st = @import("core").safetensors;
const q = @import("quant_format.zig");
const ref = @import("quant_reference.zig");
const cfg = @import("../config.zig");

fn tensor(dtype: st.DType, shape: []const usize, bytes: []const u8) st.Tensor {
    var t: st.Tensor = .{ .dtype = dtype, .rank = @intCast(shape.len), .shape = @splat(1), .bytes = bytes };
    @memcpy(t.shape[0..shape.len], shape);
    return t;
}

test "NVFP4 rejects shape dtype negative nan scales and invalid globals" {
    const codes: [32]u8 = @splat(0x98);
    var scales: [4]u8 = @splat(0x38);
    var global: f32 = 1;
    const w = tensor(.u8, &.{ 1, 32 }, &codes);
    const s = tensor(.f8_e4m3, &.{ 1, 4 }, &scales);
    const g = tensor(.f32, &.{}, std.mem.asBytes(&global));
    try q.nvfp4(w, s, g, 1, 64);
    for ([_]u8{ 0x7f, 0x81, 0xff }) |bad| {
        scales[0] = bad;
        try std.testing.expectError(error.BadNvfp4Scale, q.nvfp4(w, s, g, 1, 64));
    }
    for ([_]u8{ 0, 0x80 }) |zero| {
        scales[0] = zero;
        try q.nvfp4(w, s, g, 1, 64);
    }
    for ([_]f32{ 0, -1, std.math.inf(f32), std.math.nan(f32) }) |bad| {
        global = bad;
        try std.testing.expectError(error.BadScale, q.nvfp4(w, s, g, 1, 64));
    }
    global = 1;
    try std.testing.expectError(error.BadNvfp4Shape, q.nvfp4(tensor(.f8_e4m3, &.{ 1, 32 }, &codes), s, g, 1, 64));
    try std.testing.expectError(error.BadNvfp4Shape, q.nvfp4(w, tensor(.f8_e4m3, &.{ 4, 1 }, &scales), g, 1, 64));
    try std.testing.expectError(error.BadNvfp4Shape, q.nvfp4(w, s, tensor(.f32, &.{1}, std.mem.asBytes(&global)), 1, 64));
    try std.testing.expectError(error.BadNvfp4Shape, q.nvfp4(tensor(.u8, &.{ 1, 32 }, codes[0..31]), s, g, 1, 64));
    try std.testing.expectError(error.BadNvfp4Shape, q.nvfp4(w, s, tensor(.f32, &.{}, &.{}), 1, 64));
}

fn mapText(a: std.mem.Allocator, ids: []const u8, omit: bool) ![]u8 {
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(a);
    try text.append(a, '{');
    for (0..cfg.layers - @as(usize, @intFromBool(omit))) |i| try text.print(a, "{s}\"{d}\":{s}", .{ if (i == 0) "" else ",", i, ids });
    try text.append(a, '}');
    return text.toOwnedSlice(a);
}

test "expert metadata is complete ascending unique and keeps at least eight original IDs" {
    const a = std.testing.allocator;
    const good = "[1,3,5,7,9,11,13,255]";
    for ([_][]const u8{ good, "[1,3,5,7,9,11,13,13]", "[3,1,5,7,9,11,13,15]", "[1,3,5,7,9,11,13,256]", "[1,3]", "[1,3,5,7,9,11,13,1.5]" }, 0..) |ids, i| {
        const text = try mapText(a, ids, false);
        defer a.free(text);
        const tree = try std.json.parseFromSlice(std.json.Value, a, text, .{});
        defer tree.deinit();
        if (i == 0) {
            const h = try q.parseHeld(tree.value);
            const slots = h.slots(0);
            try std.testing.expectEqual(@as(i32, -1), slots[0]);
            try std.testing.expectEqual(@as(i32, 7), slots[255]);
            var names = std.StringHashMapUnmanaged(void).empty;
            defer names.deinit(a);
            try names.put(a, "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", {});
            try std.testing.expectError(error.UnlistedExpert, q.checkNames(&h, names.keyIterator()));
        } else try std.testing.expectError(error.BadExpertMap, q.parseHeld(tree.value));
    }
    const text = try mapText(a, good, true);
    defer a.free(text);
    const tree = try std.json.parseFromSlice(std.json.Value, a, text, .{});
    defer tree.deinit();
    try std.testing.expectError(error.BadExpertMap, q.parseHeld(tree.value));
}

test "independent scalar reference covers E2M1 signs and E4M3 boundary scales" {
    const values = [_]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    for (values, 0..) |v, i| {
        try std.testing.expectEqual(v, ref.fp4(@intCast(i)));
        try std.testing.expectEqual(-v, ref.fp4(@intCast(i + 8)));
    }
    try std.testing.expectEqual(@as(f64, 1.0 / 512.0), ref.fp8(1));
    try std.testing.expectEqual(@as(f64, 7.0 / 512.0), ref.fp8(7));
    try std.testing.expectEqual(@as(f64, 1.0 / 64.0), ref.fp8(8));
    try std.testing.expectEqual(@as(f64, 1), ref.fp8(0x38));
    try std.testing.expectEqual(@as(f64, 448), ref.fp8(0x7e));
    for (0..127) |code| try std.testing.expectEqual(-ref.fp8(@intCast(code)), ref.fp8(@intCast(code + 128)));
    try std.testing.expectEqual(@as(f64, 1), ref.roundBf16(1.00390625));
    try std.testing.expectEqual(@as(f64, 1.015625), ref.roundBf16(1.01171875));
}
