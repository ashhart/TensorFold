const std = @import("std");
const xpu = @import("xpu");
const cfg = @import("config.zig");
const model = @import("xpu_model.zig");

const conv_bytes = 6 * cfg.hidden * 2;
const slot_bytes = cfg.kv_heads * cfg.head_dim * 2;

fn fill(buf: xpu.rt.Buffer, seed: u8) void {
    for (buf.ptr.?[0..buf.len], 0..) |*byte, i| byte.* = seed +% @as(u8, @truncate(i * 31 + 7));
}

fn expectFill(buf: xpu.rt.Buffer, bytes: usize, seed: u8) !void {
    for (buf.ptr.?[0..bytes], 0..) |byte, i| try std.testing.expectEqual(seed +% @as(u8, @truncate(i * 31 + 7)), byte);
}

test "LFM3 snapshot resume restores logical state and retires the prior window carry" {
    const a = std.testing.allocator;
    var runtime: xpu.rt.Runtime = .{ .gpa = a };
    var m: model.Model = undefined;
    var layers: [3]@TypeOf(m.layers[0]) = undefined;
    layers[0].mixer = .{ .conv = undefined };
    layers[1].mixer = .{ .attn = undefined };
    layers[1].mixer.attn.ring = 8;
    layers[1].mixer.attn.window = 0;
    layers[2].mixer = .{ .attn = undefined };
    layers[2].mixer.attn.ring = 5;
    layers[2].mixer.attn.window = cfg.sliding_window;
    var q: model.Seq = .{ .pos = 3 };
    m.gpa = a;
    m.r = &runtime;
    m.layers = &layers;
    m.seq = &q;
    m.pos = 3;

    q.conv[0] = try runtime.alloc(conv_bytes);
    defer q.conv[0].free();
    q.ext[0] = try runtime.alloc((6 + model.window_rows) * cfg.hidden * 2);
    defer q.ext[0].free();
    q.kc[1] = try runtime.alloc(8 * slot_bytes);
    defer q.kc[1].free();
    q.vc[1] = try runtime.alloc(8 * slot_bytes);
    defer q.vc[1].free();
    q.kc[2] = try runtime.alloc(5 * slot_bytes);
    defer q.kc[2].free();
    q.vc[2] = try runtime.alloc(5 * slot_bytes);
    defer q.vc[2].free();

    fill(q.conv[0], 11);
    fill(q.ext[0], 17);
    fill(q.kc[1], 23);
    fill(q.vc[1], 29);
    fill(q.kc[2], 31);
    fill(q.vc[2], 37);
    const saved = try m.snapSave();
    defer m.snapDrop(saved);

    fill(q.conv[0], 101);
    fill(q.kc[1], 103);
    fill(q.vc[1], 107);
    fill(q.kc[2], 109);
    fill(q.vc[2], 113);
    q.pos = 11;
    q.win_start = 8;
    q.win_rows = 3;
    m.pos = 11;
    try m.snapRestore(saved);

    const fresh: model.Seq = .{ .pos = 3 };
    try std.testing.expectEqual(fresh.pos, q.pos);
    try std.testing.expectEqual(fresh.win_start, q.win_start);
    try std.testing.expectEqual(fresh.win_rows, q.win_rows);
    try std.testing.expectEqual(@as(u32, 3), m.pos);
    try expectFill(q.conv[0], conv_bytes, 11);
    try expectFill(q.kc[1], 3 * slot_bytes, 23);
    try expectFill(q.vc[1], 3 * slot_bytes, 29);
    try expectFill(q.kc[2], 5 * slot_bytes, 31);
    try expectFill(q.vc[2], 5 * slot_bytes, 37);

    try m.keepRows(2);
    try std.testing.expectEqual(@as(u32, 3), m.pos);
    try std.testing.expectEqual(@as(u32, 3), q.pos);
    try expectFill(q.conv[0], conv_bytes, 11);
}
