//! Layout guards: constants copied from each engine are checked against its source text so drift fails loudly.
const std = @import("std");
const options = @import("handoff_options");
const layout = @import("../kv/layout.zig");
const kv_state = @import("../kv_state.zig");
const g = @import("geometry.zig");
const metal = @import("metal.zig");
const plan = @import("plan.zig");

fn has(comptime text: []const u8, comptime needle: []const u8) !void {
    if (std.mem.indexOf(u8, text, needle) == null) {
        std.debug.print("layout guard: expected `{s}`\n", .{needle});
        return error.LayoutChanged;
    }
}

const Cfg = struct {
    layers: u32 = g.release.layers,
    pub fn isKvSource(_: Cfg, layer: u32) bool {
        return g.release.ratioOf(layer) != null;
    }
    pub fn compressRatio(_: Cfg, layer: u32) u32 {
        return g.release.ratioOf(layer).?;
    }
};

test "CUDA's cache families: record sizes, page, KV sources and ratios" {
    try std.testing.expectEqual(@as(u32, g.cuda_record_bytes), layout.compRowBytes(.fp8));
    try std.testing.expectEqual(@as(u32, g.cuda_index_fp8_bytes), layout.indexRowBytes(.fp8));
    try std.testing.expectEqual(@as(u32, g.cuda_index_bf16_bytes), layout.indexRowBytes(.bf16));
    try std.testing.expectEqual(@as(u32, g.release.page), (layout.Options{}).page);
    const l = try layout.Layout.fromConfig(Cfg{}, .{});
    try std.testing.expectEqual(@as(u32, g.release.sources.len), l.nsources);
    for (g.release.sources, 0..) |s, i| try std.testing.expectEqual(s.layer, l.sources[i]);
}

test "the engine's record geometry: block.zig's row and index bytes, the rings, the carry and Engram lookback" {
    const block = @embedFile("../block.zig");
    try has(block, "pub const row_bytes: i64 = 584;");
    try has(block, "pub const ik_bytes: i64 = 132;");
    try has(block, "pub const page: i64 = 256;");
    const kv = @embedFile("../kv/sess.zig");
    try has(kv, "pub const row_bytes: u64 = 584;");
    try has(kv, "pub const lookback: u64 = 3;");
    try std.testing.expectEqual(@as(u32, 3), g.release.lookback);
    try std.testing.expectEqual(@as(u32, 128), g.release.window);
}

test "main's Metal lane (metal/kv.zig): window mirror, 68-byte index keys, bf16 pool rows, carry slots" {
    const kv = @embedFile("../metal/kv.zig");
    try has(kv, "return c.window + pipelines.max_rows;");
    try has(kv, "pub const index_key_bytes = 68;");
    try has(kv, "const wbytes = 2 * @as(usize, windowSlots(c)) * c.head_dim * 2;");
    try has(@embedFile("../metal/pipelines.zig"), "pub const max_rows = 16;");
    const rel: metal.Release = .{ .layers = &.{} };
    try std.testing.expectEqual(@as(u32, 128 + 16), rel.window_slots);
    try std.testing.expectEqual(@as(u32, 128), rel.ds_window);
}

test "the served lane's physical formats (kv_state.zig's encodings, PR 2373's strides)" {
    // the encodings' names carry their row bytes
    comptime std.debug.assert(@hasField(kv_state.Encoding, "cuda_fp8_584") and @hasField(kv_state.Encoding, "cuda_index_fp8_132"));
    comptime std.debug.assert(@hasField(kv_state.Encoding, "metal_fp4_288") and @hasField(kv_state.Encoding, "metal_index_fp4_68"));
    try std.testing.expectEqual(@as(usize, 288), metal.Packed.main_stride);
    try std.testing.expectEqual(@as(usize, 68), metal.Packed.index_stride);
    try std.testing.expectEqual(@as(usize, 528), metal.Packed.ring_stride);
    try std.testing.expectEqual(@as(usize, 4), metal.Packed.owners.len);
    for (metal.Packed.owners, metal.Packed.ratios, g.release.sources) |owner, ratio, src| {
        try std.testing.expectEqual(src.layer, owner);
        try std.testing.expectEqual(src.ratio, ratio);
    }
}

test "the neutral format's constants: any change to them is a new format" {
    try std.testing.expectEqual(@as(u64, 2 << 20), plan.chunk_bytes);
    try std.testing.expectEqual(@as(u32, 40), g.release.layers);
    try std.testing.expectEqual(@as(u32, 3), g.release.mtp_layers);
    try std.testing.expectEqual(@as(u32, 512), g.release.head_dim);
    try std.testing.expectEqual(@as(u32, 64), g.release.rope_dim);
    try std.testing.expectEqual(@as(u32, 128), g.release.index_dim);
    try std.testing.expectEqual(@as(usize, 4), g.release.sources.len);
}

test "PR 2373's metal_kv_state.zig, when this branch carries it" {
    if (comptime !options.exl3_kv) return error.SkipZigTest;
    const src = @embedFile("../metal_kv_state.zig");
    try has(src, "pub const main_stride = 288;");
    try has(src, "pub const index_stride = 68;");
    try has(src, "pub const ring_stride = 528;");
    try has(src, "pub const window = 128;");
    try has(src, "pub const owners = [_]u32{ 2, 8, 14, 20 };");
    try has(src, "pub const ratios = [_]u32{ 2, 2, 2, 1 };");
    try has(src, "pub const capture_rows: u32 = 8;");
    try has(src, "pub const capture_width: u32 = 512;");
    // the carry: 7 ratio-sum slots of (values, gates) f32 rows of 512
    try has(src, "return 7 * 2 * 512 * 4;");
    try has(src, "partial_off += 2 * r * 512 * 4;");
    try std.testing.expectEqual(@as(usize, metal.Packed.partial_bytes), 7 * 2 * 512 * 4);
    try std.testing.expectEqual(@as(usize, metal.Packed.capture_bytes), 8 * 2 * 512 * 4);
}
