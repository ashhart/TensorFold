//! The MTP head's load: its layer as plain DSA + MoE, norms and eh_proj, and the draft head from the head's rows.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const formats = @import("cuda_formats.zig");
const weights = @import("cuda_weights.zig");
const quant = @import("cuda_quant.zig");
const q4 = @import("cuda_q4.zig");

const prefix = "model.language_model.";

/// weights.load's MTPW: layer `layers` as plain DSA + MoE, norms, eh_proj, draft head from the rank's head rows.
pub fn load(L: anytype, head_ck: *core.Checkpoint) !weights.Mtp {
    const c = &L.w.config;
    const i = c.layers;
    try L.whole(try L.get("layers.{d}.input_layernorm.weight", .{i}), .bf16);
    const in_norm = try L.put("layers.{d}.in_norm", .{i});
    try L.whole(try L.get("layers.{d}.post_attention_layernorm.weight", .{i}), .bf16);
    const post_norm = try L.put("layers.{d}.post_norm", .{i});
    const none: weights.Hc = .{ .fn_w = 0, .base = 0, .scale = 0 };
    const layer_w: weights.Layer = .{ .index = i, .attn_hc = none, .ffn_hc = none, .in_norm = in_norm, .post_norm = post_norm, .mixer = .{ .mla = try L.dsa(i) }, .ffn = .{ .moe = try L.moeLayer(i) } };
    L.ck.release();
    var out: [4]u64 = undefined;
    for (&out, [_][]const u8{ "enorm.weight", "hnorm.weight", "eh_proj.weight", "shared_head.norm.weight" }, [_][]const u8{ "enorm", "hnorm", "eh", "norm" }) |*p, name, what| {
        try L.whole(try L.ck.get(try std.fmt.bufPrint(&L.name_buf, prefix ++ "layers.{d}.{s}", .{ i, name })), .bf16);
        p.* = try L.put("mtp.{s}", .{what});
    }
    // the draft head: quantize4 of the head's bf16 rows (an MXFP8 head's widened exactly), then make_q4's tiling
    const qk = L.tools.quant orelse return error.NoQuantizer;
    const tk = L.tools.q4 orelse return error.NoQuantizer;
    const n = L.w.share.vocab.len();
    const k: usize = c.hidden;
    var raw: [3]cuda.DeviceBuffer = undefined;
    var made: usize = 0;
    defer for (raw[0..made]) |*b| b.free();
    for (&raw, [_]usize{ n * k / 2, n * k / 32, n * k / 32 }) |*b, size| {
        b.* = try L.scratch(size);
        made += 1;
    }
    var wide: ?cuda.DeviceBuffer = null;
    defer if (wide) |*b| b.free();
    const rows: u64 = switch (L.w.head) {
        .bf16 => |p| p,
        .mxfp8 => r: {
            wide = try formats.mxfp8.wideRows(L.d, L.gpa, head_ck, "lm_head", L.w.share.vocab.lo, L.w.share.vocab.hi);
            break :r wide.?.ptr;
        },
        .fp8g => return error.UnreadableFormat, // weights.load reads a head as bf16 or MXFP8 only
    };
    try quant.q4(qk, L.stream, rows, n, k, raw[0].ptr, raw[1].ptr, raw[2].ptr);
    var tiled: [3]u64 = undefined;
    for (&tiled, q4.bytes(n, k), [_][]const u8{ "words", "scales", "biases" }) |*p, size, what| {
        const b = try cuda.DeviceBuffer.alloc(L.d, size);
        try L.w.buffers.append(L.gpa, b);
        try L.w.named.append(L.gpa, .{ .name = try std.fmt.allocPrint(L.gpa, "draft_head.{s}", .{what}), .ptr = b.ptr, .len = b.len });
        p.* = b.ptr;
    }
    const draft: q4.Q4 = .{ .w = tiled[0], .s = tiled[1], .b = tiled[2], .n = n, .k = k, .npad = q4.npadOf(n) };
    try q4.pack(tk, L.stream, raw[0].ptr, raw[1].ptr, raw[2].ptr, draft);
    try L.stream.synchronize();
    return .{ .enorm = out[0], .hnorm = out[1], .eh = out[2], .norm = out[3], .layer = layer_w, .draft = draft };
}
