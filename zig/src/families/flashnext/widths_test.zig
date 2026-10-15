//! Per-module widths: one module under any key spelling, its own width or a refusal, never a guess from shapes.
const std = @import("std");
const st = @import("../../core/safetensors.zig");
const config = @import("config.zig");
const widths = @import("widths.zig");
const a = std.testing.allocator;
const checked_config = @embedFile("fixtures/config.json");

fn overridden(entries: []const u8) ![]u8 {
    const declaration = try std.fmt.allocPrint(a, "\"mode\": \"affine\", {s}", .{entries});
    defer a.free(declaration);
    return std.mem.replaceOwned(u8, a, checked_config, "\"mode\": \"affine\"", declaration);
}

const mixed =
    \\"language_model.model.layers.0.self_attn.v_proj": {"bits": 8, "group_size": 64},
    \\"model.layers.1.linear_attn.in_proj_z.weight": {"bits": 5, "group_size": 128},
    \\"language_model.model.layers.0.mlp.gate": false,
    \\"model.layers.2.self_attn.o_proj": {"bits": 4}
;

fn expectWidth(w: *const widths.Widths, path: []const u8, bits: u8, group: u16) !void {
    const got = (try w.of(path)).?;
    try std.testing.expectEqual(bits, got.bits);
    try std.testing.expectEqual(group, got.group);
}

test "a mixed-override config resolves each module under either prefix and never fills a missing field" {
    const bytes = try overridden(mixed);
    defer a.free(bytes);
    var c = try config.parse(a, bytes);
    defer c.deinit();
    var w = try widths.Widths.init(a, &c);
    defer w.deinit();
    try std.testing.expect(!w.uniform());
    try expectWidth(&w, "model.layers.0.self_attn.v_proj", 8, 64);
    try expectWidth(&w, "language_model.model.layers.0.self_attn.v_proj.weight", 8, 64);
    try expectWidth(&w, "language_model.model.layers.1.linear_attn.in_proj_z", 5, 128);
    try expectWidth(&w, "language_model.model.layers.1.linear_attn.in_proj_qkv", 6, 32);
    try std.testing.expect((try w.of("model.layers.0.mlp.gate")) == null);
    // the family's own reader fills 4-bit group 64 here; core/quant refuses the half-declared width
    try std.testing.expectError(error.UnsupportedQuantization, w.of("language_model.model.layers.2.self_attn.o_proj"));
}

test "two spellings of one module that disagree are refused" {
    const bytes = try overridden(
        \\"language_model.model.layers.0.self_attn.v_proj": {"bits": 8, "group_size": 64},
        \\"model.layers.0.self_attn.v_proj.weight": {"bits": 4, "group_size": 64}
    );
    defer a.free(bytes);
    var c = try config.parse(a, bytes);
    defer c.deinit();
    try std.testing.expectError(error.ConflictingAffineAlias, widths.Widths.init(a, &c));
}

fn entry(dtype: st.DType, rows: usize, cols: usize) st.Entry {
    var shape: [st.max_rank]usize = @splat(0);
    shape[0] = rows;
    shape[1] = cols;
    return .{ .dtype = dtype, .rank = 2, .shape = shape, .begin = 0, .end = rows * cols * dtype.size() };
}

/// A stored [rows, k] matrix's three entries at `bits`/`group`.
fn stored(rows: usize, k: usize, bits: usize, group: usize) [3]st.Entry {
    return .{ entry(.u32, rows, k * bits / 32), entry(.bf16, rows, k / group), entry(.bf16, rows, k / group) };
}

test "a tensor packed at its override width is admitted and one at a guessed width is refused" {
    const bytes = try overridden(mixed);
    defer a.free(bytes);
    var c = try config.parse(a, bytes);
    defer c.deinit();
    var w = try widths.Widths.init(a, &c);
    defer w.deinit();
    const v = "model.layers.0.self_attn.v_proj"; // its override is spelled under language_model.
    const at_override = stored(8, 128, 8, 64);
    const shape = try w.shaped(v, at_override[0], at_override[1], at_override[2]);
    try std.testing.expectEqual(@as(usize, 128), shape.k);
    try std.testing.expectEqual(@as(usize, 2), shape.groups);
    const at_global = stored(8, 128, 6, 32); // the global width, which a prefix-blind lookup would guess
    try std.testing.expectError(error.InvalidAffineShape, w.shaped(v, at_global[0], at_global[1], at_global[2]));
    // a module with no override is read at the global width only, whatever width its shapes would fit
    try std.testing.expectError(error.InvalidAffineShape, w.shaped("model.layers.0.self_attn.k_proj", at_override[0], at_override[1], at_override[2]));
    try std.testing.expectError(error.UnquantizedAffineTensor, w.shaped("model.layers.0.mlp.gate", at_global[0], at_global[1], at_global[2]));
}
