//! Head-local BF16 K and V projections for absorbed Hy4-flash attention.
const std = @import("std");
const core = @import("core");
const loader = @import("loader.zig");

pub const latent = 512;
pub const nope = 192;
pub const rope = 64;
pub const value_width = 256;
pub const query_stride = nope + rope;
pub const absorbed_stride = latent + rope;

pub const HeadRange = struct { first: u32, count: u32 };

/// The checkpoint's own affine kv_b rows for the selected heads (per head
pub const Packed = struct {
    bits: u8,
    words: []u8,
    scales: []u8,
    biases: []u8,

    pub fn deinit(self: Packed, a: std.mem.Allocator) void {
        a.free(self.words);
        a.free(self.scales);
        a.free(self.biases);
    }
};

pub const Weights = struct {
    key: []u16,
    value: []u16,
    packed_rows: ?Packed = null,

    pub fn deinit(self: *Weights, a: std.mem.Allocator) void {
        a.free(self.key);
        a.free(self.value);
        if (self.packed_rows) |p| p.deinit(a);
        self.* = undefined;
    }
};

const per_head = nope + value_width;
const Input = union(enum) { bf16: []const u8, fp32: []const f32 };

fn word(input: Input, at: usize) u16 {
    return switch (input) {
        .bf16 => |bytes| std.mem.readInt(u16, bytes[2 * at ..][0..2], .little),
        .fp32 => |elements| core.quant.convert.bf16(elements[at]),
    };
}

fn split(a: std.mem.Allocator, input: Input, count: usize) !Weights {
    const keys = try a.alloc(u16, count * nope * latent);
    errdefer a.free(keys);
    const values = try a.alloc(u16, count * value_width * latent);
    errdefer a.free(values);
    for (0..count) |head| {
        const key_start = head * per_head * latent;
        const value_start = key_start + nope * latent;
        for (keys[head * nope * latent ..][0 .. nope * latent], 0..) |*item, i| item.* = word(input, key_start + i);
        for (values[head * value_width * latent ..][0 .. value_width * latent], 0..) |*item, i| item.* = word(input, value_start + i);
    }
    return .{ .key = keys, .value = values };
}

/// The owning rank reads only its selected heads. DFly2 reads its own BF16 projection when sharing target KV.
pub fn read(a: std.mem.Allocator, model: *loader.Loaded, kind: loader.Attention, layer: u32, range: HeadRange) !Weights {
    const index = try model.attentionTargetLayer(kind, layer);
    const t = model.config.value.text_config;
    const shape = t.headShape(index);
    if (t.kv_lora_rank != latent or shape.non_rotary != nope or shape.rotary != rope or shape.value != value_width)
        return error.UnsupportedHy4MlaGeometry;
    if (range.count == 0 or range.first > shape.heads or range.count > shape.heads - range.first)
        return error.BadHeadRange;
    var key_buffer: [96]u8 = undefined;
    const prefix = switch (kind) {
        .target => "model.layers",
        .draft => "model.dfly_layers",
    };
    const base = try std.fmt.bufPrint(&key_buffer, "{s}.{d}.self_attn.kv_b_proj", .{ prefix, layer });
    const from = @as(usize, range.first) * per_head;
    const to = from + @as(usize, range.count) * per_head;
    const span: core.quant.Span = .{ .from = from, .to = to };
    const rows = @as(usize, shape.heads) * per_head;
    var name: [108]u8 = undefined;
    const weight_key = try std.fmt.bufPrint(&name, "{s}.weight", .{base});
    switch (kind) {
        .draft => {
            const raw = try model.checkpoint.expect(weight_key, .bf16, &.{ rows, latent });
            const selected = try core.quant.slice.takeRows(a, raw, &.{span});
            return split(a, .{ .bf16 = selected.bytes }, range.count);
        },
        .target => {
            const word_count = latent * @as(usize, model.bits) / 32;
            const words = try model.checkpoint.expect(weight_key, .u32, &.{ rows, word_count });
            const scale_key = try std.fmt.bufPrint(&name, "{s}.scales", .{base});
            const scales = try model.checkpoint.expect(scale_key, .bf16, &.{ rows, latent / 64 });
            const bias_key = try std.fmt.bufPrint(&name, "{s}.biases", .{base});
            const biases = try model.checkpoint.expect(bias_key, .bf16, &.{ rows, latent / 64 });
            const w_rows = try core.quant.slice.takeRows(a, words, &.{span});
            const s_rows = try core.quant.slice.takeRows(a, scales, &.{span});
            const b_rows = try core.quant.slice.takeRows(a, biases, &.{span});
            const dequant = try core.quant.mlx.dequant(a, w_rows, s_rows, b_rows, 64);
            defer a.free(dequant);
            var out = try split(a, .{ .fp32 = dequant }, range.count);
            errdefer out.deinit(a);
            const pw = try a.dupe(u8, w_rows.bytes);
            errdefer a.free(pw);
            const ps = try a.dupe(u8, s_rows.bytes);
            errdefer a.free(ps);
            out.packed_rows = .{ .bits = model.bits, .words = pw, .scales = ps, .biases = try a.dupe(u8, b_rows.bytes) };
            return out;
        },
    }
}
