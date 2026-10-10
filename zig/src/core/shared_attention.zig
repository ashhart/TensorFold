//! Shared-context attention, per-row RoPE and cache append; byte masks alone decide what each row sees.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Output = enum { f32, bf16 };
pub const Rope = enum { split_half, interleaved };
pub const Shape = struct {
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    rope: Rope = .split_half,
    capacity: usize,
    output: Output = .f32,

    pub fn validate(s: Shape) !void {
        if (s.query_heads == 0 or s.kv_heads == 0 or s.query_heads % s.kv_heads != 0 or
            s.query_heads > 128 or s.head_dim == 0 or s.head_dim > 256 or s.head_dim % 32 != 0 or
            s.rotary_dim > s.head_dim or s.rotary_dim % 2 != 0 or s.capacity == 0 or s.capacity > std.math.maxInt(u32)) return error.InvalidAttentionShape;
        _ = try std.math.mul(usize, s.capacity, try std.math.mul(usize, s.kv_heads, s.head_dim));
    }
};
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
const Args = extern struct { rows: u32, keys: u32 = 0, mask_stride: u32 = 0, heads: u32 = 0, scale_bits: u32 = 0 };

pub fn source(a: std.mem.Allocator, s: Shape) ![]u8 {
    try s.validate();
    return std.fmt.allocPrint(a, "#define TF_QH {d}\n#define TF_KVH {d}\n#define TF_D {d}\n#define TF_RD {d}\n#define TF_INTERLEAVED {d}\n#define TF_OUT {s}\n{s}", .{
        s.query_heads, s.kv_heads, s.head_dim, s.rotary_dim, @intFromBool(s.rope == .interleaved), if (s.output == .f32) "float" else "bfloat", sources.core_shared_attention,
    });
}

pub const Attention = struct {
    shape: Shape,
    attend: mtl.Pipeline,
    decode: mtl.Pipeline,
    rope: mtl.Pipeline,
    append: mtl.Pipeline,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, s: Shape) !Attention {
        const text = try source(a, s);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        const attend = try mtl.Pipeline.init(device, lib, "tf_shared_lane_attention", false);
        errdefer attend.deinit();
        const decode = try mtl.Pipeline.init(device, lib, "tf_core_decode_attention", false);
        errdefer decode.deinit();
        const rope = try mtl.Pipeline.init(device, lib, "tf_lane_rope", false);
        errdefer rope.deinit();
        const append = try mtl.Pipeline.init(device, lib, "tf_lane_append_kv", false);
        errdefer append.deinit();
        if (attend.simdWidth() != 32 or attend.maxThreads() < 32) return error.UnsupportedAttentionPipeline;
        return .{ .shape = s, .attend = attend, .decode = decode, .rope = rope, .append = append };
    }
    pub fn deinit(p: Attention) void {
        p.attend.deinit();
        p.decode.deinit();
        p.rope.deinit();
        p.append.deinit();
    }

    /// Q is rotated, cached K was rotated at append; no causal mask is inferred: mask[row*stride+key] decides.
    pub fn encode(p: Attention, e: mtl.ComputeEncoder, rows: usize, keys: usize, stride: usize, scale: f32, q: Ref, k: Ref, v: Ref, mask: Ref, out: Ref) !void {
        if (rows == 0 or rows > 16 or keys == 0 or keys > p.shape.capacity or stride < keys or stride > std.math.maxInt(u32) or
            !std.math.isFinite(scale)) return error.InvalidAttentionView;
        try check(q, rows * p.shape.query_heads * p.shape.head_dim * 2, 2);
        try check(out, rows * p.shape.query_heads * p.shape.head_dim * p.outputBytes(), p.outputBytes());
        try check(k, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try check(v, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try check(mask, try std.math.mul(usize, rows, stride), 1);
        try disjoint(out, q);
        try disjoint(out, k);
        try disjoint(out, v);
        try disjoint(out, mask);
        e.setPipeline(p.attend);
        for ([_]Ref{ q, k, v, mask }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Args{ .rows = @intCast(rows), .keys = @intCast(keys), .mask_stride = @intCast(stride), .scale_bits = @bitCast(scale) }, 4);
        e.setBuffer(out.buf, out.off, 5);
        e.dispatchGroups(mtl.Size.of(p.shape.query_heads, rows, 1), mtl.Size.of(32, 1, 1));
    }

    fn outputBytes(p: Attention) usize {
        return if (p.shape.output == .f32) 4 else 2;
    }

    /// Ordinary one-sequence decode, all keys visible. Its fp32 arithmetic is exactly the shared row's.
    pub fn ordinary(p: Attention, e: mtl.ComputeEncoder, keys: usize, scale: f32, q: Ref, k: Ref, v: Ref, out: Ref) !void {
        if (keys == 0 or keys > p.shape.capacity or !std.math.isFinite(scale)) return error.InvalidAttentionView;
        try check(q, p.shape.query_heads * p.shape.head_dim * 2, 2);
        try check(out, p.shape.query_heads * p.shape.head_dim * p.outputBytes(), p.outputBytes());
        try check(k, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try check(v, keys * p.shape.kv_heads * p.shape.head_dim * 2, 2);
        try disjoint(out, q);
        try disjoint(out, k);
        try disjoint(out, v);
        e.setPipeline(p.decode);
        for ([_]Ref{ q, k, v }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Args{ .rows = 1, .keys = @intCast(keys), .scale_bits = @bitCast(scale) }, 3);
        e.setBuffer(out.buf, out.off, 4);
        e.dispatchGroups(mtl.Size.of(p.shape.query_heads, 1, 1), mtl.Size.of(32, 1, 1));
    }

    /// positions u32[rows], frequencies f32[rotary_dim/2], storage bf16[rows,heads,D]; prompt chunks too.
    pub fn rotate(p: Attention, e: mtl.ComputeEncoder, rows: usize, heads: usize, x: Ref, positions: Ref, frequencies: Ref, y: Ref) !void {
        if (rows == 0 or rows > p.shape.capacity or (heads != p.shape.query_heads and heads != p.shape.kv_heads)) return error.InvalidRopeView;
        const elements = try std.math.mul(usize, rows, heads * p.shape.head_dim);
        if (elements > std.math.maxInt(u32)) return error.InvalidRopeView;
        try check(x, elements * 2, 2);
        try check(y, elements * 2, 2);
        try check(positions, rows * 4, 4);
        try check(frequencies, @max(p.shape.rotary_dim / 2 * 4, 4), 4);
        try disjoint(x, y);
        try disjoint(y, positions);
        try disjoint(y, frequencies);
        e.setPipeline(p.rope);
        for ([_]Ref{ x, positions, frequencies }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Args{ .rows = @intCast(rows), .heads = @intCast(heads) }, 3);
        e.setBuffer(y.buf, y.off, 4);
        e.dispatchThreads(mtl.Size.of(elements, 1, 1), mtl.Size.of(128, 1, 1));
    }

    /// Prepared K and unmodified V go directly to unique physical slots; caller orders subsequent attention.
    pub fn appendKv(p: Attention, e: mtl.ComputeEncoder, slots: []const u32, k: Ref, v: Ref, cache_k: Ref, cache_v: Ref) !void {
        if (slots.len == 0 or slots.len > 16) return error.InvalidAppendView;
        for (slots, 0..) |slot, i| {
            if (slot >= p.shape.capacity) return error.InvalidAppendSlot;
            for (slots[0..i]) |prior| if (prior == slot) return error.DuplicateAppendSlot;
        }
        const width = p.shape.kv_heads * p.shape.head_dim;
        try check(k, slots.len * width * 2, 2);
        try check(v, slots.len * width * 2, 2);
        try check(cache_k, p.shape.capacity * width * 2, 2);
        try check(cache_v, p.shape.capacity * width * 2, 2);
        try disjoint(k, cache_k);
        try disjoint(v, cache_k);
        try disjoint(k, cache_v);
        try disjoint(v, cache_v);
        try disjoint(cache_k, cache_v);
        e.setPipeline(p.append);
        e.setBuffer(k.buf, k.off, 0);
        e.setBuffer(v.buf, v.off, 1);
        e.setBytes(std.mem.sliceAsBytes(slots), 2);
        e.setValue(Args{ .rows = @intCast(slots.len) }, 3);
        e.setBuffer(cache_k.buf, cache_k.off, 4);
        e.setBuffer(cache_v.buf, cache_v.off, 5);
        e.dispatchThreads(mtl.Size.of(slots.len * width, 1, 1), mtl.Size.of(128, 1, 1));
    }
};

fn check(r: Ref, need: usize, alignment: usize) !void {
    if (r.off % alignment != 0) return error.MisalignedAttentionBuffer;
    const len = r.buf.length();
    if (r.off > len or need > len - r.off) return error.AttentionBufferTooSmall;
}
fn disjoint(a: Ref, b: Ref) !void {
    // Conservative: use separate Metal resources for read/write operations, even at disjoint byte offsets.
    if (a.buf.id == b.buf.id) return error.AliasedAttentionBuffer;
}

test "attention shapes keep grouped-query and rotary dimensions explicit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const good = Shape{ .query_heads = 4, .kv_heads = 2, .head_dim = 64, .rotary_dim = 32, .capacity = 128 };
    const text = try source(arena.allocator(), good);
    try std.testing.expect(std.mem.indexOf(u8, text, "#define TF_QH 4\n#define TF_KVH 2\n") != null);
    var bad = good;
    bad.query_heads = 3;
    try std.testing.expectError(error.InvalidAttentionShape, bad.validate());
    bad = good;
    bad.rotary_dim = 33;
    try std.testing.expectError(error.InvalidAttentionShape, bad.validate());
    bad = good;
    bad.head_dim = 31;
    try std.testing.expectError(error.InvalidAttentionShape, bad.validate());
}
