//! Checked borrowed GGUF source views for Ling-3.0-tiny on CPU; not a serving admission path.
const std = @import("std");
const g = @import("core").gguf;
pub const reference = @import("core").gguf_source_reference;
pub const views = @import("cpu_gguf_views.zig");
pub const names = @import("gguf_names.zig");

fn integer(d: *const g.Document, key: []const u8, expected: u64) !void {
    const v = d.get(key) orelse return error.MissingLingGgufMetadata;
    const n: u64 = switch (v) {
        .uint32 => |x| x,
        .uint64 => |x| x,
        else => return error.BadLingGgufMetadata,
    };
    if (n != expected) return error.UnsupportedLingGgufGeometry;
}
fn metadata(d: *const g.Document) !void {
    const arch = d.get("general.architecture") orelse return error.MissingLingGgufMetadata;
    if (arch != .string or !std.mem.eql(u8, arch.string, "bailingmoe3")) return error.UnsupportedLingGgufArchitecture;
    const checks = .{ .{ "block_count", 24 }, .{ "embedding_length", 1536 }, .{ "feed_forward_length", 4608 }, .{ "attention.head_count", 16 }, .{ "expert_count", 128 }, .{ "expert_used_count", 8 }, .{ "expert_group_count", 8 }, .{ "expert_group_used_count", 4 }, .{ "expert_gating_func", 2 }, .{ "vocab_size", 157184 }, .{ "ssm.conv_kernel", 4 }, .{ "kda.head_dim", 128 }, .{ "attention.q_lora_rank", 256 }, .{ "attention.kv_lora_rank", 512 }, .{ "rope.dimension_count", 64 }, .{ "expert_feed_forward_length", 512 }, .{ "expert_shared_feed_forward_length", 512 }, .{ "leading_dense_block_count", 1 } };
    inline for (checks) |check| try integer(d, "bailingmoe3." ++ check[0], check[1]);
    try integer(d, "bailingmoe3.context_length", 131072);
    try integer(d, "bailingmoe3.attention.key_length_mla", 192);
    try integer(d, "bailingmoe3.attention.value_length_mla", 128);
    try integer(d, "bailingmoe3.expert_shared_count", 1);
    inline for (.{ .{ "rope.freq_base", 6000000.0 }, .{ "attention.layer_norm_rms_epsilon", 0.000001 }, .{ "kda.gate_lower_bound", -5.0 }, .{ "expert_weights_scale", 2.5 } }) |check| {
        const value = d.get("bailingmoe3." ++ check[0]) orelse return error.MissingLingGgufMetadata;
        const actual = try value.float();
        if (!std.math.isFinite(actual) or @as(f32, @floatCast(actual)) != @as(f32, check[1])) return error.UnsupportedLingGgufGeometry;
    }
    inline for (.{ "kda.safe_gate", "expert_weights_norm" }) |key| {
        const value = d.get("bailingmoe3." ++ key) orelse return error.MissingLingGgufMetadata;
        if (value != .boolean or !value.boolean) return error.UnsupportedLingGgufGeometry;
    }
    const kv = d.get("bailingmoe3.attention.head_count_kv") orelse return error.MissingLingGgufMetadata;
    if (kv != .array or kv.array.len != 24) return error.UnsupportedLingGgufGeometry;
    var it = kv.array.iterator();
    for (0..24) |layer| {
        const value = (try it.next()) orelse return error.UnsupportedLingGgufGeometry;
        if (try value.unsigned() != @as(u64, if (layer % 4 == 3) 1 else 0)) return error.UnsupportedLingGgufGeometry;
    }
    const tokens = d.get("tokenizer.ggml.tokens") orelse return error.MissingLingGgufMetadata;
    if (tokens != .array or tokens.array.kind != .string or tokens.array.len != 157184) return error.UnsupportedLingGgufGeometry;
    if (d.get("bailingmoe3.swiglu_clamp_exp") != null or d.get("bailingmoe3.swiglu_clamp_shexp") != null) return error.UnsupportedLingGgufClamp;
}
pub fn checkTensor(t: g.Tensor, n: names.Name) !void {
    if (t.rank != n.rank or !std.mem.eql(u64, t.shape[0..t.rank], n.dims[0..n.rank])) return error.BadLingGgufTensorShape;
    switch (n.role) {
        .positive_exp_decay => {
            if (t.kind != .f32 or t.bytes.len != 64) return error.UnsupportedLingGgufSmallType;
            for (0..16) |i| {
                const rate: f32 = @bitCast(std.mem.readInt(u32, t.bytes[i * 4 ..][0..4], .little));
                if (!std.math.isFinite(rate) or rate <= 0) return error.BadLingGgufDecay;
            }
        },
        .vector, .convolution => if (t.kind != .f32) return error.UnsupportedLingGgufSmallType,
        else => switch (t.kind) {
            .f32, .q4_k, .q6_k, .q5_0 => {},
            else => return error.UnsupportedLingGgufQuant,
        },
    }
}
pub const File = struct {
    source: g.File,
    arena: std.heap.ArenaAllocator,
    plan: []names.Name,
    joined_mla: [24]?[]f32 = @splat(null),
    validated: std.StringHashMapUnmanaged(void) = .empty,
    pub fn open(a: std.mem.Allocator, io: std.Io, path: []const u8) !File {
        var source = try g.File.open(a, io, path, .{});
        errdefer source.close(io);
        try metadata(&source.document);
        var arena = std.heap.ArenaAllocator.init(a);
        errdefer arena.deinit();
        const plan = try names.tiny(arena.allocator());
        if (source.document.tensors.len != plan.len) return error.UnexpectedLingGgufTensor;
        for (plan) |name| try checkTensor(source.document.getTensor(name.source) orelse return error.MissingLingGgufTensor, name);
        return .{ .source = source, .arena = arena, .plan = plan };
    }
    pub fn config(f: *const File) !@import("config.zig").Config {
        return @import("cpu_gguf_config.zig").tiny(&f.source.document);
    }
    pub fn view(f: *const File, key: []const u8) !views.View {
        return views.resolve(&f.source.document, f.plan, key);
    }
    pub fn values(f: *File, a: std.mem.Allocator, key: []const u8, shape: []const usize) ![]f32 {
        const v = try f.view(key);
        if (v.name.role != .mla_key_transposed) return @import("cpu_gguf_values.zig").small(a, v, shape);
        if (!std.mem.eql(usize, shape, &.{ 4096, 512 })) return error.BadLingGgufTensorShape;
        const prefix = "model.layers.";
        if (!std.mem.startsWith(u8, key, prefix)) return error.MissingLingGgufTensor;
        const tail = key[prefix.len..];
        const end = std.mem.indexOfScalar(u8, tail, '.') orelse return error.MissingLingGgufTensor;
        const layer = try std.fmt.parseInt(usize, tail[0..end], 10);
        if (layer >= 24 or layer % 4 != 3) return error.BadLingGgufTensorShape;
        if (f.joined_mla[layer] == null) {
            var buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "blk.{d}.attn_v_b.weight", .{layer});
            const value = f.source.document.getTensor(name) orelse return error.MissingLingGgufTensor;
            f.joined_mla[layer] = try @import("cpu_gguf_values.zig").joinMla(f.arena.child_allocator, v.tensor, value);
        }
        return f.joined_mla[layer].?;
    }
    pub fn project(f: *File, key: []const u8, input: []const f32, output: []f32, count: usize, width: usize, n: usize, rounded: bool, threads: usize) !void {
        const v = try f.view(key);
        if (v.tensor.rank != 2 or v.tensor.shape[0] != width or v.tensor.shape[1] != n) return error.BadLingGgufMatrix;
        if (v.tensor.kind != .f32 and !f.validated.contains(key)) {
            try g.affine.validate(v.tensor.kind, v.tensor.bytes);
            const dup = try f.arena.allocator().dupe(u8, key);
            try f.validated.put(f.arena.allocator(), dup, {});
        }
        try @import("cpu_gguf_rows.zig").projectValidatedOn(v.tensor, input, output, count, rounded, threads);
    }
    pub fn embed(f: *const File, ids: []const u32, output: []f32) !void {
        try @import("cpu_gguf_values.zig").embedding(&f.source.document, ids, output);
    }
    pub fn close(f: *File, io: std.Io) void {
        for (f.joined_mla) |entry| if (entry) |data| f.arena.child_allocator.free(data);
        f.arena.deinit();
        f.source.close(io);
        f.* = undefined;
    }
};

test "Ling GGUF source views refuse wrong dimensions and quantized small tensors" {
    const n = names.Name{ .source = "matrix", .native = "matrix", .role = .matrix, .rank = 2, .dims = .{ 1536, 512, 1 } };
    var t = g.Tensor{ .name = "matrix", .kind = .q4_k, .rank = 2, .shape = .{ 1536, 512, 1, 1 }, .offset = 0, .len = 0 };
    try checkTensor(t, n);
    t.shape[1] = 511;
    try std.testing.expectError(error.BadLingGgufTensorShape, checkTensor(t, n));
    const gain = names.Name{ .source = "gain", .native = "gain", .role = .vector, .rank = 1, .dims = .{ 1536, 1, 1 } };
    t = .{ .name = "gain", .kind = .q4_k, .rank = 1, .shape = .{ 1536, 1, 1, 1 }, .offset = 0, .len = 0 };
    try std.testing.expectError(error.UnsupportedLingGgufSmallType, checkTensor(t, gain));
    t.kind = .f32;
    try checkTensor(t, gain);
}
