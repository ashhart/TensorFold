//! Qwen35 GGUF metadata maps to the dense family's configuration; unrecognized computation settings refuse by key.
const std = @import("std");
const g = @import("core").gguf;
const cfg = @import("config.zig");
pub const Problem = struct {
    bytes: [256]u8 = undefined,
    len: usize = 0,
    pub fn message(p: *const Problem) []const u8 {
        return p.bytes[0..p.len];
    }
    pub fn fail(p: *Problem, comptime fmt: []const u8, args: anytype) error{UnsupportedQwenGguf} {
        const text = std.fmt.bufPrint(&p.bytes, fmt, args) catch "Qwen27 GGUF: invalid setting";
        if (text.ptr != &p.bytes) @memcpy(p.bytes[0..text.len], text);
        p.len = text.len;
        return error.UnsupportedQwenGguf;
    }
};
pub const fields = [_][]const u8{ "block_count", "context_length", "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv", "attention.key_length", "attention.value_length", "rope.dimension_count", "rope.freq_base", "rope.dimension_sections", "attention.layer_norm_rms_epsilon", "ssm.conv_kernel", "ssm.state_size", "ssm.group_count", "ssm.time_step_rank", "ssm.inner_size", "attention.recurrent_layers", "full_attention_interval" };
const Read = struct {
    doc: *const g.Document,
    arch: []const u8,
    p: *Problem,
    fn value(r: Read, key: []const u8) !g.Value {
        return g.views.scoped(r.doc, r.arch, key) orelse r.p.fail("{s}.{s}: missing", .{ r.arch, key });
    }
    fn uint(r: Read, key: []const u8) !usize {
        const v = (try r.value(key)).unsigned() catch return r.p.fail("{s}.{s}: expected positive integer", .{ r.arch, key });
        if (v == 0 or v > std.math.maxInt(u32)) return r.p.fail("{s}.{s}: value {d} is out of range", .{ r.arch, key, v });
        return @intCast(v);
    }
    fn real(r: Read, key: []const u8) !f64 {
        const v = (try r.value(key)).float() catch return r.p.fail("{s}.{s}: expected float", .{ r.arch, key });
        if (!std.math.isFinite(v) or v <= 0) return r.p.fail("{s}.{s}: expected positive finite float", .{ r.arch, key });
        return v;
    }
};
pub fn parse(doc: *const g.Document, p: *Problem) !cfg.Config {
    const architecture = doc.get("general.architecture") orelse return p.fail("general.architecture: missing", .{});
    if (architecture != .string) return p.fail("general.architecture: expected string", .{});
    const arch = architecture.string;
    if (!std.mem.eql(u8, arch, "qwen35")) return p.fail("general.architecture: unsupported {s}", .{arch});
    for (doc.metadata) |entry| {
        if (!std.mem.startsWith(u8, entry.key, arch) or entry.key.len <= arch.len or entry.key[arch.len] != '.') continue;
        const suffix = entry.key[arch.len + 1 ..];
        var known = false;
        for (fields) |field| known = known or std.mem.eql(u8, field, suffix);
        if (!known) return p.fail("{s}: unsupported metadata key", .{entry.key});
    }
    const r = Read{ .doc = doc, .arch = arch, .p = p };
    // Validate typed fields by key before taking the shared config view.
    inline for (.{ "block_count", "context_length", "embedding_length", "feed_forward_length", "attention.head_count", "attention.head_count_kv", "attention.key_length", "attention.value_length", "rope.dimension_count" }) |key| _ = try r.uint(key);
    _ = try r.real("rope.freq_base");
    _ = try r.real("attention.layer_norm_rms_epsilon");
    const view = try doc.config();
    const tok = doc.tokenizer() catch |err| return p.fail("tokenizer.ggml: {s}", .{@errorName(err)});
    const tokens = tok.tokens orelse return p.fail("tokenizer.ggml.tokens: missing", .{});
    if (tok.eos == null) return p.fail("tokenizer.ggml.eos_token_id: missing", .{});
    if (tokens.len == 0 or tokens.len > std.math.maxInt(u32)) return p.fail("tokenizer.ggml.tokens: invalid vocabulary", .{});
    const vh = try r.uint("ssm.time_step_rank");
    const inner = try r.uint("ssm.inner_size");
    if (inner % vh != 0) return p.fail("{s}.ssm.inner_size: not divisible by time_step_rank", .{arch});
    if (view.key_length.? != view.value_length.?) return p.fail("{s}.attention.value_length: differs from key_length", .{arch});
    var c: cfg.Config = .{ .hidden = @intCast(view.embedding.?), .intermediate = @intCast(view.feed_forward.?), .layers = @intCast(view.layers.?), .vocab = tokens.len, .heads = @intCast(view.heads.?), .kv_heads = @intCast(view.kv_heads.?), .head_dim = @intCast(view.key_length.?), .k_heads = try r.uint("ssm.group_count"), .v_heads = vh, .dk = try r.uint("ssm.state_size"), .dv = inner / vh, .conv_kernel = try r.uint("ssm.conv_kernel"), .eps = @floatCast(view.rms_epsilon.?), .rope_dims = @intCast(view.rope_dimensions.?), .rope_theta = view.rope_base.?, .max_position = @intCast(view.context.?) };
    if (c.layers > cfg.max_layers) return p.fail("{s}.block_count: exceeds {d}", .{ arch, cfg.max_layers });
    if (!std.math.isFinite(c.eps) or c.eps <= 0) return p.fail("{s}.attention.layer_norm_rms_epsilon: not representable", .{arch});
    c.eos_count = @intCast(g.stops.read(doc, &.{"<|endoftext|>"}, &c.eos) catch |err| return p.fail("tokenizer.ggml stop set: {s}", .{@errorName(err)}));
    if (g.views.scoped(doc, arch, "rope.dimension_sections")) |value| {
        if (value != .array or value.array.len != 4) return p.fail("{s}.rope.dimension_sections: expected four integers", .{arch});
        var it = value.array.iterator();
        for (0..4) |i| {
            const x = (try it.next()).?.unsigned() catch return p.fail("{s}.rope.dimension_sections: expected integers", .{arch});
            if (x > c.rope_dims or (i == 3 and x != 0)) return p.fail("{s}.rope.dimension_sections: unsupported value {d}", .{ arch, x });
            if (i < 3) c.mrope_section[i] = @intCast(x);
        }
    }
    if (c.mrope_section[0] + c.mrope_section[1] + c.mrope_section[2] != c.rope_dims / 2) return p.fail("{s}.rope.dimension_sections: wrong total", .{arch});
    var interval: ?usize = null;
    if (g.views.scoped(doc, arch, "full_attention_interval") != null) interval = try r.uint("full_attention_interval");
    if (g.views.scoped(doc, arch, "attention.recurrent_layers")) |value| {
        if (value != .array or value.array.kind != .boolean or value.array.len != c.layers) return p.fail("{s}.attention.recurrent_layers: expected block_count booleans", .{arch});
        var it = value.array.iterator();
        for (c.kinds[0..c.layers], 0..) |*kind, i| {
            kind.* = if ((try it.next()).?.boolean) .linear else .attention;
            if (interval) |n| if ((kind.* == .attention) != ((i + 1) % n == 0)) return p.fail("{s}.attention.recurrent_layers: conflicts with full_attention_interval at {d}", .{ arch, i });
        }
    } else {
        const n = interval orelse return p.fail("{s}.full_attention_interval: missing layer pattern", .{arch});
        for (c.kinds[0..c.layers], 0..) |*kind, i| kind.* = if ((i + 1) % n == 0) .attention else .linear;
    }
    inline for (.{ .{ "attention.key_length", c.head_dim, 256 }, .{ "ssm.state_size", c.dk, 128 }, .{ "ssm.inner_size / time_step_rank", c.dv, 128 }, .{ "ssm.conv_kernel", c.conv_kernel, 4 }, .{ "rope.dimension_count", c.rope_dims, 64 } }) |field| {
        if (field[1] != field[2]) return p.fail("{s}.{s}: unsupported value {d}", .{ arch, field[0], field[1] });
    }
    @import("geometry.zig").supported(c) catch |err| return p.fail("{s}: {s} (head dimensions, projection tiles or layer pattern)", .{ arch, @errorName(err) });
    return c;
}
