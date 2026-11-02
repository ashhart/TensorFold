//! BF16 tensor requirements and header-only checkpoint admission for dots.mocr.
const std = @import("std");
const st = @import("core").safetensors;
const cfg = @import("config.zig");
const Allocator = std.mem.Allocator;

pub const Spec = struct {
    name: []const u8,
    shape: [4]usize = @splat(1),
    rank: usize,

    pub fn bytes(s: Spec) usize {
        var n: usize = 2;
        for (s.shape[0..s.rank]) |d| n *= d;
        return n;
    }

    pub fn check(s: Spec, e: st.Entry) !void {
        if (e.dtype != .bf16) return error.DotsDType;
        if (e.rank != s.rank or !std.mem.eql(usize, e.shape[0..e.rank], s.shape[0..s.rank])) return error.DotsShape;
        if (e.end < e.begin or e.end - e.begin != s.bytes()) return error.DotsExtent;
    }
};

/// Names and shapes live in the caller's arena; device loading uses this same inventory.
pub fn inventory(a: Allocator, c: cfg.Config) ![]Spec {
    var out: std.ArrayList(Spec) = .empty;
    const h = c.hidden;
    const k = c.kvWidth();
    const m = c.intermediate;
    try add(a, &out, "model.embed_tokens.weight", &.{ c.vocab, h });
    try add(a, &out, "lm_head.weight", &.{ c.vocab, h });
    try add(a, &out, "model.norm.weight", &.{h});
    for (0..c.layers) |i| {
        const p = try std.fmt.allocPrint(a, "model.layers.{d}.", .{i});
        try sub(a, &out, p, "input_layernorm.weight", &.{h});
        try sub(a, &out, p, "post_attention_layernorm.weight", &.{h});
        inline for (.{ "q", "k", "v" }, .{ h, k, k }) |name, width| {
            try sub(a, &out, p, "self_attn." ++ name ++ "_proj.weight", &.{ width, h });
            try sub(a, &out, p, "self_attn." ++ name ++ "_proj.bias", &.{width});
        }
        try sub(a, &out, p, "self_attn.o_proj.weight", &.{ h, h });
        try sub(a, &out, p, "mlp.gate_proj.weight", &.{ m, h });
        try sub(a, &out, p, "mlp.up_proj.weight", &.{ m, h });
        try sub(a, &out, p, "mlp.down_proj.weight", &.{ h, m });
    }
    const v = c.vision;
    const vh = v.hidden;
    const vm = v.intermediate;
    const merged = v.merged();
    const p = "vision_tower.";
    try sub(a, &out, p, "patch_embed.patchifier.proj.weight", &.{ vh, 3, v.patch, v.patch });
    try sub(a, &out, p, "patch_embed.patchifier.proj.bias", &.{vh});
    try sub(a, &out, p, "patch_embed.patchifier.norm.weight", &.{vh});
    for (0..v.layers) |i| {
        const b = try std.fmt.allocPrint(a, p ++ "blocks.{d}.", .{i});
        try sub(a, &out, b, "norm1.weight", &.{vh});
        try sub(a, &out, b, "norm2.weight", &.{vh});
        try sub(a, &out, b, "attn.qkv.weight", &.{ 3 * vh, vh });
        try sub(a, &out, b, "attn.proj.weight", &.{ vh, vh });
        try sub(a, &out, b, "mlp.fc1.weight", &.{ vm, vh });
        try sub(a, &out, b, "mlp.fc3.weight", &.{ vm, vh });
        try sub(a, &out, b, "mlp.fc2.weight", &.{ vh, vm });
    }
    try sub(a, &out, p, "post_trunk_norm.weight", &.{vh});
    try sub(a, &out, p, "merger.ln_q.weight", &.{vh});
    try sub(a, &out, p, "merger.ln_q.bias", &.{vh});
    try sub(a, &out, p, "merger.mlp.0.weight", &.{ merged, merged });
    try sub(a, &out, p, "merger.mlp.0.bias", &.{merged});
    try sub(a, &out, p, "merger.mlp.2.weight", &.{ h, merged });
    try sub(a, &out, p, "merger.mlp.2.bias", &.{h});
    return out.toOwnedSlice(a);
}

fn add(a: Allocator, out: *std.ArrayList(Spec), name: []const u8, shape: []const usize) !void {
    var s: Spec = .{ .name = name, .rank = shape.len };
    @memcpy(s.shape[0..shape.len], shape);
    try out.append(a, s);
}

fn sub(a: Allocator, out: *std.ArrayList(Spec), prefix: []const u8, suffix: []const u8, shape: []const usize) !void {
    try add(a, out, try std.mem.concat(a, u8, &.{ prefix, suffix }), shape);
}

pub const Summary = struct { tensors: usize, weight_bytes: usize, file_bytes: usize, shards: usize };

pub fn check(a: Allocator, c: cfg.Config, entries: st.Header) !usize {
    const specs = try inventory(a, c);
    if (entries.count() != specs.len) return error.DotsTensorCount;
    var bytes: usize = 0;
    for (specs) |s| {
        try s.check(entries.get(s.name) orelse return error.DotsMissingTensor);
        bytes += s.bytes();
    }
    return bytes;
}

/// Read existing shards' headers only; verify index agreement, complete extents and every served tensor.
pub fn inspect(gpa: Allocator, io: std.Io, dir: []const u8, c: cfg.Config) !Summary {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try cfg.readFile(a, io, dir, "model.safetensors.index.json");
    const root = try std.json.parseFromSliceLeaky(std.json.Value, a, text, .{});
    const map = try st.index.weightMap(root);
    var seen: std.StringHashMapUnmanaged(void) = .empty;
    var entries: st.Header = .empty;
    var file_bytes: usize = 0;
    for (map.values()) |shard| {
        const name = shard.string;
        if (seen.contains(name)) continue;
        try seen.put(a, name, {});
        const path = try std.fs.path.join(a, &.{ dir, name });
        var md = try st.Metadata.read(gpa, io, path);
        defer md.deinit();
        var extent: usize = 0;
        for (md.names.keys(), md.names.values()) |key, e| {
            const target = map.get(key) orelse return error.DotsIndexMismatch;
            if (!std.mem.eql(u8, target.string, name) or entries.contains(key)) return error.DotsIndexMismatch;
            try entries.put(a, try a.dupe(u8, key), e);
            extent = @max(extent, e.end);
        }
        if (md.data + extent != md.file_size) return error.DotsExtent;
        file_bytes = try std.math.add(usize, file_bytes, md.file_size);
    }
    if (entries.count() != map.count()) return error.DotsIndexMismatch;
    const bytes = try check(a, c, entries);
    return .{ .tensors = entries.count(), .weight_bytes = bytes, .file_bytes = file_bytes, .shards = seen.count() };
}
