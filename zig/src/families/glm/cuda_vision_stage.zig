//! The GLM vision tower's host-resident weights staged a unit at a time into one device slot, in the pass's order.
const std = @import("std");
const cuda = @import("cuda");

/// A tensor the tower placed in host memory: its name after the tower's prefix, mapped address and length.
pub const Entry = struct { name: []const u8, at: u64, len: usize };
const Copy = struct { src: u64, dst: u64, len: usize };

/// A pass's units: the patch embedding, each block, the post-norm and downsample, the merger's projection, its MLP.
pub fn count(depth: usize) usize {
    return depth + 4;
}

/// The unit a tensor belongs to; a name no pass reads stays with the patch embedding.
pub fn unitOf(name: []const u8, depth: usize) usize {
    if (std.mem.startsWith(u8, name, "blocks.")) {
        const rest = name["blocks.".len..];
        const end = std.mem.indexOfScalar(u8, rest, '.') orelse return 0;
        const i = std.fmt.parseInt(usize, rest[0..end], 10) catch return 0;
        return if (i < depth) 1 + i else 0;
    }
    if (std.mem.startsWith(u8, name, "post_layernorm.") or std.mem.startsWith(u8, name, "downsample.")) return depth + 1;
    if (std.mem.startsWith(u8, name, "merger.proj.") or std.mem.startsWith(u8, name, "merger.post_projection_norm."))
        return depth + 2;
    if (std.mem.startsWith(u8, name, "merger.")) return depth + 3;
    return 0;
}

pub const Stage = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    slot: cuda.DeviceBuffer,
    units: [][]Copy,

    /// A slot as large as the largest unit; each entry's address in `names` becomes its place in the slot.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, entries: []const Entry, names: *std.StringHashMap(u64), depth: usize) !Stage {
        const n = count(depth);
        const sizes = try gpa.alloc(u64, n);
        defer gpa.free(sizes);
        const counts = try gpa.alloc(usize, n);
        defer gpa.free(counts);
        @memset(sizes, 0);
        @memset(counts, 0);
        for (entries) |e| {
            const u = unitOf(e.name, depth);
            sizes[u] += std.mem.alignForward(u64, e.len, 256);
            counts[u] += 1;
        }
        var s: Stage = .{ .gpa = gpa, .d = d, .slot = try cuda.DeviceBuffer.alloc(d, std.mem.max(u64, sizes)), .units = try gpa.alloc([]Copy, n) };
        for (s.units) |*u| u.* = &.{};
        errdefer s.deinit();
        for (s.units, counts) |*u, c| u.* = try gpa.alloc(Copy, c);
        @memset(sizes, 0);
        @memset(counts, 0);
        for (entries) |e| {
            const u = unitOf(e.name, depth);
            const dst = s.slot.ptr + sizes[u];
            s.units[u][counts[u]] = .{ .src = e.at, .dst = dst, .len = e.len };
            try names.put(e.name, dst);
            sizes[u] += std.mem.alignForward(u64, e.len, 256);
            counts[u] += 1;
        }
        return s;
    }

    pub fn deinit(s: *Stage) void {
        for (s.units) |u| s.gpa.free(u);
        s.gpa.free(s.units);
        s.slot.free();
    }

    /// Enqueue unit `u`'s copies on `stream`: ordered after the work that read the slot before, and before its own.
    pub fn bring(s: *const Stage, stream: cuda.Stream, u: usize) !void {
        for (s.units[u]) |c| try s.d.check(s.d.api.cuMemcpyDtoDAsync_v2(c.dst, c.src, c.len, stream.handle), "cuMemcpyDtoDAsync");
    }
};

test "vision units follow a pass: embedding, each block, post-norm and downsample, merger projection, merger MLP" {
    try std.testing.expectEqual(@as(usize, 0), unitOf("patch_embed.proj.weight", 24));
    try std.testing.expectEqual(@as(usize, 1), unitOf("blocks.0.attn.qkv.weight", 24));
    try std.testing.expectEqual(@as(usize, 24), unitOf("blocks.23.mlp.down_proj.bias", 24));
    try std.testing.expectEqual(@as(usize, 25), unitOf("downsample.weight", 24));
    try std.testing.expectEqual(@as(usize, 25), unitOf("post_layernorm.weight", 24));
    try std.testing.expectEqual(@as(usize, 26), unitOf("merger.proj.weight", 24));
    try std.testing.expectEqual(@as(usize, 26), unitOf("merger.post_projection_norm.bias", 24));
    try std.testing.expectEqual(@as(usize, 27), unitOf("merger.gate_proj.weight", 24));
    try std.testing.expectEqual(@as(usize, 0), unitOf("blocks.99.norm1.weight", 24));
    try std.testing.expectEqual(@as(usize, 28), count(24));
}
