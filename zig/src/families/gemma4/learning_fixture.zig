//! A disposable six-layer group-64 checkpoint exercises local and global tied-value training paths on host instruments.
const std = @import("std");
const core = @import("core");
const n = @import("learning_numeric.zig");
const a = std.testing.allocator;
const Item = struct { name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8 };
const Builder = struct {
    a: std.mem.Allocator,
    items: std.ArrayList(Item) = .empty,
    fn item(b: *Builder, name: []const u8, dtype: []const u8, shape: []const usize, bytes: []const u8) !void {
        try b.items.append(b.a, .{ .name = try b.a.dupe(u8, name), .dtype = dtype, .shape = try b.a.dupe(usize, shape), .bytes = bytes });
    }
    fn vector(b: *Builder, name: []const u8, width: usize) !void {
        const out = try b.a.alloc(u8, width * 2);
        for (0..width) |i| std.mem.writeInt(u16, out[i * 2 ..][0..2], n.bits(if (std.mem.indexOf(u8, name, ".q_norm.") != null or std.mem.indexOf(u8, name, ".k_norm.") != null) 0.0625 + @as(f32, @floatFromInt(i % 5)) / 1024 else 0.875 + @as(f32, @floatFromInt(i % 5)) / 32), .little);
        try b.item(name, "BF16", &.{width}, out);
    }
    fn matrix(b: *Builder, name: []const u8, rows: usize, cols: usize) !void {
        inline for (.{ "weight", "scales", "biases" }, 0..) |part, j| {
            const words = cols / (if (j == 0) @as(usize, 8) else 64);
            const out = try b.a.alloc(u8, rows * words * (if (j == 0) @as(usize, 4) else 2));
            if (j == 0) {
                const seed = std.hash.Wyhash.hash(713, name);
                for (out, 0..) |*v, i| v.* = @truncate((i * 37 + (seed % 65536) * 13) ^ (i >> 3));
            } else for (0..out.len / 2) |i| std.mem.writeInt(u16, out[i * 2 ..][0..2], n.bits(if (j == 1) 0.00390625 else -0.03125), .little);
            try b.item(try std.fmt.allocPrint(b.a, "{s}.{s}", .{ name, part }), if (j == 0) "U32" else "BF16", &.{ rows, words }, out);
        }
    }
    fn write(b: *Builder, dir: std.Io.Dir) !void {
        var header: std.Io.Writer.Allocating = .init(b.a);
        try header.writer.writeAll("{");
        var at: usize = 0;
        for (b.items.items, 0..) |entry, i| {
            try header.writer.print("{s}\"{s}\":{{\"dtype\":\"{s}\",\"shape\":[", .{ if (i == 0) "" else ",", entry.name, entry.dtype });
            for (entry.shape, 0..) |dim, j| try header.writer.print("{s}{d}", .{ if (j == 0) "" else ",", dim });
            try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ at, at + entry.bytes.len });
            at += entry.bytes.len;
        }
        try header.writer.writeAll("}");
        const h = header.written();
        const image = try b.a.alloc(u8, 8 + h.len + at);
        std.mem.writeInt(u64, image[0..8], h.len, .little);
        @memcpy(image[8..][0..h.len], h);
        at = 8 + h.len;
        for (b.items.items) |entry| {
            @memcpy(image[at..][0..entry.bytes.len], entry.bytes);
            at += entry.bytes.len;
        }
        try dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = image });
    }
};
pub const Fixture = struct {
    arena: std.heap.ArenaAllocator,
    tmp: std.testing.TmpDir,
    b: *n.B,
    e: *n.E,
    path: []const u8,
    pub fn open() !Fixture {
        var arena: std.heap.ArenaAllocator = .init(a);
        errdefer arena.deinit();
        const mem = arena.allocator();
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        var builder: Builder = .{ .a = mem };
        try builder.matrix("language_model.model.embed_tokens", 64, 64);
        try builder.vector("language_model.model.norm.weight", 64);
        const shape: @import("learning_profile.zig").Shape = .{ .hidden = 64, .intermediate = 64, .layers = 6, .heads = 2, .sliding_kv = 1, .global_kv = 1, .vocab = 64 };
        for (0..shape.layers) |l| {
            inline for (.{ "input_layernorm.weight", "post_attention_layernorm.weight", "pre_feedforward_layernorm.weight", "post_feedforward_layernorm.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight", "layer_scalar" }, 0..) |suffix, i| {
                try builder.vector(try std.fmt.allocPrint(mem, "language_model.model.layers.{d}.{s}", .{ l, suffix }), if (i == 6) 1 else if (i >= 4) shape.dim(l) else 64);
            }
            inline for (.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" }, 0..) |suffix, i| {
                if (!(l % 6 == 5 and i == 2)) {
                    const rows = if (i == 0) shape.heads * shape.dim(l) else if (i == 1 or i == 2) shape.kv(l) * shape.dim(l) else 64;
                    const cols = if (i == 3) shape.heads * shape.dim(l) else 64;
                    try builder.matrix(try std.fmt.allocPrint(mem, "language_model.model.layers.{d}.{s}", .{ l, suffix }), rows, cols);
                }
            }
        }
        try builder.write(tmp.dir);
        try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"gemma4\",\"text_config\":{\"num_hidden_layers\":6,\"hidden_size\":64,\"intermediate_size\":64},\"quantization\":{\"bits\":4,\"group_size\":64}}" });
        const path = try std.fmt.allocPrint(mem, ".zig-cache/tmp/{s}", .{tmp.sub_path});
        const b = try mem.create(n.B);
        b.* = .{};
        const e = try mem.create(n.E);
        e.* = .{ .dir = path, .r = try n.E.Runtime.init(a, b, shape, std.math.maxInt(u64)) };
        return .{ .arena = arena, .tmp = tmp, .b = b, .e = e, .path = path };
    }
    pub fn deinit(f: *Fixture) void {
        f.e.r.close() catch unreachable;
        std.testing.expectEqual(@as(usize, 0), f.b.held) catch unreachable;
        f.tmp.cleanup();
        f.arena.deinit();
    }
};
