//! The CUDA tower's checkpoint names, shapes and byte order, shared with its CPU regression fixtures.
const std = @import("std");
const core = @import("core");
const dequant = @import("qwen_cuda_dequant.zig");
const mlx_prefix = "vision_tower.";
const hf_prefix = "model.visual.";
pub const patch = 16;
pub const temporal = 2;
pub const channels = 3;
pub const patch_width = channels * temporal * patch * patch;
pub const grid_side = 48;

/// The actual tower load into a byte sink; `put` finishes using its source before it returns.
pub fn load(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: anytype, sink: anytype) !void {
    var reader = try Reader.open(gpa, io, dir);
    defer core.checkpoint.cache.close(&reader.ck);
    try reader.copy(gpa, c, sink);
}

const Reader = struct {
    ck: core.Checkpoint,
    prefix: []const u8,
    hf: bool,

    fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !Reader {
        var ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, hf_prefix);
        const hf = ck.has(hf_prefix ++ "patch_embed.proj.weight");
        if (!hf) {
            core.checkpoint.cache.close(&ck);
            ck = try core.Checkpoint.openModelPrefix(gpa, io, dir, mlx_prefix);
        }
        return .{ .ck = ck, .prefix = if (hf) hf_prefix else mlx_prefix, .hf = hf };
    }

    fn get(r: *Reader, sink: anytype, name: []const u8, shape: []const usize) !u64 {
        var buf: [96]u8 = undefined;
        const tn = try r.ck.expect(try std.fmt.bufPrint(&buf, "{s}{s}", .{ r.prefix, name }), .bf16, shape);
        return sink.put(tn.bytes);
    }

    /// A linear's weight as bf16: stored so, or ModelOpt's MXFP8 or NVFP4 planes and scales decoded on the host.
    fn linear(r: *Reader, gpa: std.mem.Allocator, sink: anytype, name: []const u8, shape: [2]usize) !u64 {
        const n = shape[0];
        const k = shape[1];
        var buf: [96]u8 = undefined;
        const full = try std.fmt.bufPrint(&buf, "{s}{s}", .{ r.prefix, name });
        const stored = try r.ck.get(full);
        const kind: dequant.Kind = switch (stored.dtype) {
            .f8_e4m3 => .mxfp8,
            .u8 => .nvfp4,
            else => return r.get(sink, name, &shape),
        };
        const width = if (kind == .nvfp4) k / 2 else k;
        const shaped = std.mem.endsWith(u8, full, ".weight") and k % dequant.group(kind) == 0 and stored.is(stored.dtype, &.{ n, width });
        if (!shaped) return refuse(full, kind);
        var sb: [112]u8 = undefined;
        const stem = full[0 .. full.len - ".weight".len];
        const scale = try r.ck.get(try std.fmt.bufPrint(&sb, "{s}.weight_scale", .{stem}));
        const scale_ok = switch (kind) {
            .mxfp8 => scale.is(.u8, &.{ n, k / 32 }) or scale.is(.f8_e8m0, &.{ n, k / 32 }),
            .nvfp4 => scale.is(.f8_e4m3, &.{ n, k / 16 }),
        };
        if (!scale_ok) return refuse(full, kind);
        var global: f32 = 1;
        if (kind == .nvfp4) {
            const g = try r.ck.get(try std.fmt.bufPrint(&sb, "{s}.weight_scale_2", .{stem}));
            if (g.dtype != .f32 or g.bytes.len != 4) return refuse(full, kind);
            global = @bitCast(std.mem.readInt(u32, g.bytes[0..4], .little));
        }
        const activation = try std.fmt.bufPrint(&sb, "{s}.input_scale", .{stem});
        if (r.ck.has(activation)) _ = try r.ck.get(activation); // W4A4's static activation scale; the tower runs bf16
        const out = try gpa.alloc(u16, n * k);
        defer gpa.free(out);
        switch (kind) {
            .nvfp4 => dequant.nvfp4(out, stored.bytes, scale.bytes, global, n, k),
            .mxfp8 => dequant.mxfp8(out, stored.bytes, scale.bytes, n, k),
        }
        return sink.put(std.mem.sliceAsBytes(out));
    }

    fn refuse(name: []const u8, kind: dequant.Kind) error{UnexpectedTensor} {
        std.log.err("{s}: not a ModelOpt {t} vision linear of the tower's shape", .{ name, kind });
        return error.UnexpectedTensor;
    }

    fn copy(r: *Reader, gpa: std.mem.Allocator, c: anytype, sink: anytype) !void {
        const hidden = c.hidden;
        const intermediate = c.intermediate;
        const merged = c.merged();
        if (r.hf) { // HF: [hidden, C, T, P, P] already, uploaded as stored
            const tn = try r.ck.expect(hf_prefix ++ "patch_embed.proj.weight", .bf16, &.{ hidden, channels, temporal, patch, patch });
            sink.patch_w = try sink.put(tn.bytes);
        } else { // [hidden, T, P, P, C] as stored (MLX) to [hidden, C, T, P, P]
            const tn = try r.ck.expect(mlx_prefix ++ "patch_embed.proj.weight", .bf16, &.{ hidden, temporal, patch, patch, channels });
            const host = try gpa.alloc(u16, hidden * patch_width);
            defer gpa.free(host);
            const stored: []align(1) const u16 = std.mem.bytesAsSlice(u16, tn.bytes);
            for (0..hidden) |o| for (0..channels) |ch| for (0..temporal) |tt| for (0..patch) |y| for (0..patch) |x| {
                host[o * patch_width + ((ch * temporal + tt) * patch + y) * patch + x] = stored[(((o * temporal + tt) * patch + y) * patch + x) * channels + ch];
            };
            sink.patch_w = try sink.put(std.mem.sliceAsBytes(host));
        }
        sink.patch_b = try r.get(sink, "patch_embed.proj.bias", &.{hidden});
        sink.pos = try r.get(sink, "pos_embed.weight", &.{ grid_side * grid_side, hidden });
        for (sink.blocks[0..c.depth], 0..) |*b, i| {
            const shapes = .{
                .{ "norm1_w", "norm1.weight", .{hidden} },                        .{ "norm1_b", "norm1.bias", .{hidden} },
                .{ "qkv_w", "attn.qkv.weight", .{ 3 * hidden, hidden } },         .{ "qkv_b", "attn.qkv.bias", .{3 * hidden} },
                .{ "proj_w", "attn.proj.weight", .{ hidden, hidden } },           .{ "proj_b", "attn.proj.bias", .{hidden} },
                .{ "norm2_w", "norm2.weight", .{hidden} },                        .{ "norm2_b", "norm2.bias", .{hidden} },
                .{ "fc1_w", "mlp.linear_fc1.weight", .{ intermediate, hidden } }, .{ "fc1_b", "mlp.linear_fc1.bias", .{intermediate} },
                .{ "fc2_w", "mlp.linear_fc2.weight", .{ hidden, intermediate } }, .{ "fc2_b", "mlp.linear_fc2.bias", .{hidden} },
            };
            inline for (shapes) |sh| {
                var nb: [96]u8 = undefined;
                const shape: [sh[2].len]usize = sh[2];
                const at = try std.fmt.bufPrint(&nb, "blocks.{d}.{s}", .{ i, sh[1] });
                @field(b, sh[0]) = if (shape.len == 2) try r.linear(gpa, sink, at, shape) else try r.get(sink, at, &shape);
            }
        }
        sink.merger_norm_w = try r.get(sink, "merger.norm.weight", &.{hidden});
        sink.merger_norm_b = try r.get(sink, "merger.norm.bias", &.{hidden});
        sink.fc1_w = try r.linear(gpa, sink, "merger.linear_fc1.weight", .{ merged, merged });
        sink.fc1_b = try r.get(sink, "merger.linear_fc1.bias", &.{merged});
        sink.fc2_w = try r.linear(gpa, sink, "merger.linear_fc2.weight", .{ c.out_hidden, merged });
        sink.fc2_b = try r.get(sink, "merger.linear_fc2.bias", &.{c.out_hidden});
        if (r.ck.unused() != 0) return error.UnusedVisionTensors;
    }
};
