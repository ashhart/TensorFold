//! Bolt's dense (bf16, no scales) matrices: the struct, the head test and the builder, kept beside the weights.
const mtl = @import("metal");
const cfg = @import("config.zig");
const ckpt = @import("../../core/checkpoint_metal.zig");
const Builder = @import("weights.zig").Builder;

/// A plain bf16 matrix [n, k] a gemv reads in place (nemotron_h_puzzle's latent projections; no scales).
pub const Dense = struct { w: mtl.Buffer, off: usize, n: usize, k: usize };

/// MLX quantizes a module only where {path}.scales exists: Bolt's lm_head stays bf16 and is read as a dense head.
pub fn denseHead(c: cfg.Config, ck: *const ckpt.Checkpoint) bool {
    return c.bolt and !ck.has("lm_head.scales");
}

/// A plain bf16 matrix [n, k] referenced in place (the puzzle latent projections; no scales to read).
pub fn dense(b: *Builder, comptime fmt: []const u8, args: anytype, n: usize, k: usize) !Dense {
    var name: [160]u8 = undefined;
    const full = try std.fmt.bufPrint(&name, fmt, args);
    var scales: [168]u8 = undefined;
    if (b.ck.has(try std.fmt.bufPrint(&scales, "{s}.scales", .{full[0 .. full.len - ".weight".len]}))) {
        b.why.set("{s} is quantized; the Metal dense kernel reads it as bf16 [{d}, {d}] with no scales", .{ full, n, k });
        return error.UnsupportedQuantization;
    }
    if (!b.ck.has(full)) {
        b.why.set("the checkpoint has no {s}, a bf16 [{d}, {d}] the Metal dense kernel reads", .{ full, n, k });
        return error.MissingTensor;
    }
    const t = try b.typed(fmt, args, .bf16, &.{ n, k });
    return .{ .w = t.buffer, .off = t.offset, .n = n, .k = k };
}
